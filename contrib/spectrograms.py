# Copyright 2022 The MT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Audio spectrogram functions."""

import dataclasses

import tensorflow as tf

# The spectrogram front end is cheap and runs on CPU; keep TensorFlow off the
# GPU entirely so it never grabs device memory that PyTorch (which runs the
# actual model) needs. Must happen before any GPU is initialized.
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

# The mel-spectrogram front end below was previously imported from
# `ddsp.spectral_ops`. The `ddsp` pip package drags in the jax/flax/t5x stack,
# which makes the conda environment unsolvable on the cluster. Since ddsp's
# compute_logmel is itself implemented purely with tf.signal ops (which this
# file already depends on), we vendor those few functions verbatim from
# ddsp 3.3.4 (spectral_ops.py / core.py). The math and ops are identical, so the
# model receives bit-for-bit the same input features -- we just drop the heavy,
# unsolvable dependency.


def _tf_float32(x):
    """Ensure array/tensor is a float32 tf.Tensor (ddsp.core.tf_float32)."""
    if isinstance(x, tf.Tensor):
        return tf.cast(x, dtype=tf.float32)
    return tf.convert_to_tensor(x, tf.float32)


def _stft(audio, frame_size=2048, overlap=0.75, pad_end=True):
    """Differentiable stft, computed in batch (ddsp.spectral_ops.stft)."""
    audio = _tf_float32(audio)
    if len(audio.shape) == 3:
        audio = tf.squeeze(audio, axis=-1)
    return tf.signal.stft(
        signals=audio,
        frame_length=int(frame_size),
        frame_step=int(frame_size * (1.0 - overlap)),
        fft_length=None,  # Use enclosing power of 2.
        pad_end=pad_end)


def _compute_mag(audio, size=2048, overlap=0.75, pad_end=True):
    """ddsp.spectral_ops.compute_mag."""
    mag = tf.abs(_stft(audio, frame_size=size, overlap=overlap, pad_end=pad_end))
    return _tf_float32(mag)


def _compute_mel(audio,
                 lo_hz=0.0,
                 hi_hz=8000.0,
                 bins=64,
                 fft_size=2048,
                 overlap=0.75,
                 pad_end=True,
                 sample_rate=16000):
    """Calculate Mel Spectrogram (ddsp.spectral_ops.compute_mel)."""
    mag = _compute_mag(audio, fft_size, overlap, pad_end)
    num_spectrogram_bins = int(mag.shape[-1])
    linear_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        bins, num_spectrogram_bins, sample_rate, lo_hz, hi_hz)
    mel = tf.tensordot(mag, linear_to_mel_matrix, 1)
    mel.set_shape(mag.shape[:-1].concatenate(linear_to_mel_matrix.shape[-1:]))
    return mel


def _safe_log(x, eps=1e-5):
    """Avoid taking the log of a non-positive number (ddsp.core.safe_log)."""
    safe_x = tf.where(x <= 0.0, eps, x)
    return tf.math.log(safe_x)


def _compute_logmel(audio,
                    lo_hz=80.0,
                    hi_hz=7600.0,
                    bins=64,
                    fft_size=2048,
                    overlap=0.75,
                    pad_end=True,
                    sample_rate=16000):
    """Logarithmic amplitude of mel-scaled spectrogram (ddsp.spectral_ops.compute_logmel)."""
    mel = _compute_mel(audio, lo_hz, hi_hz, bins,
                       fft_size, overlap, pad_end, sample_rate)
    return _safe_log(mel)


# defaults for spectrogram config
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_HOP_WIDTH = 128
DEFAULT_NUM_MEL_BINS = 512

# fixed constants; add these to SpectrogramConfig before changing
FFT_SIZE = 2048
MEL_LO_HZ = 20.0


@dataclasses.dataclass
class SpectrogramConfig:
    """Spectrogram configuration parameters."""
    sample_rate: int = DEFAULT_SAMPLE_RATE
    hop_width: int = DEFAULT_HOP_WIDTH
    num_mel_bins: int = DEFAULT_NUM_MEL_BINS

    @property
    def abbrev_str(self):
        s = ''
        if self.sample_rate != DEFAULT_SAMPLE_RATE:
            s += 'sr%d' % self.sample_rate
        if self.hop_width != DEFAULT_HOP_WIDTH:
            s += 'hw%d' % self.hop_width
        if self.num_mel_bins != DEFAULT_NUM_MEL_BINS:
            s += 'mb%d' % self.num_mel_bins
        return s

    @property
    def frames_per_second(self):
        return self.sample_rate / self.hop_width


def split_audio(samples, spectrogram_config):
    """Split audio into frames."""
    return tf.signal.frame(
        samples,
        frame_length=spectrogram_config.hop_width,
        frame_step=spectrogram_config.hop_width,
        pad_end=True)


def compute_spectrogram(samples, spectrogram_config):
    """Compute a mel spectrogram."""
    overlap = 1 - (spectrogram_config.hop_width / FFT_SIZE)
    return _compute_logmel(
        samples,
        bins=spectrogram_config.num_mel_bins,
        lo_hz=MEL_LO_HZ,
        overlap=overlap,
        fft_size=FFT_SIZE,
        sample_rate=spectrogram_config.sample_rate)


def flatten_frames(frames):
    """Convert frames back into a flat array of samples."""
    return tf.reshape(frames, [-1])


def input_depth(spectrogram_config):
    return spectrogram_config.num_mel_bins
