#!/opt/homebrew/bin/python3
"""
Name: main.py
Purpose: Transcribes MP3 files into MIDI files through the PyTorch implementation of MT3
"""

__author__ = "Ojas Chaturvedi"
__github__ = "github.com/ojas-chaturvedi"
__license__ = "MIT"

import argparse
from inference import InferenceHandler

def main():
    parser = argparse.ArgumentParser(description="Transcribe MP3 files into MIDI using MT3")
    parser.add_argument('-i', '--input', required=True, help='Path to input MP3 file')
    parser.add_argument('-o', '--output', required=True, help='Path to output MIDI file')
    args = parser.parse_args()

    handler = InferenceHandler('./pretrained')
    handler.inference(args.input, args.output)

if __name__ == "__main__":
    main()
