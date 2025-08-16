# src/kenazlbm/cli.py
import argparse
from .core import preprocess  # example function in your package

def main():
    parser = argparse.ArgumentParser(description="KenazLBM CLI")
    parser.add_argument("--input", type=str, help="Input file path")
    args = parser.parse_args()

    # call your code
    preprocess(args.input)
