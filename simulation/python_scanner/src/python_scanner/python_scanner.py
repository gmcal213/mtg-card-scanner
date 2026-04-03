# Main Entry Point for Scanner app
import argparse
import os
from pathlib import Path
from .preprocess import preprocess

def main():
    parser = argparse.ArgumentParser(description="Scan an image for an MTG Card")

    parser.add_argument("input_image", help="Path to Image")
    parser.add_argument("-out", "-o", help="Output file location", default=os.getcwd())
    args = parser.parse_args()

    input_file = Path(args.input_image)

    print(f"Running your app")

    print(f"Making sure its a valid file")
    if not input_file.exists() or not input_file.is_file():
        print(f"File {input_file} does not exist.")
        return None

    print(f"Running Preprocessing Pipeline")
    preprocess(input_file, args.out)

    print(f"detecting ")
    

if __name__ == "__main__":
    main()