# Main Entry Point for Scanner app
import argparse
import os
from pathlib import Path
from .preprocess import preprocess
from .edge_detect import edge_detect
from .contour_find import contour_find
from .title_extract import title_extract
from .text_extract import text_extract

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

    print(f"Running Preprocessing")
    image_preprocess = preprocess(input_file)

    print(f"Running Edge Detector")
    edge_detections = edge_detect(image_preprocess)

    print(f"Finding Bounding Rectangle")
    rect = contour_find(edge_detections)

    print(f"Extracting Card and Title from Bounding Rectangle")
    title_image = title_extract(image_preprocess, rect)

    print(f"Running OCR Pipeline on Title Image")
    title = text_extract(title_image)


if __name__ == "__main__":
    main()