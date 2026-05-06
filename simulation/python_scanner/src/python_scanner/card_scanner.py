import argparse
import cv2
import numpy as np
from pathlib import Path
from .preprocess import preprocess
from .edge_detect import edge_detect
from .contour_find import contour_find
from .title_extract import title_extract
from .text_extract import text_extract

# Actually apply the OCR pipeline
def scan_card(image, blur, edge, thresh, debug=False):

    try:
        # detect edges
        edge_detections = edge_detect(image, edge, debug)

        # find largest contour -> rectangle
        rect = contour_find(edge_detections, debug)

        # crop title section of image
        title_image = title_extract(image, rect, debug)

        # preprocess title section
        title_processed = preprocess(title_image, blur, thresh, debug)

        # Run OCR engine on title
        title, confidence = text_extract(title_processed, debug)

        return title, confidence
    except Exception as e:
        if debug:
            print(f"[scan_card ERROR] {e}")
        return "", 0
    

# Main Entry Point for Scanner app
def main():
    parser = argparse.ArgumentParser(description="Scan an image for an MTG Card")
    parser.add_argument("input_image", help="Image Path")
    parser.add_argument("--blur", "-b", help="Blurring Method", default="gaussian")
    parser.add_argument("--edge", "-e", help="Edge Detection Method", default="sobel")
    parser.add_argument("--thresh", "-t", help="Threshold Detecting Method", default="global")
    parser.add_argument("--debug", "-d", help="Print extra debug information", action="store_true")
    args = parser.parse_args()
    blur = args.blur
    edge = args.edge
    thresh = args.thresh
    debug = args.debug

    # parse input arguments
    input_path = Path(args.input_image)

    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(f"{input_path} does not exist")

    # read in input image
    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError(f"Failed to load image from {input_path}")

    # run card scanner program
    title, confidence = scan_card(image, blur, edge, thresh, debug)

    if title == "":
        print("Title not found")
    else:
        print(f"Card title is {title} with confidence {confidence}")


if __name__ == "__main__":
    main()