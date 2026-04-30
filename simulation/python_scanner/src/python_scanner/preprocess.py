# file_format.py for doing iomage preprocessing
import cv2
import os

def preprocess(input_image, debug):
    # Read in image
    image = cv2.imread(input_image)

    # Convert to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    image_blur = cv2.GaussianBlur(image_gray, (3, 3), 0)

    if debug:
        output_dir = os.getcwd()
        output_path = os.path.join(output_dir, "preprocess.png")
        cv2.imwrite(output_path, image_blur)


    return image_blur