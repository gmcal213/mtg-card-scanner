# file_format.py for doing iomage preprocessing
import cv2
import os
import numpy as np

def preprocess(input_image, blur_method, thresh_method, debug):

    valid_blur_methods = {"gaussian", "bilateral", "clahe"}

    if blur_method not in valid_blur_methods:
        raise ValueError(
            f"Invalid blur_method '{blur_method}'. "
            f"Choose from {valid_blur_methods}."
        )
    
    valid_thresh_methods = {"global", "otsu", "adaptive"}

    if thresh_method not in valid_thresh_methods:
        raise ValueError(
            f"Invalid thresh_method '{thresh_method}'. "
            f"Choose from {valid_thresh_methods}."
        )
    
    if input_image is None or input_image.size == 0:
        raise ValueError("Empty image passed to preprocess")

    # Convert to grayscale
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Upscale
    scale = 4  

    upscaled = cv2.resize(
        gray,
        (gray.shape[1] * scale, gray.shape[0] * scale),
        interpolation=cv2.INTER_CUBIC
    )

    # Apply selected blur method
    if blur_method == "gaussian":
        # Gaussian blur
        image_blur = cv2.GaussianBlur(upscaled, (7, 7), 0)
    elif blur_method == "bilateral":
        # bilateral fliter
        image_blur= cv2.bilateralFilter(
                        upscaled,
                        d=7,              # diameter of pixel neighborhood
                        sigmaColor=50,    # how much colors can differ
                        sigmaSpace=50     # how far pixels influence each other
                    )
    elif blur_method == "clahe":
        # Contrast-Limited Adaptive Histogram Equalization
        # Create CLAHE object
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )
        image_blur = clahe.apply(upscaled)

    if debug:
        debug_dir = os.path.join(os.getcwd(), "debug")
        os.makedirs(debug_dir, exist_ok=True)

        output_path = os.path.join(debug_dir, "title_preprocess.png")
        cv2.imwrite(output_path, image_blur)

    # Apply selected thresholding method
    if thresh_method == "global":
        # compute mean intensity
        thresh_val = np.mean(image_blur)

        _, th = cv2.threshold(image_blur, thresh_val, 255, cv2.THRESH_BINARY)
    elif thresh_method == "otsu":
        # otsu's method
        _, th = cv2.threshold(
            image_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    elif thresh_method == "adaptive":
        # Adaptive threshold the preprocessed image
        th = cv2.adaptiveThreshold(
            image_blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 10
        )

    if debug:
        debug_dir = os.path.join(os.getcwd(), "debug")
        os.makedirs(debug_dir, exist_ok=True)

        output_path = os.path.join(debug_dir, "title_threshold.png")
        cv2.imwrite(output_path, th)


    return th