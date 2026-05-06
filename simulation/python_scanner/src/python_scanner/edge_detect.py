import cv2
import os
import numpy as np

def zero_crossing(log_img, threshold=0):
    rows, cols = log_img.shape
    zc = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(1, rows-1):
        for j in range(1, cols-1):

            patch = log_img[i-1:i+2, j-1:j+2]
            p = log_img[i, j]

            # check if sign change exists in neighborhood
            if (p > 0 and np.any(patch < 0)) or (p < 0 and np.any(patch > 0)):

                # optional noise threshold
                if np.max(patch) - np.min(patch) > threshold:
                    zc[i, j] = 255

    return zc

def edge_detect(input_image, edge_method, debug):

    valid_methods = {"sobel", "canny", "log"}

    if edge_method not in valid_methods:
        raise ValueError(
            f"Invalid edge_method '{edge_method}'. "
            f"Choose from {valid_methods}."
        )

    # Convert to grayscale
    image_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Apply selected edge detection method
    if edge_method == "sobel":
        # Sobel Edge detector

        # Gaussian blur
        image_blur = cv2.GaussianBlur(image_gray, (5, 5), 0)

        ddepth = cv2.CV_16S
        # X gradient
        grad_x = cv2.Sobel(image_blur, ddepth, 1, 0)
        # Y gradient
        grad_y = cv2.Sobel(image_blur, ddepth, 0, 1)

        # absolute value
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        # Approximate gradient magnitude
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        # threshold weak edges
        thresh = grad.max() * 0.3 # Set threshold
        _, edge_map = cv2.threshold(grad, thresh, 255, cv2.THRESH_BINARY)

    elif edge_method == "canny":
        # Canny Edge Detector
        edge_map = cv2.Canny(
            image_gray,
            threshold1=100,   # lower threshold
            threshold2=200   # upper threshold
        )

    elif edge_method == "log":
        # Laplacian of Gaussian
        image_blur = cv2.GaussianBlur(image_gray, (5,5), 0)
        log = cv2.Laplacian(image_blur, cv2.CV_64F)

        # Find zero crozzings with thresholding
        edge_map = zero_crossing(log, 20)

    if debug:
        debug_dir = os.path.join(os.getcwd(), "debug")
        os.makedirs(debug_dir, exist_ok=True)

        output_path = os.path.join(debug_dir, "edge.png")
        cv2.imwrite(output_path, edge_map)

    return edge_map