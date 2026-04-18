import cv2
import os

def edge_detect(input_image):   
    ddepth = cv2.CV_16S
    # X gradient
    grad_x = cv2.Sobel(input_image, ddepth, 1, 0)
    # Y gradient
    grad_y = cv2.Sobel(input_image, ddepth, 0, 1)

    # absolute value
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    # Approximate gradient magnitude
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    # threshold weak edges
    thresh = grad.max() * 0.3 # Set threshold
    thresh, edge_map = cv2.threshold(grad, thresh, 255, cv2.THRESH_BINARY)

    output_dir = os.getcwd()
    output_path = os.path.join(output_dir, "edge_detect.png")
    cv2.imwrite(output_path, edge_map)

    return edge_map