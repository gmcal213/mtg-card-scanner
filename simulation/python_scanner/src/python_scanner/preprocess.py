# file_format.py for doing iomage preprocessing
import cv2
import os

def preprocess(input_file, output_file):
    # Read in image
    image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    image_blur = cv2.GaussianBlur(image, (5, 5), 0)


    output_dir = output_file  # or args.output
    output_path = os.path.join(output_dir, "output_image.png")
    cv2.imwrite(output_path, image)

    output_path = os.path.join(output_dir, "output_image_blur.png")
    cv2.imwrite(output_path, image_blur)

    output_path = os.path.join(output_dir, "output_image_contrast.png")
    cv2.imwrite(output_path, image_contrast)