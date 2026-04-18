import cv2
import os

def title_extract(input_image, rect):

    # Get stats on rectangle
    (center, (w, h), angle) = rect

    # enforce vertical orientation
    if w > h:
        angle += 90
        w, h = h, w

    # Apply rotation
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]))
    output_dir = os.getcwd()
    output_path = os.path.join(output_dir, "rotated.png")
    cv2.imwrite(output_path, rotated)

    # Crop Rectangle
    cropped = cv2.getRectSubPix(rotated, (int(w), int(h)), center)
    output_path = os.path.join(output_dir, "cropped.png")
    cv2.imwrite(output_path, cropped)

    # Extract title
    h, w = cropped.shape[:2]
    title_region = cropped[
        int(0.05 * h):int(0.10 * h),
        int(0.06 * w):int(0.94 * w)
    ]   

    output_path = os.path.join(output_dir, "title.png")
    cv2.imwrite(output_path, title_region)

    return title_region