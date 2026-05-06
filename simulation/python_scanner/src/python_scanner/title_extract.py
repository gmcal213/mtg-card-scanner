import cv2
import os

def title_extract(input_image, rect, debug):

    # Get stats on rectangle
    (center, (w, h), angle) = rect

    if w < 10 or h < 10:
        raise ValueError(f"Invalid rect size: w={w}, h={h}")

    # enforce vertical orientation
    if w > h:
        angle += 90
        w, h = h, w

    # Apply rotation
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]))
    if debug:
        debug_dir = os.path.join(os.getcwd(), "debug")
        os.makedirs(debug_dir, exist_ok=True)

        output_path = os.path.join(debug_dir, "rotated.png")
        cv2.imwrite(output_path, rotated)

    # Crop Rectangle
    cropped = cv2.getRectSubPix(rotated, (int(w), int(h)), center)

    if cropped is None or cropped.size == 0:
        raise ValueError("Empty cropped image")

    if debug:
        debug_dir = os.path.join(os.getcwd(), "debug")
        os.makedirs(debug_dir, exist_ok=True)

        output_path = os.path.join(debug_dir, "cropped.png")
        cv2.imwrite(output_path, cropped)

    # Extract Title
    title_region = cropped[
        int(0 * h):int(0.12 * h),
        int(0.02 * w):int(0.75 * w)
    ]   

    if title_region is None or title_region.size == 0:
        raise ValueError("Empty title region")

    if debug:
        debug_dir = os.path.join(os.getcwd(), "debug")
        os.makedirs(debug_dir, exist_ok=True)

        output_path = os.path.join(debug_dir, "title.png")
        cv2.imwrite(output_path, title_region)

    return title_region