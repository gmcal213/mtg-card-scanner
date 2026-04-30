import cv2
import os

def title_extract(input_image, rect, debug):

    # Get stats on rectangle
    (center, (w, h), angle) = rect

    # enforce vertical orientation
    if w > h:
        angle += 90
        w, h = h, w

    # Apply rotation
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]))
    if debug:
        output_dir = os.getcwd()
        output_path = os.path.join(output_dir, "rotated.png")
        cv2.imwrite(output_path, rotated)

    # Crop Rectangle
    cropped = cv2.getRectSubPix(rotated, (int(w), int(h)), center)

    if debug:
        output_path = os.path.join(output_dir, "cropped.png")
        cv2.imwrite(output_path, cropped)

    # Extract Title
    title_region = cropped[
        int(0.05 * h):int(0.12 * h),
        int(0.06 * w):int(0.94 * w)
    ]   

    # Upscale and sharpen
    scale = 3  # or 2–4 depending on your needs

    upscaled = cv2.resize(
        title_region,
        (title_region.shape[1] * scale, title_region.shape[0] * scale),
        interpolation=cv2.INTER_LANCZOS4
    )

    if debug:
        output_path = os.path.join(output_dir, "title.png")
        cv2.imwrite(output_path, upscaled)

    return upscaled