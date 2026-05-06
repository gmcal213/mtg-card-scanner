import cv2
import os

# Finds contours and extracts rectangle
def contour_find(edge_map, debug):

    contours, _ = cv2.findContours(
    edge_map,
    cv2.RETR_EXTERNAL,   # or cv2.RETR_TREE
    cv2.CHAIN_APPROX_SIMPLE
    )

    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    if not contours:
        raise ValueError("No contours found")

    # minimum bounding rectangle
    rect = cv2.minAreaRect(largest_contour)

    if debug:
        # Convert grayscale to color for visualization
        output = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)
        box = cv2.boxPoints(rect)
        box = box.astype(int)

        cv2.drawContours(output, [box], 0, (0, 255, 0), 2)

        debug_dir = os.path.join(os.getcwd(), "debug")
        os.makedirs(debug_dir, exist_ok=True)

        output_path = os.path.join(debug_dir, "contour.png")
        cv2.imwrite(output_path, output)

    return rect