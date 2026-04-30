import cv2
import os

def contour_find(edge_map, debug):

    # Convert grayscale to color for visualization
    output = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)

    contours, hierarchy = cv2.findContours(
    edge_map,
    cv2.RETR_EXTERNAL,   # or cv2.RETR_TREE
    cv2.CHAIN_APPROX_SIMPLE
    )

    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = box.astype(int)

    cv2.drawContours(output, [box], 0, (0, 255, 0), 2)

    if debug:
        output_dir = os.getcwd()
        output_path = os.path.join(output_dir, "contour_detect.png")
        cv2.imwrite(output_path, output)

    return rect