import os
import cv2
import numpy as np

def line_detect(edge_map):
    # Hough line transform
    lines = cv2.HoughLinesP(edge_map, 1, np.pi / 180, 100, 1000, 10)

    edge_color = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(edge_color, (x1, y1), (x2, y2), (0, 255, 0), 2)

    output_dir = os.getcwd()
    output_path = os.path.join(output_dir, "line_detect.png")
    cv2.imwrite(output_path, edge_color)