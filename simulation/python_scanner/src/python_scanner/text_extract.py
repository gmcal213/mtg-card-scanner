import cv2
import os
import easyocr
import warnings

def text_extract(input_image):

    # Perform simple binary thresholding
    ret,thresh1 = cv2.threshold(input_image,127,255,cv2.THRESH_BINARY)

    output_dir = os.getcwd()
    output_path = os.path.join(output_dir, "title_thresh.png")
    cv2.imwrite(output_path, thresh1)

    warnings.filterwarnings(
        "ignore",
        message=".*pin_memory.*"
    )

    # initialize easy ocr module
    reader = easyocr.Reader(['en'], gpu=False) # english

    result = reader.readtext(thresh1)

    print(f"Title is: {result[0][1]}")