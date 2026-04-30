import os
import easyocr
import warnings

def text_extract(input_image, debug):

    warnings.filterwarnings(
        "ignore",
        message=".*pin_memory.*"
    )

    # initialize easy ocr module
    reader = easyocr.Reader(['en'], gpu=False) # english

    result = reader.readtext(input_image)

    print(f"Title is: {result[0][1]}")