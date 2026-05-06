import easyocr
import warnings

# initialize global easy ocr module
_reader = None # <- no eager initialization

# intialize first time called, otherwise return reader
def get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['en'], gpu=False)
    return _reader

# Returns title and confidence score, respectively
def text_extract(input_image, debug):

    if input_image.shape[0] < 10 or input_image.shape[1] < 10:
        return "", 0

    warnings.filterwarnings(
        "ignore",
        message=".*pin_memory.*"
    )

    # get reader
    reader = get_reader()

    # Run OCR image
    result = reader.readtext(input_image)

    if not result:
        return "", 0
    else: # filter by most confident result
        best = max(result, key=lambda x: x[2])

        return best[1], best[2]
