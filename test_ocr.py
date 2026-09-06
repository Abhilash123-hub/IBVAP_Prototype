import cv2
from ocr_module import OCRModule

# Load a test image with a visible license plate
img = cv2.imread('test_plate.jpg')  # Use any frame from your video
ocr = OCRModule()

# Test reading
result = ocr.read_plate(img)
print(f"Detected plate: {result}")