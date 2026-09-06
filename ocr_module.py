"""
Automatic Number Plate Recognition (ANPR) Engine (IBVAP)
Extracts, enhances, and reads license plates from vehicle surveillance streams
with Indian license plate validation and normalization.
"""

import re
import cv2
import numpy as np
import easyocr
from thefuzz import fuzz


class OCRModule:
    def __init__(self, use_gpu=False):
        print("Initializing IBVAP EasyOCR ANPR Engine...")
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)
        # Indian license plate standard patterns: e.g. HR26DK8901, DL1C1234, JK02AB1234
        self.plate_pattern = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$')
        self.char_replacements_digit = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6'}
        self.char_replacements_alpha = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B'}

    def preprocess_plate(self, plate_crop):
        """Enhances contrast and clarifies character contours."""
        if plate_crop is None or plate_crop.size == 0:
            return None

        # Resize to standard height for OCR accuracy
        h, w = plate_crop.shape[:2]
        if h < 40 or w < 80:
            plate_crop = cv2.resize(plate_crop, (240, 70), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        # Bilateral filter reduces noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return thresh

    def clean_plate_text(self, text):
        """Sanitizes text and corrects common alphanumeric OCR substitutions."""
        cleaned = re.sub(r'[^A-Za-z0-9]', '', text).upper()
        if len(cleaned) < 6:
            return cleaned

        # Standard Indian format: 2 letters (state) + 2 digits (district) + 1-2 letters + 4 digits
        # Correct State Code prefix (first 2 characters must be letters)
        p_list = list(cleaned)
        for i in range(min(2, len(p_list))):
            if p_list[i] in self.char_replacements_alpha:
                p_list[i] = self.char_replacements_alpha[p_list[i]]

        # District code (characters 2 and 3 should be digits)
        if len(p_list) >= 4:
            for i in range(2, 4):
                if p_list[i] in self.char_replacements_digit:
                    p_list[i] = self.char_replacements_digit[p_list[i]]

        # Last 4 characters should be digits
        if len(p_list) >= 8:
            for i in range(len(p_list) - 4, len(p_list)):
                if p_list[i] in self.char_replacements_digit:
                    p_list[i] = self.char_replacements_digit[p_list[i]]

        return "".join(p_list)

    def read_plate(self, plate_image):
        """
        Reads license plate characters from an image crop.
        Returns dict with plate string, confidence score, and validity flag.
        """
        try:
            if plate_image is None or plate_image.size == 0:
                return None

            # Run OCR on original and thresholded
            results = self.reader.readtext(plate_image, detail=1)
            if not results:
                # Try thresholded preprocessing
                preprocessed = self.preprocess_plate(plate_image)
                if preprocessed is not None:
                    results = self.reader.readtext(preprocessed, detail=1)

            if not results:
                return None

            best_text = ""
            best_conf = 0.0

            for bbox, text, conf in results:
                cleaned = self.clean_plate_text(text)
                if len(cleaned) >= 5 and conf > best_conf:
                    best_text = cleaned
                    best_conf = float(conf)

            if not best_text:
                return None

            is_valid = bool(self.plate_pattern.match(best_text))
            return {
                "plate": best_text,
                "confidence": round(best_conf, 2),
                "is_valid": is_valid
            }

        except Exception as e:
            print(f"ANPR Engine Error: {e}")
            return None


if __name__ == "__main__":
    ocr = OCRModule()
    sample = "HR 26 DK 8901"
    cleaned = ocr.clean_plate_text(sample)
    print("Sanitization Test:", sample, "->", cleaned, "| Valid:", bool(ocr.plate_pattern.match(cleaned)))