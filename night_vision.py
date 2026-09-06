"""
Night Vision & Thermal Analytics Engine (IBVAP)
Provides low-light contrast enhancement (CLAHE), false-color FLIR/thermal infrared simulation,
and motion saliency for border surveillance under zero-light conditions.
"""

import cv2
import numpy as np


class NightVisionEngine:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=25, detectShadows=False)
        self.thermal_colormaps = {
            "inferno": cv2.COLORMAP_INFERNO,
            "ironbow": cv2.COLORMAP_JET,
            "bone": cv2.COLORMAP_BONE,
            "turbo": cv2.COLORMAP_TURBO
        }

    def enhance_clahe(self, frame):
        """Dynamic CLAHE enhancement on Luminance channel for low-light/foggy surveillance."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = self.clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def apply_thermal(self, frame, palette="inferno"):
        """
        Simulates military FLIR/thermal imaging using false-color radiometric palettes.
        Inverts and contrasts intensity gradients to highlight human body heat signatures.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Contrast stretching to highlight thermal heat differences
        normalized = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        # Apply slight bilateral blur to simulate thermal sensor diffusion
        blurred = cv2.bilateralFilter(normalized, 5, 50, 50)
        cmap = self.thermal_colormaps.get(palette.lower(), cv2.COLORMAP_INFERNO)
        thermal = cv2.applyColorMap(blurred, cmap)
        return thermal

    def detect_motion_saliency(self, frame, min_area=400):
        """
        Detects movement in low-light/thermal environments using background subtraction.
        Returns foreground mask and bounding boxes of moving entities.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = self.bg_subtractor.apply(gray)
        
        # Morphological noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel, iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_boxes = []
        for c in contours:
            if cv2.contourArea(c) >= min_area:
                x, y, w, h = cv2.boundingRect(c)
                motion_boxes.append((x, y, x + w, y + h))

        return fg_mask, motion_boxes
