import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from web3 import Web3
from thefuzz import fuzz
import sqlite3

print("✅ Streamlit:", st.__version__)
print("✅ OpenCV:", cv2.__version__)
print("✅ NumPy:", np.__version__)
print("✅ Ultralytics (YOLO): Imported Successfully")
print("✅ EasyOCR: Imported Successfully")
print("✅ Web3: Imported Successfully")
print("✅ FuzzyWuzzy: Imported Successfully")
print("✅ SQLite3: Built-in and ready")
print("\n🎉 ALL DEPENDENCIES INSTALLED CORRECTLY! READY TO BUILD.")
