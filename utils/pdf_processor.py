# utils/pdf_processor.py
import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import numpy as np
import cv2
from joblib import Memory
from utils.path_log import CACHE_DIR, THRESHOLD_OCR


memory = Memory(CACHE_DIR, verbose=0)

class PDFProcessor:
    """This class extract data from pdf files"""
    @staticmethod
    def file_signature(file_path):
        
        """Generate unique signature for file state"""
        try:
            stat = os.stat(file_path)
            return f"{file_path}-{stat.st_size}-{stat.st_mtime_ns}"
        except Exception as e:
            print(f"Error getting file signature: {e}")
            return None

    @memory.cache
    def extract_text(pdf_path, version_tag):
        """ Extract text from file
            if len(text) < THRESHOLD, converts pdf to image for OCR processing
        """
        try:
            with fitz.open(pdf_path) as doc:
                text = ""
                for page in doc:
                    # Replace existing image handling with:
                    pix = page.get_pixmap()
                    with Image.frombytes("RGB", [pix.width, pix.height], pix.samples) as img:
                        page_text = page.get_text("text")
                        if len(page_text) < THRESHOLD_OCR:
                            page_text += "\n" + pytesseract.image_to_string(img)
                    text += page_text
                return text
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return ""

    @memory.cache
    def extract_visual_features(pdf_path, version_tag):
        """Cached visual feature extraction"""
        try:
            with fitz.open(pdf_path) as doc:
                if len(doc) == 0:
                    return [0]*4
                
                page = doc.load_page(0)
                pix = page.get_pixmap()
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                edge_density = np.mean(edges)
                
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour_count = len(contours)
                
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                white_ratio = np.mean(binary) / 255
                
                return [edge_density, contour_count, white_ratio, pix.width*pix.height]
        except Exception as e:
            print(f"Visual feature error {pdf_path}: {e}")
            return [0]*4
