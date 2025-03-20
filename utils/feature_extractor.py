import fitz
import os
from datetime import datetime
import pytesseract
from PIL import Image
import cv2
import numpy as np
import csv

from utils.path_log import THRESHOLD_OCR

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update path

class PDFFeatureExtractor:
    
    # Log file for errors
    error_log_dir = os.path.abspath("../data/error_log_pdf")
    error_log_file = os.path.join(error_log_dir, "Batch_2_Cosmos_error_log.csv") #Change Name

    @staticmethod
    def log_error(pdf_path, page_num, error_message):
        """Logs error details to a CSV file."""
        try:
            # Create directory structure if it doesn't exist
            os.makedirs(PDFFeatureExtractor.error_log_dir, exist_ok=True)  
            
            # Create the log file if it doesn't exist and write headers
            file_exists = os.path.exists(PDFFeatureExtractor.error_log_file)
            
            with open(PDFFeatureExtractor.error_log_file, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(['Timestamp', 'PDF File Path', 'Page Number', 'Error Message'])
                writer.writerow([datetime.now(), pdf_path, page_num, error_message])
            
        except Exception as e:
            print(f"Failed to write error log: {str(e)}") 
             
    @staticmethod
    def extract_text(pdf_path):
        """Hybrid text extraction with OCR fallback"""
        page_num = "N/A"  # Initialize before loop
        try:
            with fitz.open(pdf_path) as doc:
                text = ""
                
                for page_num, page in enumerate(doc, start=1):
                    # Try regular text extraction first
                    page_text = page.get_text("text")
                    if len(page_text) < THRESHOLD_OCR:  # Fallback to OCR if text is sparse
                        pix = page.get_pixmap()
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        page_text += "\n" + pytesseract.image_to_string(img)
                    text += page_text
                return text
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            error_message = f"Error processing page {page_num}: {e}"
            PDFFeatureExtractor.log_error(pdf_path, page_num, error_message)  # Log error to CSV
            return ""

    @staticmethod
    def extract_visual_features(pdf_path):
        """Extract visual features from first page"""
        try:
            with fitz.open(pdf_path) as doc:
                if len(doc) == 0:
                    return [0]*4
                
                page = doc.load_page(0)
                pix = page.get_pixmap()
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                
                # Convert to grayscale
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                
                # Edge detection
                edges = cv2.Canny(gray, 100, 200)
                edge_density = np.mean(edges)
                
                # Contour analysis
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour_count = len(contours)
                
                # Layout analysis
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                white_ratio = np.mean(binary) / 255
                
                return [edge_density, contour_count, white_ratio, pix.width*pix.height]
        except Exception as e:
            print(f"Visual feature error {pdf_path}: {e}")
            error_message = f"Visual feature extraction error: {e}"
            PDFFeatureExtractor.log_error(pdf_path, "N/A", error_message)
            return [0]*4
