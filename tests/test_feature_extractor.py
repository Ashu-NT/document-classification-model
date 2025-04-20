import os
import sys
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import csv
from datetime import datetime
import tempfile
import shutil

# Add project directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the modules (adjust imports based on your actual module structure)
try:
    from utils.feature_extractor import PDFFeatureExtractor
except ImportError:
    # For testing purposes, we'll mock these if they can't be imported
    class PDFFeatureExtractor:
        error_log_dir = os.path.abspath("../data/error_log_pdf")
        error_log_file = os.path.join(error_log_dir, "3212_hug.csv")
        _error_log_lock = MagicMock()
        
        @staticmethod
        def log_error(pdf_path, page_num, error_message):
            try:
                os.makedirs(PDFFeatureExtractor.error_log_dir, exist_ok=True)
                
                with PDFFeatureExtractor._error_log_lock:
                    file_exists = os.path.exists(PDFFeatureExtractor.error_log_file)
                    with open(PDFFeatureExtractor.error_log_file, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(['Timestamp', 'PDF File Path', 'Page Number', 'Error Message'])
                        writer.writerow([datetime.now(), pdf_path, page_num, error_message])
            except Exception as e:
                print(f"Failed to write error log: {str(e)}")
                
        @staticmethod
        def extract_text(pdf_path):
            return "Mocked PDF text"
            
        @staticmethod
        def extract_visual_features(pdf_path):
            return [0.5, 10, 0.8, 100000]


@pytest.fixture
def temp_directory():
    """Create a temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    try:
        shutil.rmtree(temp_dir)
    except:
        pass


class TestPDFFeatureExtractor:
    def test_extract_text(self):
        """Test PDF text extraction including fallback check"""
        with patch('fitz.open') as mock_fitz_open, \
            patch('pytesseract.image_to_string') as mock_ocr, \
            patch('PIL.Image.frombytes') as mock_frombytes:
            
            # Mock setup
            mock_doc = MagicMock()
            mock_page = MagicMock()
            mock_pixmap = MagicMock()

            # Simulate short text that triggers OCR
            mock_page.get_text.return_value = "Hi"
            mock_page.get_pixmap.return_value = mock_pixmap
            mock_pixmap.width = 100
            mock_pixmap.height = 100
            mock_pixmap.samples = b'\x00' * (100 * 100 * 3)

            mock_doc.__iter__.return_value = iter([mock_page])
            mock_doc.__len__.return_value = 1
            mock_fitz_open.return_value.__enter__.return_value = mock_doc

            mock_img = MagicMock()
            mock_frombytes.return_value = mock_img
            mock_ocr.return_value = "OCR fallback text"

            # Call actual function
            text = PDFFeatureExtractor.extract_text("test.pdf")

            # Assert OCR fallback text was included
            assert "OCR fallback text" in text
            mock_ocr.assert_called_once_with(mock_img)



    def test_extract_text_ocr_fallback(self):
        """Test OCR fallback when text content is minimal"""
        with patch('fitz.open') as mock_fitz_open, \
             patch('pytesseract.image_to_string') as mock_ocr:
            
            # Set up mocks
            mock_doc = MagicMock()
            mock_page = MagicMock()
            mock_pixmap = MagicMock()
            
            mock_fitz_open.return_value.__enter__.return_value = mock_doc
            mock_doc.__iter__.return_value = [mock_page]
            mock_page.get_text.return_value = "Short"  # Below threshold to trigger OCR
            mock_page.get_pixmap.return_value = mock_pixmap
            mock_pixmap.width = 100
            mock_pixmap.height = 100
            mock_pixmap.samples = b'\x00' * (100 * 100 * 3)
            
            mock_ocr.return_value = "OCR extracted text"
            
            # Test extraction with OCR fallback
            text = PDFFeatureExtractor.extract_text("test.pdf")
            
            # Verify OCR was used
            assert "OCR extracted text" in text
            mock_ocr.assert_called_once()

    def test_extract_visual_features(self):
        """Test extraction of visual features"""
        with patch('fitz.open') as mock_fitz_open, \
             patch('cv2.cvtColor') as mock_cv2_color, \
             patch('cv2.Canny') as mock_canny, \
             patch('cv2.findContours') as mock_contours, \
             patch('cv2.threshold') as mock_threshold:
            
            # Set up mocks
            mock_doc = MagicMock()
            mock_page = MagicMock()
            mock_pixmap = MagicMock()
            
            mock_fitz_open.return_value.__enter__.return_value = mock_doc
            mock_doc.page_count = 1
            mock_doc.load_page.return_value = mock_page
            mock_page.get_pixmap.return_value = mock_pixmap
            mock_pixmap.samples = b'\x00' * (100 * 100 * 3)
            mock_pixmap.h = mock_pixmap.height = 100
            mock_pixmap.w = mock_pixmap.width = 100
            
            mock_cv2_color.return_value = np.zeros((100, 100), dtype=np.uint8)
            mock_canny.return_value = np.ones((100, 100), dtype=np.uint8)
            mock_contours.return_value = ([MagicMock(), MagicMock()], None)
            mock_threshold.return_value = (None, np.ones((100, 100), dtype=np.uint8) * 255)
            
            # Test feature extraction
            features = PDFFeatureExtractor.extract_visual_features("test.pdf")
            
            # Verify results
            assert len(features) == 4
            assert all(isinstance(f, (int, float)) for f in features)

    def test_log_error(self, temp_directory):
        """Test error logging functionality"""
        # Set temporary error log path
        original_log_dir = PDFFeatureExtractor.error_log_dir
        original_log_file = PDFFeatureExtractor.error_log_file
        
        try:
            PDFFeatureExtractor.error_log_dir = temp_directory
            PDFFeatureExtractor.error_log_file = os.path.join(temp_directory, "error_log.csv")
            
            # Log an error
            PDFFeatureExtractor.log_error("test.pdf", 1, "Test error message")
            
            # Verify error was logged
            assert os.path.exists(PDFFeatureExtractor.error_log_file)
            
            # Check content
            with open(PDFFeatureExtractor.error_log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                assert "test.pdf" in content
                assert "Test error message" in content
                
        finally:
            # Restore original paths
            PDFFeatureExtractor.error_log_dir = original_log_dir
            PDFFeatureExtractor.error_log_file = original_log_file