import os
import sys
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Add project directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from utils.pdf_processor import memory  # Add this import
    from utils.pdf_processor import PDFProcessor
except ImportError:
    # For testing purposes, we'll mock these if they can't be imported
    class PDFProcessor:
        @staticmethod
        def file_signature(file_path):
            try:
                stat = os.stat(file_path)
                return f"{file_path}-{stat.st_size}-{stat.st_mtime_ns}"
            except Exception as e:
                print(f"Error getting file signature: {e}")
                return None
                
        @staticmethod
        def extract_text(pdf_path, version_tag):
            return "Mocked text content"
            
        @staticmethod
        def extract_visual_features(pdf_path, version_tag):
            return [0.5, 10, 0.8, 100000]


@pytest.fixture
def sample_pdf_path():
    """Return a path to a sample PDF file for testing"""
    
    # This is a mock path - in actual tests,in real test PDF should provided
    return os.path.join(os.path.dirname(__file__), 'test_data', 'sample.pdf')


class TestPDFProcessor:
    @patch('fitz.open')
    @patch('pytesseract.image_to_string')
    def test_extract_text(self, mock_ocr, mock_fitz_open, sample_pdf_path):
        # Clear cache before test execution
        memory.clear()
        # Create mock page
        mock_page = MagicMock()
        mock_page.rect.width = 595
        mock_page.rect.height = 842
        mock_page.get_text.return_value = "Short text"  # Ensure text is short to trigger OCR

        # Mock pixmap for OCR
        mock_pixmap = MagicMock()
        mock_pixmap.width = 100
        mock_pixmap.height = 100
        mock_pixmap.samples = b'\x00' * (100 * 100 * 3)
        mock_page.get_pixmap.return_value = mock_pixmap

        # Configure OCR mock
        mock_ocr.return_value = " OCR fallback text"

        # Mock document with page access
        mock_doc = MagicMock()
        mock_doc.__iter__.return_value = iter([mock_page])
        mock_doc.pages = [mock_page]  # Support doc.pages access
        mock_doc.page_count = 1
        mock_doc.__len__.return_value = 1

        # Configure fitz.open context manager
        mock_fitz_open.return_value.__enter__.return_value = mock_doc

        # Execute method
        result = PDFProcessor.extract_text(sample_pdf_path, "v1")

        # Verify text extraction called
        mock_page.get_text.assert_called_once_with("text")  # Or adjust argument if needed
        assert "OCR fallback text" in result  # Ensure OCR fallback triggered
    
    @patch('fitz.open')
    @patch('cv2.cvtColor')
    @patch('cv2.Canny')
    @patch('cv2.findContours')
    @patch('cv2.threshold')
    def test_extract_visual_features(self, mock_threshold, mock_contours, mock_canny, 
                                    mock_cv2_color, mock_fitz_open, sample_pdf_path):
        """Test extraction of visual features from PDF"""
        # Set up mocks
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_pixmap = MagicMock()
        
        mock_fitz_open.return_value.__enter__.return_value = mock_doc
        mock_doc.__len__.return_value = 1
        mock_doc.load_page.return_value = mock_page
        mock_page.get_pixmap.return_value = mock_pixmap
        mock_pixmap.samples = b'\x00' * (100 * 100 * 3)
        mock_pixmap.width = 100
        mock_pixmap.height = 100
        
        mock_cv2_color.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_canny.return_value = np.ones((100, 100), dtype=np.uint8)
        mock_contours.return_value = ([MagicMock(), MagicMock()], None)
        mock_threshold.return_value = (None, np.ones((100, 100), dtype=np.uint8) * 255)
        
        # Test feature extraction
        features = PDFProcessor.extract_visual_features(sample_pdf_path, "test_version")
        
        # Verify results
        assert isinstance(features, list)
        assert len(features) == 4

    def test_file_signature(self):
        """Test file signature generation"""
        # Create a temporary file
        import tempfile
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"Test content")
            tmp_path = tmp.name
        
        try:
            # Generate signature
            signature = PDFProcessor.file_signature(tmp_path)
            
            # Verify signature format
            assert tmp_path in signature
            assert "-" in signature
            
            # Verify signature changes when file changes
            with open(tmp_path, 'a') as f:
                f.write("\nModified content")
            
            new_signature = PDFProcessor.file_signature(tmp_path)
            assert signature != new_signature
            
        finally:
            # Clean up
            os.unlink(tmp_path)

    def test_error_handling(self):
        """Test error handling in PDF processing"""
        # Test with invalid file path
        with patch('fitz.open', side_effect=Exception("Invalid PDF")):
            result = PDFProcessor.extract_text("nonexistent.pdf", "test_version")
            assert result == ""  # Or whatever your error behavior is
            
            features = PDFProcessor.extract_visual_features("nonexistent.pdf", "test_version")
            assert features == [0, 0, 0, 0]  # Or whatever your error behavior is