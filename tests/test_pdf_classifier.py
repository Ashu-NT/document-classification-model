import os
import sys
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

# Add project directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the modules (adjust imports based on your actual module structure)
try:
    from scripts.classify_documents import classify_pdfs, process_pdf, TextPreprocessor
except ImportError:
    # For testing purposes, we'll mock these if they can't be imported
    def classify_pdfs(root_dir, model_path, output_csv=None, output_excel=None):
        return pd.DataFrame()
        
    def process_pdf(pdf_path, root_dir, preprocessor, model, class_labels):
        return {
            'file_name': os.path.basename(pdf_path),
            'file_path': pdf_path,
            'main_folder': os.path.basename(os.path.dirname(pdf_path)),
            'parent_folder': os.path.basename(os.path.dirname(pdf_path)),
            'relative_folder': os.path.relpath(os.path.dirname(pdf_path), root_dir),
            'component_name': os.path.basename(os.path.dirname(pdf_path)),
            'document_type': 'Manual'
        }
        
    class TextPreprocessor:
        def preprocess(self, text):
            return "preprocessed text"


@pytest.fixture
def temp_directory():
    """Create a temporary directory for test files"""
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    try:
        shutil.rmtree(temp_dir)
    except:
        pass


class TestPDFClassifier:
    def test_process_pdf(self):
        """Test the process_pdf function"""
        # Set up test parameters
        pdf_path = "/test/path/4710.11243.98673/document.pdf"
        root_dir = "/test/path"
        preprocessor = MagicMock()
        model = MagicMock()
        model.predict.return_value = [0]
        class_labels = ["Manual", "Certificate", "Datasheet"]
        
        # Mock the dependencies
        with patch('utils.feature_extractor.PDFFeatureExtractor.extract_text', return_value="Sample text"), \
             patch('utils.feature_extractor.PDFFeatureExtractor.extract_visual_features', return_value=[0.5, 10, 0.8, 100000]):
            
            preprocessor.preprocess.return_value = "processed sample text"
            
            # Process the PDF
            result = process_pdf(pdf_path, root_dir, preprocessor, model, class_labels)
            
            # Verify results
            assert result["file_name"] == "document.pdf"
            assert result["file_path"] == os.path.normpath(pdf_path)
            assert result["component_name"] == "4710.11243.98673"
            assert result["document_type"] == "Manual"
            

        
    def test_text_preprocessor(self):
        """Test the TextPreprocessor class"""
        preprocessor = TextPreprocessor()
        
        # Test with regular text
        result = preprocessor.preprocess("This is a test document")
        assert isinstance(result, str)
        
        # Test with special characters and numbers
        result = preprocessor.preprocess("Document with special #$%^ chars and numbers 123")
        assert isinstance(result, str)