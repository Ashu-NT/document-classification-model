# tests/test_classify.py

import unittest
import os
import sys
from unittest.mock import MagicMock, patch, mock_open, ANY
import pandas as pd
import numpy as np

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.classify_documents import classify_pdfs

class TestClassify(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment with proper absolute paths"""
        cls.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        cls.test_data_dir = os.path.join(cls.project_root, "data", "test")
        cls.model_path = os.path.join(
            cls.project_root, "models", "cached_multimodal_doc_classifier_v1.pkl"
        )
        cls.output_csv = os.path.join(cls.test_data_dir, "test_results.csv")
        cls.output_excel = os.path.join(cls.test_data_dir, "test_results.xlsx")

        # Ensure test data directory exists
        os.makedirs(cls.test_data_dir, exist_ok=True)

    @patch('pandas.DataFrame.to_excel')
    @patch('pandas.DataFrame.to_csv')
    @patch('joblib.Parallel')
    @patch('joblib.load')
    @patch('scripts.classify_documents.TextPreprocessor')
    @patch('scripts.classify_documents.PDFFeatureExtractor')
    def test_classify_pdfs(self,
                          mock_extractor,
                          mock_textproc,
                          mock_joblib_load,
                          mock_parallel,
                          mock_to_csv,
                          mock_to_excel):
        """Test full classification workflow with mocked dependencies"""
        # Create a dummy PDF so we have something to classify
        sample_pdf = os.path.join(self.test_data_dir, "sample.pdf")
        with open(sample_pdf, "wb") as f:
            f.write(b"%PDF-1.4\n%EOF")

        # Mocked TextPreprocessor
        fake_preprocessor = MagicMock()
        fake_preprocessor.preprocess.return_value = "processed sample text"
        mock_textproc.return_value = fake_preprocessor

        # Mocked PDFFeatureExtractor
        mock_extractor.extract_text.return_value = "sample text"
        mock_extractor.extract_visual_features.return_value = [0.5, 0.3, 0.2]

        # Mocked model and classifier
        fake_classifier = MagicMock()
        fake_classifier.classes_ = ['Manual', 'Drawing', 'DataSheet']
        fake_model = MagicMock()
        fake_model.steps = [('preprocessor', None), ('classifier', fake_classifier)]
        fake_model.predict.return_value = np.array([0])  # Always predict 'Manual'
        mock_joblib_load.return_value = fake_model

        # Mock parallel processing
        mock_results = [
            {
                'file_name': 'sample.pdf',
                'file_path': sample_pdf,
                'main_folder': 'test',
                'parent_folder': 'test',
                'relative_folder': '.',
                'component_name': '.',
                'document_type': 'Manual'
            }
        ]
        mock_parallel.return_value = MagicMock()
        mock_parallel.return_value.__enter__.return_value = mock_results

        # Mock os.walk to return our single PDF file
        with patch('os.walk') as mock_walk, \
             patch('os.cpu_count', return_value=4):
            mock_walk.return_value = [
                (self.test_data_dir, [], ['sample.pdf', 'not_a_pdf.txt'])
            ]

            # Execute the function under test
            result_df = classify_pdfs(
                root_dir=self.test_data_dir,
                model_path=self.model_path,
                output_csv=self.output_csv,
                output_excel=self.output_excel
            )

        # Verify model was loaded
        mock_joblib_load.assert_called_once_with(self.model_path)

        # Verify DataFrame was created with results
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertFalse(result_df.empty)

        # Verify CSV output was saved
        mock_to_csv.assert_any_call(self.output_csv, index=False)

        # Verify Excel outputs were saved (one for results, one for non-PDF files)
        self.assertEqual(mock_to_excel.call_count, 2)


    @classmethod
    def tearDownClass(cls):
        """Clean up test files"""
        for path in [
            os.path.join(cls.test_data_dir, "sample.pdf"),
            cls.output_csv,
            cls.output_excel
        ]:
            if os.path.exists(path):
                os.remove(path)

if __name__ == "__main__":
    unittest.main()