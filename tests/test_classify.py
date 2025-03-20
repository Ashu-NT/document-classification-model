# tests/test_classify.py

# verify all paths 15 --> 27

import unittest
import os
import pandas as pd
from scripts.classify_documents import classify_pdfs
from utils.preprocessing import TextProcessor
from utils.pdf_processor import PDFProcessor

class TestClassify(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.test_data_dir = os.path.abspath("../data/test")
        cls.model_path = "../models/cached_multimodal_doc_classifier_v1.pkl"
        cls.output_csv = os.path.join(cls.test_data_dir, "test_results.csv")
        cls.output_excel = os.path.join(cls.test_data_dir, "test_results.xlsx")

        # Ensure the test data directory exists
        os.makedirs(cls.test_data_dir, exist_ok=True)

    def test_classify_pdfs(self):
        """Test the classify_pdfs function."""
        # Create a sample PDF for testing
        sample_pdf_path = os.path.join(self.test_data_dir, "sample.pdf")
        with open(sample_pdf_path, "w") as f:
            f.write("This is a sample PDF for testing.")

        # Run classification
        result_df = classify_pdfs(
            root_dir=self.test_data_dir,
            model_path=self.model_path,
            output_csv=self.output_csv,
            output_excel=self.output_excel
        )

        # Check if results were saved
        self.assertTrue(os.path.exists(self.output_csv))
        self.assertTrue(os.path.exists(self.output_excel))

        # Check if the result DataFrame is not empty
        self.assertFalse(result_df.empty)

        # Clean up
        os.remove(sample_pdf_path)
        os.remove(self.output_csv)
        os.remove(self.output_excel)

if __name__ == "__main__":
    unittest.main()