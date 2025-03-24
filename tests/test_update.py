
# tests/test_update.py
import unittest
import os
import pandas as pd

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..",'tests')))

from scripts.update_model import ModelUpdater
from utils.preprocessing import TextProcessor
from utils.pdf_processor import PDFProcessor

class TestUpdate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.test_data_dir = os.path.abspath("../data/test")
        cls.new_data_path = os.path.join(cls.test_data_dir, "new_data.csv")
        cls.model_path = "../models/cached_multimodal_doc_classifier_v1.pkl"

        # Create a sample new dataset
        sample_data = {
            "File Path": [os.path.join(cls.test_data_dir, "new_sample1.pdf"), os.path.join(cls.test_data_dir, "new_sample2.pdf")],
            "Document types": ["Manual", "Datasheet"]
        }
        cls.new_df = pd.DataFrame(sample_data)
        cls.new_df.to_csv(cls.new_data_path, index=False)

        # Create sample PDFs
        for file_path in cls.new_df["File Path"]:
            with open(file_path, "w") as f:
                f.write("This is a new sample PDF for testing.")

    def test_model_update(self):
        """Test the model update workflow."""
        # Initialize the ModelUpdater
        updater = ModelUpdater(self.model_path, "../data/cache")

        # Run the update process
        updater.update(self.new_data_path)

        # Check if the new model version was saved
        new_model_path = f"../models/cached_multimodal_doc_classifier_v{updater.current_version}.pkl"
        self.assertTrue(os.path.exists(new_model_path))

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Remove sample files
        for file_path in cls.new_df["File Path"]:
            if os.path.exists(file_path):
                os.remove(file_path)
        if os.path.exists(cls.new_data_path):
            os.remove(cls.new_data_path)
        if os.path.exists(cls.model_path):
            os.remove(cls.model_path)

if __name__ == "__main__":
    unittest.main()