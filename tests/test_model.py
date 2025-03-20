
# tests/test_model.py
import unittest
import os
import pandas as pd

from scripts.dd_OCR_cache_model import main
from utils.preprocessing import TextProcessor
from utils.pdf_processor import PDFProcessor

class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.test_data_dir = os.path.abspath("../data/test")
        cls.training_data_path = os.path.join(cls.test_data_dir, "training_data.csv")
        cls.model_path = "../models/cached_multimodal_doc_classifier_v1.pkl"

        # Create a sample training dataset
        sample_data = {
            "File Path": [os.path.join(cls.test_data_dir, "sample1.pdf"), os.path.join(cls.test_data_dir, "sample2.pdf")],
            "Document types": ["Manual", "Datasheet"]
        }
        cls.sample_df = pd.DataFrame(sample_data)
        cls.sample_df.to_csv(cls.training_data_path, index=False)

        # Create sample PDFs
        for file_path in cls.sample_df["File Path"]:
            with open(file_path, "w") as f:
                f.write("This is a sample PDF for testing.")

    def test_training(self):
        """Test the model training workflow."""
        # Run the training script
        main()

        # Check if the model was saved
        self.assertTrue(os.path.exists(self.model_path))

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Remove sample files
        for file_path in cls.sample_df["File Path"]:
            if os.path.exists(file_path):
                os.remove(file_path)
        if os.path.exists(cls.training_data_path):
            os.remove(cls.training_data_path)
        if os.path.exists(cls.model_path):
            os.remove(cls.model_path)

if __name__ == "__main__":
    unittest.main()