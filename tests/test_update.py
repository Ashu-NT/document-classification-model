# tests/test_update.py

import unittest
import os
import pandas as pd
import sys
from unittest.mock import patch, MagicMock, Mock

# Add the root path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class TestUpdate(unittest.TestCase):
    @patch.dict("sys.modules", {"seaborn": Mock(), "matplotlib": Mock(), "matplotlib.pyplot": Mock()})
    @patch("pandas.read_csv")
    @patch("joblib.dump")
    @patch("joblib.load")
    @patch("os.path.exists")
    def test_model_update(
        self,
        mock_exists,
        mock_load,
        mock_dump,
        mock_read_csv,
    ):
        # Import AFTER sys.modules mocks
        from scripts.update_model import ModelUpdater

        # Mock version file behavior
        mock_exists.side_effect = lambda path: "version.txt" in path or "processed_data.joblib" in path or "pkl" in path

        # Fake previous model
        mock_model = MagicMock()
        mock_model.fit.return_value = mock_model
        mock_model.predict.return_value = [0, 1]
        mock_load.side_effect = [
            mock_model,  # load previous model
            pd.DataFrame({  # load previous cache
                "File Path": ["existing.pdf"],
                "Raw_Text": ["text"],
                "Processed_Text": ["processed"],
                "Visual_Features": [[0.1, 0.2]],
                "Document types": ["TypeA"]
            })
        ]

        # Fake new CSV input
        mock_read_csv.return_value = pd.DataFrame({
            "File Path": ["new.pdf"],
            "Document types": ["TypeB"]
        })

        # Patch internals
        with patch("scripts.update_model.PDFProcessor.extract_text", return_value="sample text"), \
             patch("scripts.update_model.PDFProcessor.extract_visual_features", return_value=[0.3, 0.4]), \
             patch("scripts.update_model.TextProcessor.load_cache"), \
             patch("scripts.update_model.TextProcessor.preprocess", return_value="processed new text"), \
             patch("scripts.update_model.build_model_pipeline", return_value=mock_model):

            model_path = os.path.abspath("../models/cached_multimodal_doc_classifier")
            cache_dir = os.path.abspath("../data/cache")

            updater = ModelUpdater(model_path, cache_dir)
            updater.update("fake_path.csv")

            # Asserts
            mock_read_csv.assert_called_once()
            mock_model.fit.assert_called_once()
            mock_dump.assert_called()

if __name__ == "__main__":
    unittest.main()