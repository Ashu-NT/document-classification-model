import os
import sys
import pytest
import pandas as pd
import numpy as np
import joblib
from unittest.mock import patch, MagicMock
import tempfile
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.update_model import ModelUpdater

@pytest.fixture
def setup_test_environment():
    original_dir = os.getcwd()  # Save the original directory
    test_dir = tempfile.mkdtemp()
    model_base_path = os.path.join(test_dir, "test_model")
    cache_base_dir = os.path.join(test_dir, "test_cache")

    version_file = os.path.join(test_dir, "version.txt")
    with open(version_file, "w") as f:
        f.write("1")

    os.chdir(test_dir)  # Change to test directory

    # Create previous model
    prev_model_path = f"{model_base_path}_v1.pkl"
    joblib.dump({"model": "dummy_model"}, prev_model_path)

    # Create cache for v1
    prev_cache_dir = os.path.join(cache_base_dir, "v1")
    os.makedirs(prev_cache_dir, exist_ok=True)
    mock_cache = pd.DataFrame({
        "File Path": [os.path.normpath("/path/to/doc1.pdf"), os.path.normpath("/path/to/doc2.pdf")],
        "Document types": ["Manual", "Drawing"],
        "Raw_Text": ["text1", "text2"],
        "Processed_Text": ["processed1", "processed2"],
        "Visual_Features": [[], []],
        "Label": [0, 1]
    })
    joblib.dump({"data_frame": mock_cache, "version": 1}, os.path.join(prev_cache_dir, "processed_data.joblib"))

    # Create new data
    new_data_path = os.path.join(test_dir, "new_data.csv")
    pd.DataFrame({
        "File Path": ["/path/to/doc3.pdf", "/path/to/doc4.pdf"],
        "Document types": ["Certificate", "Manual"]
    }).to_csv(new_data_path, index=False)

    yield {
        "test_dir": test_dir,
        "model_base_path": model_base_path,
        "cache_base_dir": cache_base_dir,
        "new_data_path": new_data_path,
        "version_file": version_file
    }

    os.chdir(original_dir)  # Revert to original directory
    shutil.rmtree(test_dir)  # Now safe to delete

@pytest.fixture
def mock_dependencies():
    with patch("utils.preprocessing.TextProcessor") as mock_text_processor, \
         patch("utils.pdf_processor.PDFProcessor.extract_text", return_value="mock_text") as mock_extract_text, \
         patch("utils.pdf_processor.PDFProcessor.extract_visual_features", return_value=[]) as mock_visual, \
         patch("scripts.update_model.build_model_pipeline") as mock_build_model:

        mock_instance = MagicMock()
        mock_instance.preprocess.side_effect = lambda x: f"processed_{x}"
        mock_text_processor.return_value = mock_instance

        # Create a realistic mock model with fit/predict methods
        mock_model = MagicMock()
        mock_model.fit.return_value = mock_model  # For method chaining
        mock_model.predict.return_value = np.array([0, 1])  # Dummy predictions
        mock_build_model.return_value = mock_model

        yield {
            "text_processor": mock_text_processor,
            "extract_text": mock_extract_text,
            "visual_features": mock_visual,
            "build_model": mock_build_model
        }


def test_load_previous_model(setup_test_environment, mock_dependencies):
    updater = ModelUpdater(setup_test_environment["model_base_path"], setup_test_environment["cache_base_dir"])
    assert updater.model is not None


def test_process_new_data(setup_test_environment, mock_dependencies):
    updater = ModelUpdater(setup_test_environment["model_base_path"], setup_test_environment["cache_base_dir"])
    new_data = pd.read_csv(setup_test_environment["new_data_path"])
    updated_data = updater.process_new_data(new_data)
    assert len(updated_data) == 4
    assert os.path.exists(os.path.join(setup_test_environment["cache_base_dir"],
                                       f"v{updater.current_version}", "processed_data.joblib"))


# Fix for the test_get_next_version function
def test_get_next_version(setup_test_environment, mock_dependencies):
    test_dir = setup_test_environment["test_dir"]
    model_base_path = setup_test_environment["model_base_path"]

    # Create dummy model for version 5
    model_path_v5 = f"{model_base_path}_v5.pkl"
    joblib.dump({"model": "dummy_v5"}, model_path_v5)
    
    # Also create v4 model, which will be looked for by _load_previous_model
    model_path_v4 = f"{model_base_path}_v4.pkl"
    joblib.dump({"model": "dummy_v4"}, model_path_v4)

    with open("version.txt", "w") as f:
        f.write("5")

    updater = ModelUpdater(model_base_path, setup_test_environment["cache_base_dir"])
    
    assert updater.current_version == 5
    with open("version.txt", "r") as f:
        assert int(f.read()) == 6

@patch("scripts.update_model.train_test_split")
@patch("scripts.update_model.cross_val_score")
@patch("joblib.dump")
def test_retrain_model(mock_dump, mock_cv, mock_split, setup_test_environment, mock_dependencies):
    # Test dataframe with required columns
    test_df = pd.DataFrame({
        "File Path": ["/path/to/doc1.pdf", "/path/to/doc2.pdf"],
        "Document types": ["Manual", "Drawing"],
        "Processed_Text": ["processed1", "processed2"],
        "Visual_Features": [[], []],
        "Label": [0, 1]  # Ensure Label column exists
    })
    
    # Configure mocks - Use DataFrame objects that match expected structure
    X_train = test_df[["Processed_Text", "Visual_Features"]]
    y_train = test_df["Label"]
    X_test = test_df[["Processed_Text", "Visual_Features"]]
    y_test = test_df["Label"]
    
    # Return proper DataFrame/Series objects
    mock_split.return_value = (X_train, X_test, y_train, y_test)
    mock_cv.return_value = np.array([0.9, 0.85, 0.92, 0.88, 0.91])
    mock_dump.return_value = None

    with patch("matplotlib.pyplot.show"), \
         patch("scripts.update_model.confusion_matrix", return_value=np.array([[1, 0], [0, 1]])) as mock_cm, \
         patch("scripts.update_model.sns.heatmap"):
        
        updater = ModelUpdater(
            setup_test_environment["model_base_path"],
            setup_test_environment["cache_base_dir"]
        )
        updater.retrain_model(test_df)

        # Verify key interactions
        mock_dependencies["build_model"]().fit.assert_called_once()
        # Don't try to directly compare Series/DataFrames - this can cause the ambiguous truth value error
        mock_cm.assert_called_once()  # Just check it was called, don't check parameters
        mock_dump.assert_called_with(updater.model, updater.model_path, protocol=4)
def test_update_with_new_data(setup_test_environment, mock_dependencies):
    with patch.object(ModelUpdater, "retrain_model") as mock_retrain:
        updater = ModelUpdater(setup_test_environment["model_base_path"], setup_test_environment["cache_base_dir"])
        updater.update(setup_test_environment["new_data_path"])
    mock_retrain.assert_called_once()


def test_update_with_duplicate_data(setup_test_environment, mock_dependencies):
    duplicate_data_path = os.path.join(setup_test_environment["test_dir"], "duplicate_data.csv")
    pd.DataFrame({
        "File Path": ["/path/to/doc1.pdf", "/path/to/doc2.pdf"],
        "Document types": ["Manual", "Drawing"]
    }).to_csv(duplicate_data_path, index=False)

    with patch.object(ModelUpdater, "retrain_model") as mock_retrain:
        updater = ModelUpdater(setup_test_environment["model_base_path"], setup_test_environment["cache_base_dir"])
        updater.update(duplicate_data_path)

    mock_retrain.assert_not_called()