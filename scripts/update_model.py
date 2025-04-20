
"""CODE TO UDATE MODEL WITH NEW DATA

    New DATAFRAME should be in 'csv format'
    Check the following line: 173
    
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

# Import necessary functions from training script
from dd_OCR_cache_model import (
    build_model_pipeline,
    reshape_visual_features,
)

from utils.path_log import  CACHE_DIR
from utils.pdf_processor import PDFProcessor
from utils.preprocessing import TextProcessor

reshape_visual_features

MODEL_SAVE_PATH = '../models/cached_multimodal_doc_classifier'

class ModelUpdater:
    def __init__(self, model_base_path, cache_base_dir):
        self.model_base_path = model_base_path
        self.current_version = self._get_next_version()
        self.model_path = f"{model_base_path}_v{self.current_version}.pkl"
        self.cache_dir = os.path.join(cache_base_dir, f"v{self.current_version}")
        
        os.makedirs(self.cache_dir, exist_ok=True)
        self.text_processor = TextProcessor()
        self.text_processor.load_cache()
        self.model = self._load_previous_model()
        self.previous_cache = self._load_previous_cache()
    
    def _get_next_version(self):
        """Read and increment version from version.txt"""
        version_file = "version.txt"
        try:
            with open(version_file, "r") as f:
                version = int(f.read().strip())
        except:
            version = 1  # Start from v1 if no file exists
            
        with open(version_file, "w") as f:
            f.write(str(version + 1))
            
        return version


    def _load_previous_model(self):
        """Load model from previous version"""
        #prev_version = self.current_version - 1
        # if we’re on version 1, “previous” is also v1; otherwise v(current–1)
        prev_version = 1 if self.current_version == 1 else self.current_version - 1
        
        #prev_model_path = f"../models/cached_multimodal_doc_classifier_v{prev_version}.pkl"
        # look for the .pkl file that your tests create
        prev_model_path = f"{self.model_base_path}_v{prev_version}.pkl"
        
        if os.path.exists(prev_model_path):
            print(f"Loading model from version {prev_version}")
            return joblib.load(prev_model_path)
        else:
            raise FileNotFoundError("No previous model found")
            #return None

    def _load_previous_cache(self):
        """Load cache from previous version"""
        prev_version = 1 if self.current_version == 1 else self.current_version - 1
        #prev_cache_dir = os.path.join("../data/cache", f"v{prev_version}")
        cache_file = os.path.join(self.cache_dir, "processed_data.joblib")
        
        if os.path.exists(cache_file):
            print(f"Loading cache from version {prev_version}")
            return joblib.load(cache_file)["data_frame"]
        return pd.DataFrame()

    def process_new_data(self, new_data) -> pd.DataFrame:
        """Process and merge new data with previous cache"""
        """Load previous cache, filter out used files, preprocess & save combined."""
        
        prev = self.previous_cache
        if "File Path" in prev.columns:
        # Normalize and deduplicate
            new_data["File Path"] = new_data["File Path"].apply(os.path.normpath)
            new_data = new_data[~new_data["File Path"].isin(self.previous_cache["File Path"])]
        
        if new_data.empty:
            print("No new data to process")
            return prev

        # Process new data
        print("Processing new documents...")
        new_data["Raw_Text"] = new_data["File Path"].apply(
            lambda x: PDFProcessor.extract_text(x, self.current_version)
        )
        new_data["Processed_Text"] = new_data["Raw_Text"].apply(self.text_processor.preprocess)
        new_data["Visual_Features"] = new_data["File Path"].apply(
            lambda x: PDFProcessor.extract_visual_features(x, self.current_version)
        )

        # Merge and save new cache
        updated_data = pd.concat([self.previous_cache, new_data])
        joblib.dump(
            {"data_frame": updated_data, "version": self.current_version},
            os.path.join(self.cache_dir, "processed_data.joblib")
        )
        return updated_data

    def retrain_model(self, df):
        """ Retrain the model with updated data """
        # Prepare labels
        labels = df["Document types"].unique()
        label_dict = {label: idx for idx, label in enumerate(labels)}
        df["Label"] = df["Document types"].map(label_dict)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            df[['Processed_Text', 'Visual_Features']],
            df['Label'],
            test_size=0.2,
            stratify=df['Label'],
            random_state=42
        )

        # Ensure pipeline is consistent with training
        self.model = build_model_pipeline()
        print("Retraining model with updated data...")
        self.model.fit(X_train, y_train)

        # Evaluate updated model
        try:
            y_pred = self.model.predict(X_test)
            print(f"\nUpdated Model Accuracy: {accuracy_score(y_test, y_pred):.2%}")
        except Exception as e:
            print(f"\nSkipping accuracy calculation (bad y_test/y_pred): {e}")
            
        print(classification_report(y_test, y_pred, target_names=label_dict.keys()))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_dict.keys(), yticklabels=label_dict.keys())
        plt.title('Updated Model Confusion Matrix')
        plt.show()
        
            # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
        self.model,
            df[['Processed_Text', 'Visual_Features']],
            df['Label'],
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )
        print(f"Cross-Validation Accuracy: {np.mean(cv_scores):.2%} (±{np.std(cv_scores):.2%})")

        # Save the updated model with versioning
        joblib.dump(self.model, self.model_path, protocol=4)
        print(f"Saved new model version v{self.current_version}")

    def update(self, new_data_path):
        """ Main function to process new data and update the model """
        new_data = pd.read_csv(new_data_path)
        updated_data = self.process_new_data(new_data)
        
        if not updated_data.equals(self.previous_cache):
            self.retrain_model(updated_data)
        else:
            print("No model update needed")

if __name__ == "__main__":
    # Initialize version if first run
    if not os.path.exists("version.txt"):
        with open("version.txt", "w") as f:
            f.write("2")  # Start versioning
    
    NEW_DATA_PATH = r'C:\Users\ashu\Desktop\Python Workspace\training_model\data\training\batc_2_cosmos_ABB.csv'  # Adjust path accordingly
    updater = ModelUpdater(MODEL_SAVE_PATH, CACHE_DIR)
    updater.update(NEW_DATA_PATH)
