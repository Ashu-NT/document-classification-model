# scripts/dd_OCR_cache_model.py

""" 
    Check utils.path_log.py and update as necessary
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from utils.preprocessing import TextProcessor
from utils.pdf_processor import PDFProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline
import gc

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from utils.path_log import TRAINING_DATA_PATH, MODEL_SAVE_PATH,CACHE_DIR
from utils.pdf_processor import PDFProcessor
from utils.preprocessing import TextProcessor


PYTESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Cache version (increment when changing processing logic)
PROCESSING_VERSION = "v2.1"
os.makedirs(CACHE_DIR, exist_ok=True)

# =============================================================================
# Data Management
# =============================================================================
class DataManager:
    def __init__(self, processing_version):
        self.processing_version = processing_version
        self.cache_file = os.path.join(CACHE_DIR, 'processed_data.joblib')

    def needs_processing(self, df):
        """Check if data needs reprocessing"""
        if not os.path.exists(self.cache_file):
            return True

        cached = joblib.load(self.cache_file)
        if cached['version'] != self.processing_version:
            return True

        current_sigs = df['File Path'].apply(PDFProcessor.file_signature)
        if not np.array_equal(current_sigs.values, cached['file_signatures']):
            return True

        return False

    def load_or_process(self, df, text_processor):
        """Main data processing workflow"""
        if not self.needs_processing(df):
            print("Loading cached processed data...")
            return joblib.load(self.cache_file)['data_frame']

        print("Processing data (this might take a while)...")
        df["Raw_Text"] = df["File Path"].apply(
            lambda x: PDFProcessor.extract_text(x, self.processing_version)
        )
        df["Processed_Text"] = df["Raw_Text"].apply(text_processor.preprocess)
        df["Visual_Features"] = df["File Path"].apply(
            lambda x: PDFProcessor.extract_visual_features(x, self.processing_version)
        )

        cache_data = {
            'data_frame': df,
            'file_signatures': df['File Path'].apply(PDFProcessor.file_signature).values,
            'version': self.processing_version
        }
        joblib.dump(cache_data, self.cache_file)
        
        return df


# =============================================================================
# Top-Level Functions: NEEDED DURING DUMP
# =============================================================================

def reshape_visual_features(x):
    return np.array(x.tolist()).reshape(-1, 4)

# =============================================================================
# Model Pipeline
# =============================================================================
def build_model_pipeline():
    text_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=25000,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            sublinear_tf=True
        )),
        ('svd', TruncatedSVD(n_components=100, random_state=42))
    ])

    visual_pipe = Pipeline([
        ('reshape', FunctionTransformer(reshape_visual_features)),  # Use named function
        ('scaler', RobustScaler())
    ])

    return make_imb_pipeline(
        ColumnTransformer([
            ('text', text_pipe, 'Processed_Text'),
            ('visual', visual_pipe, 'Visual_Features')
        ]),
        SMOTE(random_state=42),
        RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
    )

# =============================================================================
# Main Workflow
# =============================================================================
def main():
    # Initialize components
    text_processor = TextProcessor()
    text_processor.load_cache()
    data_manager = DataManager(PROCESSING_VERSION)

    # Load and prepare data
    df = pd.read_csv(TRAINING_DATA_PATH)
    df["File Path"] = df["File Path"].apply(os.path.normpath)
    df = data_manager.load_or_process(df, text_processor)
    text_processor.save_cache()

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

    # Build and train model
    model = build_model_pipeline()
    print("Training model...")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(classification_report(y_test, y_pred, target_names=label_dict.keys()))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_dict.keys(), yticklabels=label_dict.keys())
    plt.title('Confusion Matrix')
    plt.show()

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        model,
        df[['Processed_Text', 'Visual_Features']],
        df['Label'],
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.2%} (±{cv_scores.std():.2%})")

    # Save model
    joblib.dump(model, MODEL_SAVE_PATH, protocol=4)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")
    

def cleanup():
    """ This function is meant to clean up memory and prevent GUI-related issues when using Tkinter.
        It creates a hidden Tkinter root window, immediately destroys it, and then forces garbage collection 
    """
    import tkinter as tk
    root = tk.Tk()
    root.destroy()
    gc.collect()

if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup()