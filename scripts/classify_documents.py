
import pandas as pd
import joblib
import os
from pathlib import Path
import threading
from joblib import Parallel, delayed
from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from utils.feature_extractor import PDFFeatureExtractor
from utils.text_preprocessor import TextPreprocessor

from dd_OCR_cache_model import reshape_visual_features


reshape_visual_features


def process_pdf(pdf_path, root_dir, preprocessor, model, class_labels):
    """Parallel PDF processing function"""
    try:
        path_obj = Path(pdf_path)
        relative_path = os.path.relpath(path_obj.parent, root_dir)
        
        return {
            'file_name': path_obj.name,
            'file_path': str(path_obj),
            'main_folder': path_obj.parts[4] if len(path_obj.parts) > 4 else '',
            'parent_folder': path_obj.parent.name,
            'relative_folder': relative_path,
            'component_name': relative_path.split(os.sep)[0] if os.sep in relative_path else relative_path,
            'document_type': class_labels[model.predict(pd.DataFrame([{
                'Processed_Text': preprocessor.preprocess(PDFFeatureExtractor.extract_text(pdf_path)),
                'Visual_Features': PDFFeatureExtractor.extract_visual_features(pdf_path)
            }]))[0]]
        }
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return None

def classify_pdfs(root_dir, model_path, output_csv=None, output_excel=None):
    """Optimized classification with parallel execution"""
    model = joblib.load(model_path)
    preprocessor = TextPreprocessor()
    classifier = model.steps[-1][1]
    class_labels = classifier.classes_
    
    # Collect files in main thread
    pdf_files = []
    non_pdf_files = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            path = os.path.join(root, f)
            (pdf_files if path.lower().endswith('.pdf') else non_pdf_files).append(path)
    
    # Parallel execution
    results = Parallel(n_jobs=os.cpu_count(), backend='threading', verbose=10)(
        delayed(process_pdf)(pdf, root_dir, preprocessor, model, class_labels)
        for pdf in pdf_files
    )
    
    # Save results
    result_df = pd.DataFrame([r for r in results if r is not None])
    if output_csv: result_df.to_csv(output_csv, index=False)
    if output_excel: result_df.to_excel(output_excel, index=False)
    
    # Save non-PDF files list
    pd.DataFrame({'non_pdf_files': non_pdf_files}).to_excel(
        "../data/non_pdf_files/3212_hug.xlsx", index=False)
    
    return result_df

if __name__ == "__main__":
    model_path = "../models/cached_multimodal_doc_classifier_v1.pkl"
    input_directory = r"C:\Users\ashu\OneDrive - dintegra\David Garcia Cumplido's files - Teresita (10-02-2025)"
    output_file = "../data/classify_data/3212_hug.csv"
    output_excel = "../data/classify_data/3212_hug.xlsx"
    
    classification_results = classify_pdfs(
        input_directory,
        model_path,
        output_file,
        output_excel
    )
    print("\nClassification Results:")
    print(classification_results.head())