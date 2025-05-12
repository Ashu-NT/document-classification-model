
import pandas as pd
import joblib
import os
from pathlib import Path
from joblib import Parallel, delayed
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from utils.feature_extractor import PDFFeatureExtractor
from utils.text_preprocessor import TextPreprocessor

from dd_OCR_cache_model import reshape_visual_features
from utils.path_log import CONFIDENCE_THRESHOLD

reshape_visual_features


def process_pdf(pdf_path, root_dir, preprocessor, model, class_labels):
    """Parallel PDF processing function"""
    try:
        path_obj = Path(pdf_path)
        relative_path = os.path.relpath(path_obj.parent, root_dir)
        
        # process text and visual features
        Processed_Text = preprocessor.preprocess(PDFFeatureExtractor.extract_text(pdf_path)),
        Visual_Features = PDFFeatureExtractor.extract_visual_features(pdf_path)
        
        # prepare data for prediction
        input_df = pd.DataFrame([{
            'Processed_Text': Processed_Text,
            'Visual_Features': Visual_Features
        }])
        
        # Get predictions and probabilities
        pred_idx = model.predict(input_df)[0]
        pred_probas = model.predict_proba(input_df)[0]
        confidence = pred_probas[pred_idx] * 100  # Convert to percentage
        
        return {
            'file_name': path_obj.name,
            'file_path': str(path_obj),
            'main_folder': path_obj.parts[4] if len(path_obj.parts) > 4 else '',
            'parent_folder': path_obj.parent.name,
            'relative_folder': relative_path,
            'component_name': relative_path.split(os.sep)[0] if os.sep in relative_path else relative_path,
            'document_type': class_labels[pred_idx],
            'confidence': confidence
        }
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return None

def classify_pdfs(root_dir, model_path, output_csv=None, output_excel=None,confidence_threshold=None):
    """Optimized classification with parallel execution"""
    model = joblib.load(model_path)
    preprocessor = TextPreprocessor()
    
    # Access the final classifier to get class labels
    # Extract classifier from the pipeline
    # Find the last step containing the classifier
    for i in range(len(model.steps) - 1, -1, -1):
        step_name, step_obj = model.steps[i]
        if hasattr(step_obj, 'classes_'):
            classifier = step_obj
            break
    
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
    
   # Filter and sort results
    result_df = pd.DataFrame([r for r in results if r is not None])
    
    if not result_df.empty:
        # Format confidence as percentage with 2 decimal places
        result_df['confidence'] = result_df['confidence'].round(2)
        
        # If confidence threshold is provided, flag low confidence predictions
        if confidence_threshold is not None:
            result_df['needs_review'] = result_df['confidence'] < confidence_threshold
            
    # Save results
    if output_csv: result_df.to_csv(output_csv, index=False)
    if output_excel: result_df.to_excel(output_excel, index=False)
    
    # Save non-PDF files list
    os.makedirs(os.path.join("..", "data", "non_pdf_files"), exist_ok=True)
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
        output_excel,
        CONFIDENCE_THRESHOLD
    )
    print("\nClassification Results:")
    print(classification_results.head())
    
    if not classification_results.empty and 'needs_review' in classification_results.columns:
        review_count = classification_results['needs_review'].sum()
        total_count = len(classification_results)
        print(f"\nDocuments that need review: {review_count}/{total_count} ({review_count/total_count:.1%})")
        print("Low confidence documents:")
        print(classification_results[classification_results['needs_review']][['file_name', 'document_type', 'confidence']].head(10))