
import pandas as pd
import joblib
import os
from pathlib import Path

from utils.feature_extractor import PDFFeatureExtractor
from utils.text_preprocessor import TextPreprocessor

from dd_OCR_cache_model import reshape_visual_features


reshape_visual_features

def classify_pdfs(root_dir, model_path, output_csv=None, output_excel=None):
    """
    Classify PDF documents in directory structure and return results as DataFrame.
    
    Parameters:
    root_dir (str): Root directory to search for PDF files
    model_path (str): Path to trained model (.pkl file)
    output_csv (str): Optional path to save results as CSV
    output_excel(str): Optional path to save results as Excel
    
    Returns:
    pd.DataFrame: Classification results with metadata
    """
    # Load trained model
    print("Loading model from:", model_path)
    model = joblib.load(model_path)
    text_preprocessor = TextPreprocessor()
    
    # Get class labels from model
    classifier = model.steps[-1][1]  # Get final classifier
    class_labels = classifier.classes_
    
    # Collect PDF files
    pdf_files = []
    non_pdf_files = []  # This will store non-PDF files
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.lower().endswith('.pdf'):
                pdf_files.append(file_path)
            else:
                non_pdf_files.append(file_path)  # Collect non-PDF files
    
    results = []
    
    for pdf_path in pdf_files:
        try:
            # Extract file metadata
            path_obj = Path(pdf_path)
            relative_path = os.path.relpath(path_obj.parent, root_dir)
            file_info = {
                'file_name': path_obj.name,
                'file_path': str(path_obj),
                'main_folder': path_obj.parts[4] if len(path_obj.parts) > 0 else '', # [4] to be verified if there is David.....
                'parent_folder': path_obj.parent.name,
                'relative_folder': os.path.relpath(path_obj.parent, root_dir),
                'component_name':relative_path.split(os.sep)[0] if os.sep in relative_path else "No Subfolder"
            }
            
            # Extract features
            raw_text = PDFFeatureExtractor.extract_text(pdf_path)
            processed_text = text_preprocessor.preprocess(raw_text)
            visual_features = PDFFeatureExtractor.extract_visual_features(pdf_path)
            
            # Create input dataframe
            input_data = pd.DataFrame([{
                'Processed_Text': processed_text,
                'Visual_Features': visual_features
            }])
            
            # Predict
            pred_idx = model.predict(input_data)[0]
            file_info['document_type'] = class_labels[pred_idx]
            
            results.append(file_info)
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            continue
    
    # Create and save results
    result_df = pd.DataFrame(results)
    
    if output_csv:
        result_df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
    
    if output_excel:
        result_df.to_excel(output_excel, index=False)
        print(f"Results saved to {output_excel}")
        
    # Keep track of non-PDF files
    non_pdf_info = pd.DataFrame({'non_pdf_files': non_pdf_files})
    non_pdf_info.to_excel("../data/non_pdf_files/Batch_2_Cosmos_non_pdfs_test.xlsx", index=False)  # Save non-PDF files list to EXCEL
    print("Non-PDF files saved to non_pdf_files")
    
    return result_df

if __name__ == "__main__":
    # Confg paths
    model_path = "../models/cached_multimodal_doc_classifier_v1.pkl" # Model path
    input_directory = r"C:\Users\ashu\OneDrive - dintegra\David Garcia Cumplido's files - Batch 2" # Always check for input folder path
    output_file = "../data/classify_data/Batch_2_COSMOS_test_pdfs.csv" # rename as per project
    output_file_excel = "../data/classify_data/Batch_2_COSMOS_test_pdfs.xlsx" # rename as per project (EXCEL)
    
    classification_results = classify_pdfs(
        root_dir=input_directory,
        model_path=model_path,
        output_csv=output_file,
        output_excel = output_file_excel
    )
    
    print("\nClassification Results:")
    print(classification_results.head())