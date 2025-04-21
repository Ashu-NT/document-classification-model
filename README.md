# Document Classification System

This project implements a document classification system that categorizes PDF documents into predefined classes such as Manual, Delivery Documentation, Datasheet, Drawing, and Certificates. The system uses a combination of text and visual features extracted from PDFs to train a machine learning model. The model can be updated with new data and used to classify documents in a directory structure.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Code Structure](#code-structure)
3. [Features](#Features)
4. [Installation](#installation)
5. [Usage](#usage)
   - [Training the Model](#training-the-model)
   - [Updating the Model](#updating-the-model)
   - [Classifying Documents](#classifying-documents)
6. [Configuration](#configuration)
7. [Contributing](#contributing)

---

## Project Overview

The Document Classification System is designed to automate the categorization of PDF documents. It uses a multimodal approach, combining text and visual features extracted from PDFs, to train a Random Forest classifier. The system supports:

- **Training**: Train a model using labeled PDF documents.
- **Updating**: Incrementally update the model with new data.
- **Classification**: Classify PDF documents in a directory structure.

The system is optimized for handling large datasets and includes caching mechanisms to speed up feature extraction and processing.

---

## Code Structure

The project is organized into three main scripts:

1. **Training Script (`dd_OCR_cache_model.py`)**:
   - Handles feature extraction (text and visual features).
   - Preprocesses text data.
   - Trains and evaluates the classification model.
   - Saves the trained model and cache.

2. **Model Update Script (`update_model.py`)**:
   - Loads the existing model and cache.
   - Processes new data and merges it with the existing dataset.
   - Retrains the model with the updated dataset.
   - Saves the new model version.

3. **Classification Script (`classify_documents.py`)**:
   - Loads the trained model.
   - Classifies PDF documents in a directory structure.
   - Saves classification results to CSV/Excel files.

---
## Features
- Hybrid text extraction with OCR fallback
- Visual feature analysis (edge detection, contour analysis, layout detection)
- Text preprocessing (lemmatization, stop-word removal)
- Classification using a trained machine learning model
- Error logging for PDF processing issues
- Output results in CSV and Excel formats

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR (install from [here](https://github.com/tesseract-ocr/tesseract))
- Required Python packages (install via `pip install -r requirements.txt`)

---

## Usage

1. **Training the Model**
    1. Prepare your training data in a CSV file with columns:
        + `File Path:` Path to the PDF file.
        + `Document types:` Label for the document (e.g., Manual, Datasheet).
    2. Run the training script:
        This will:
        + Extract features from the PDFs.
        + Train the model.
        + Save the model and cache to the specified paths.

2. **Updating the Model**
    1. Prepare new data in a CSV file with the same format as the training data.
    2. Run the update script:
        This will:
        + Load the existing model and cache.
        + Process and merge new data.
        + Retrain the model with the updated dataset.
        + Save the new model version.

3. **Classifying Documents**
    1. Organize your PDFs in a directory structure where subfolders represent system codes (e.g., 4710.11243.98673).
    2. Run the classification script:
        This will:
        + Load the trained model.
        + Classify PDFs in the specified directory.
        + Save results to CSV/Excel files.

### Error Logging
    Errors during PDF processing are logged to ../data/error_log_pdf/error_log.csv.

---

## Configuration
Key Configuration Parameters
- THRESHOLD_OCR: Minimum text length threshold for OCR fallback.
- CACHE_DIR: Directory for caching processed data.
- MODEL_SAVE_PATH: Path to save the trained model.
- TRAINING_DATA_PATH: Path to the training data CSV file.
- PYTESSERACT_PATH: Path to the Tesseract OCR executable.

---

## Contributing

Contributions are welcome! Please follow these steps:
- Fork the repository.
- Create a new branch for your feature/bugfix.
- Commit your changes.
- Submit a pull request.