# setup.py
from setuptools import setup, find_packages

setup(
    name="document_classification_system",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "joblib",
        "pymupdf",
        "pytesseract",
        "opencv-python",
        "nltk",
        "imbalanced-learn",
        "seaborn",
        "matplotlib",
    ],
)