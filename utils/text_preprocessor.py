import re
import nltk
import os

# Ensure NLTK uses the correct path
nltk_data_path = os.environ.get("NLTK_DATA", os.path.join(os.getcwd(), "nltk_data"))
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import threading

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self._lock = threading.Lock()  # Thread-safe lemmatization

    def preprocess(self, text):
        """Optimized text preprocessing with thread safety"""
        text = re.sub(r'\b(page\s+\d+|section\s+[ivx]+|\d+-\d+|(IEC|ISO|DIN)\s+\d+)\b', '', text, flags=re.I)
        text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
        
        with self._lock:  # Only needed if lemmatizer isn't thread-safe
            words = [self.lemmatizer.lemmatize(w) for w in text.split() if w not in self.stop_words]
        return ' '.join(words)
