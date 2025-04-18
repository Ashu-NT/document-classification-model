import re
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
