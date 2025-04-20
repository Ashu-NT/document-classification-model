# utils/preprocessing.py
import re
import nltk
import os

# Ensure NLTK uses the correct path
nltk.data.path.append(os.environ.get("NLTK_DATA", ""))
    
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import joblib
import hashlib
import os

CACHE_DIR = os.path.abspath('../data/cache')

class TextProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.cache = {}
        self.cache_file = os.path.join(CACHE_DIR, 'text_preprocessor_cache.joblib')

    def _text_hash(self, text):
        """Generate hash for text caching"""
        return hashlib.md5(text.encode()).hexdigest()

    def preprocess(self, text):
        """Cached text preprocessing"""
        text = str(text)
        text_hash = self._text_hash(text)
        
        if text_hash in self.cache:
            return self.cache[text_hash]
        
        processed = self._process_text(text)
        self.cache[text_hash] = processed
        return processed

    def _process_text(self, text):
        """Text cleaning and normalization"""
        text = re.sub(r'\b(page\s+\d+|section\s+[ivx]+)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d+-\d+\b', 'PARTNUM', text)
        text = re.sub(r'\b(IEC|ISO|DIN)\s+\d+\b', 'STANDARD', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
        words = [self.lemmatizer.lemmatize(w) for w in text.split() if w not in self.stop_words]
        return ' '.join(words)

    def save_cache(self):
        """Persist text cache"""
        joblib.dump(self.cache, self.cache_file)

    def load_cache(self):
        """Load text cache"""
        if os.path.exists(self.cache_file):
            self.cache = joblib.load(self.cache_file)