import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess(self, text):
        """Enhanced technical text preprocessing"""
        text = str(text)
        # Remove technical artifacts
        text = re.sub(r'\b(page\s+\d+|section\s+[ivx]+)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d+-\d+\b', 'PARTNUM', text)  # Part numbers
        text = re.sub(r'\b(IEC|ISO|DIN)\s+\d+\b', 'STANDARD', text)
        # Basic cleaning
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        # Lemmatization
        words = [self.lemmatizer.lemmatize(w) for w in text.split() if w not in self.stop_words]
        return ' '.join(words)
