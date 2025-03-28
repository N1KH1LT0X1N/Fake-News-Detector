import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

def clean_text(text):
    """Lowercase, remove punctuation and digits."""
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    return text

def create_vectorizer():
    """Return a configured TF-IDF vectorizer."""
    return TfidfVectorizer(stop_words='english', max_df=0.7, max_features=5000)

def create_pipeline(model):
    """Return a Pipeline of vectorizer + model."""
    return Pipeline([
        ('tfidf', create_vectorizer()),
        ('model', model)
    ])
