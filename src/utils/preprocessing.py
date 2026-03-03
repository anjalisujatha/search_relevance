import re
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()

STOPWORDS = {
    'i', 'me', 'my', 'a', 'an', 'the', 'and', 'or', 'but', 'are',
    'is', 'was', 'to', 'of', 'in', 'it', 'its', 'this', 'that',
    'for', 'on', 'at', 'be', 'by', 'as', 'up', 'do', 'so', 'if', 's'
}


def normalize(text):
    """Lowercase, remove punctuation, strip stopwords, and lemmatize."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [lemmatizer.lemmatize(t) for t in text.split() if t not in STOPWORDS]
    return tokens
