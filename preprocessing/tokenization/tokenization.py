import nltk
from nltk.tokenize import word_tokenize
def tokenize_text(raw):
    tokens = word_tokenize(raw)
    return tokens