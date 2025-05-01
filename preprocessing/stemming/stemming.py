import nltk
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer("english")
def apply_stemm(tokens):
    stems = []
    for token in tokens:
        stem = stemmer.stem(token)
        stems.append(stem)
    return stems