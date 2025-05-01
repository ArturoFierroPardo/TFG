import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
def stwrd(tokens):
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words and token.isalpha()]
    return filtered_tokens