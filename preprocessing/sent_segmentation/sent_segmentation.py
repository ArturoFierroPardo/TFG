import spacy
nlp = spacy.load("en_core_web_sm")
def apply_segment(raw):
    doc = nlp(raw)
    sentences = list(doc.sents)
    return sentences