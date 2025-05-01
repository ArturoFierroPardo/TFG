import spacy
nlp = spacy.load("en_core_web_sm")
def apply_lemm(tokens):
    doc = nlp(" ".join(tokens))
    lemas = []
    for token in doc:
        if token.is_alpha:
            lemas.append(token.lemma_)
    return lemas