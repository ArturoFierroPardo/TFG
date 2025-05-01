import spacy
nlp = spacy.load("en_core_web_sm")
def tg_tk(tokens):
    doc = nlp(" ".join(tokens))
    pos_info = []
    for token in doc:
        if token.is_alpha:
            tokenstag = (token.text, token.pos_, token.tag_)
            pos_info.append(tokenstag)

    return pos_info