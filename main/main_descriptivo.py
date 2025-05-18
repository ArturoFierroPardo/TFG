import sys
import os

sys.path.append("C:/Users/artur/Desktop/TFG/preprocessing")
sys.path.append("C:/Users/artur/Desktop/TFG/rnn")

from rawtext.rawtext import apply_rt
from sent_segmentation.sent_segmentation import apply_segment
from tokenization.tokenization import tokenize_text
from pos_tag_tokens.pos_tag_tokens import tg_tk
from remove_stopwords.remove_stopwords import stwrd
from stemming.stemming import apply_stemm
from lemmatization.lemmatization import apply_lemm
from line_by_line import modelo2_lbl
from one_word_in_one_word_out import modelo1_owi_owo
from two_words_in_one_word_out import modelo3_twi_owo


def main():
    raw = apply_rt("C:/Users/artur/Desktop/TFG/textos/textoplano.txt")
    print("TEXTO (primeros 300 caracteres):")
    print(raw[:300], "\n")

    sentences = apply_segment(raw)
    print(f"{len(sentences)} oraciones detectadas:")
    for s in sentences[:3]: 
        print(s)
    print("\n")

    tokens = tokenize_text(raw)
    print(f"{len(tokens)} tokens generados:")
    print(tokens[:10], "\n")

    tags = tg_tk(tokens)
    print("POS tagging de los primeros 10 tokens:")
    for tag in tags[:10]:
        print(tag[0], "=>", tag[1], "=>", tag[2])
    print("\n")

    filtered_tokens = stwrd(tokens)
    print(f"{len(filtered_tokens)} tokens útiles tras filtrar stopwords:")
    print(filtered_tokens[:10], "\n")

    stems = apply_stemm(filtered_tokens)
    print(f"Stems generados:")
    print(stems[:10], "\n")

    lemas = apply_lemm(filtered_tokens)
    print(f"Lemmas generados:")
    print(lemas[:10], "\n")

    limited_lemmas = lemas[:50]

    print(modelo1_owi_owo(limited_lemmas, 300, 50, 'black'), "\n")

    print(modelo2_lbl(limited_lemmas, 300, 49, 'black'), "\n")
    
    print(modelo3_twi_owo(limited_lemmas, 300, 48, 'hole massive'), "\n")

if __name__ == "__main__":
    main()