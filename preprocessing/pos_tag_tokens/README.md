Función tg_tk(tokens):

- tg_tk() es una función que aplica etiquetado gramatical (POS tagging) a una lista de tokens usando spaCy.

- Entrada: lista de palabras (tokens) como strings.
- Sálida: lista de tuplas (token, pos, tag)

- Código:

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

[línea 10]: Importa la librería spaCy.

[línea 11]: Carga el modelo de idioma inglés preentrenado en_core_web_sm.

[línea 12]: Define la función tg_tk, que recibe como argumento una lista de tokens (str).

[línea 13]: Une todos los tokens con espacios (" ".join(tokens)) para reconstruir el texto original. Luego lo procesa con nlp(), lo que devuelve un objeto Doc con toda la información lingüística.

[línea 14]: Crea una lista vacía donde vamos a guardar el resultado final: una lista de tuplas con la info gramatical de cada palabra.

[línea 15]: Recorre cada token del documento procesado por spaCy.

[línea 16]: Esta condición filtra los tokens para que solo se incluyan los que son palabras reales (letras).

[línea 17]: Crea una tupla con tres elementos: token.text: la palabra original
                                               token.pos_: la categoría gramatical general (NOUN, VERB, ADJ, etc.)
                                               token.tag_: la etiqueta gramatical específica (NN, VBD, JJ, etc.)

[línea 18]: Añade esa tupla a la lista pos_info.

[línea 19]: Devuelve la lista completa con el etiquetado POS.
