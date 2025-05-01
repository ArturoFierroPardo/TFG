Función apply_lemm(tokens):

- apply_lemm() es una función que aplica aplica lematización a una lista de tokens usando spaCy.

- Entrada: Lista de palabras como strings (ya preprocesadas).
- Sálida: Lista de lemas (forma base de las palabras).

- Código:

import spacy
nlp = spacy.load("en_core_web_sm")
def apply_lemm(tokens):
    doc = nlp(" ".join(tokens))
    lemas = []
    for token in doc:
        if token.is_alpha:
            lemas.append(token.lemma_)
    return lemas

[línea 10]: Importamos la librería nltk.

[línea 11]: Cargamos el modelo de inglés una vez (evitamos cargarlo en cada llamada).

[línea 12]: Definimos la función que recibe una lista de palabras (tokens) como strings.

[línea 13]: Unimos los tokens en una cadena y procesamos el texto con spaCy, lo que permite analizarlo con contexto gramatical.

[línea 14]: Inicializamos la lista vacía que contendrá los stems resultantes.

[línea 15]: Recorre cada token del doc de tokens.

[línea 16]: Usamos is_alpha para excluir signos y números.

[línea 17]: Extraemos el lema con .lemma_ y lo guardamos en la lista.

[línea 18]: Devolvemos la lista final con todos los lemas.

