Función tokenize_text(raw):

- tokenize_text() es una función que tokeniza el texto en palabras usando NLTK (word_tokenize)

- Entrada: texto completo sin procesar
- Sálida: lista de palabras y signos de puntuación como tokens

- Código:

import nltk
from nltk.tokenize import word_tokenize
def tokenize_text(raw):
    tokens = word_tokenize(raw)
    return tokens

[línea 10]: Importa la librería nltk.

[línea 11]: importa word_tokenize de la librería nltk

[línea 12]: Definimos la función que toma el texto crudo (str) como entrada.

[línea 13]: Tokenizamos el texto en palabras, signos, etc.

[línea 14]: Devolvemos la lista de tokens como resultado.