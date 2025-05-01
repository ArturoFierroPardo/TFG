Función stwrd(tokens):

- stwrd() es una función que elimina las stopwords (palabras vacías) del texto tokenizado.

- Entrada: lista de palabras (tokens) como strings
- Sálida: lista de tokens sin stopwords ni símbolos

- Código:

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
def stwrd(tokens):
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words and token.isalpha()]
    return filtered_tokens

[línea 10]: Importa la librería nltk.

[línea 11]: Importa stopwords de la librería nltk

[línea 12]: Descargamos la lista de stopwords si aún no se ha hecho. Solo hace falta la primera vez.

[línea 13]: Cargamos las stopwords en inglés y las convertimos en un set para búsquedas más rápidas.

[línea 14]: Definimos la función que recibe una lista de palabras (tokens) como strings.

[línea 15]: Aplicamos un filtrado: token.lower() not in stop_words → elimina palabras vacías
                                   token.isalpha() → elimina números, símbolos, etc.

[línea 16]: Devuelve la lista limpia de tokens útiles.