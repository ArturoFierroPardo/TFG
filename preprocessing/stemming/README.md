Función apply_stemm(tokens):

- apply_stemm() es una función que aplica stemming (reducción a raíz) a una lista de tokens en inglés usando el algoritmo snowball.

- Entrada: Lista de palabras como strings (ya preprocesadas).
- Sálida: Lista de raíces (stems) de las palabras.

- Código:

import nltk
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer("english")
def apply_stemm(tokens):
    stems = []
    for token in tokens:
        stem = stemmer.stem(token)
        stems.append(stem)
    return stems

[línea 10]: Importamos la librería nltk.

[línea 11]: Importamos el algoritmo SnowballStemmer, que mejora al clásico PorterStemmer.

[línea 12]: Creamos el objeto stemmer usando reglas morfológicas para el idioma inglés. 

[línea 13]: Definimos la función stemming(), que recibe una lista de tokens como str.

[línea 14]: Inicializamos la lista vacía que contendrá los stems resultantes.

[línea 15]: Recorre cada token de la lista de tokens.

[línea 16]: Aplicamos el método .stem()

[línea 17]: Guardamos el resultado en la lista stems

[línea 18]: Devolvemos la lista final con todas las raíces.

