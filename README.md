**RAWTEXT** --> Función apply_rt(ruta):

- apply_rt() es una función simple y reutilizable que permite cargar archivos .txt como texto plano en Python. Se utiliza como paso inicial en cualquier pipeline de procesamiento de lenguaje natural, permitiendo trabajar sobre el contenido textual del archivo. Se puede extender para que lea otro tipo de archivos o urls.

- Entrada: ruta de un archivo local / string / URL de internet (Había que cambiar el código)
- Sálida: Devuelve un string con el contenido completo del archivo leído.

- Código:

def apply_rt(ruta): 
    with open(ruta, 'r', encoding='utf-8') as f:
        raw = f.read()
    return raw

[línea 10]: Define una función llamada rawtext que recibe como argumento ruta (un string con el path al archivo).

[línea 11]: Abre el archivo en modo lectura 'r', usando codificación 'utf-8'. with asegura que se cierre automáticamente cuando termine. encoding especifica cómo deben interpretarse los caracteres almacenados en el archivo. Los archivos .txt no guardan letras directamente, sino números binarios que representan letras según una codificación determinada.

[línea 12]: Lee todo el contenido del archivo y lo guarda como string en la variable raw.

[línea 13]: Devuelve el texto como resultado de la función.




**SENTENCE SEGMENTATION** --> Función apply_segment(raw):

- apply_segment() es una función que segmenta un texto en frases completas usando el modelo `en_core_web_sm` de spaCy.

- Entrada: texto completo sin procesar
- Sálida: lista de oraciones

- Código:

import spacy
nlp = spacy.load("en_core_web_sm")
def apply_segment(raw):
    doc = nlp(raw)
    sentences = list(doc.sents)
    return sentences

[línea 35]: Importa la librería spaCy.

[línea 36]: Cargamos el modelo de procesamiento en inglés. Solo lo hacemos una vez (fuera de la función) para ahorrar tiempo y memoria si la función se llama muchas veces

[línea 37]: Definimos la función que toma el texto crudo (str) como entrada.

[línea 38]: Procesamos el texto con spaCy, obteniendo un objeto Doc con toda la estructura lingüística.

[línea 39]: Recorremos todas las frases (doc.sents), extraemos el texto y quitamos espacios en blanco.

[línea 40]:  Devolvemos la lista de oraciones




**TOKENIZATION** --> Función tokenize_text(raw):

- tokenize_text() es una función que tokeniza el texto en palabras usando NLTK (word_tokenize)

- Entrada: texto completo sin procesar
- Sálida: lista de palabras y signos de puntuación como tokens

- Código:

import nltk
from nltk.tokenize import word_tokenize
def tokenize_text(raw):
    tokens = word_tokenize(raw)
    return tokens

[línea 66]: Importa la librería nltk.

[línea 67]: importa word_tokenize de la librería nltk

[línea 68]: Definimos la función que toma el texto crudo (str) como entrada.

[línea 69]: Tokenizamos el texto en palabras, signos, etc.

[línea 70]: Devolvemos la lista de tokens como resultado.




**PART OF SPEECH (POS)** --> Función tg_tk(tokens):

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
            tokenstag = (token.text, token.pos_, token.tag_)
            pos_info.append(tokenstag)
    return pos_info

[línea 94]: Importa la librería spaCy.

[línea 95]: Carga el modelo de idioma inglés preentrenado en_core_web_sm.

[línea 96]: Define la función tg_tk, que recibe como argumento una lista de tokens (str).

[línea 97]: Une todos los tokens con espacios (" ".join(tokens)) para reconstruir el texto original. Luego lo procesa con nlp(), lo que devuelve un objeto Doc con toda la información lingüística.

[línea 98]: Crea una lista vacía donde vamos a guardar el resultado final: una lista de tuplas con la info gramatical de cada palabra.

[línea 99]: Recorre cada token del documento procesado por spaCy.

[línea 100]: Crea una tupla con tres elementos: token.text: la palabra original
                                               token.pos_: la categoría gramatical general (NOUN, VERB, ADJ, etc.)
                                               token.tag_: la etiqueta gramatical específica (NN, VBD, JJ, etc.)

[línea 101]: Añade esa tupla a la lista pos_info.

[línea 102]: Devuelve la lista completa con el etiquetado POS.




**STOP WORD REMOVAL** --> Función stwrd(tokens):

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

[línea 136]: Importa la librería nltk.

[línea 137]: Importa stopwords de la librería nltk

[línea 138]: Descargamos la lista de stopwords si aún no se ha hecho. Solo hace falta la primera vez.

[línea 139]: Cargamos las stopwords en inglés y las convertimos en un set para búsquedas más rápidas.

[línea 140]: Definimos la función que recibe una lista de palabras (tokens) como strings.

[línea 141]: Aplicamos un filtrado: token.lower() not in stop_words → elimina palabras vacías
                                   token.isalpha() → elimina números, símbolos, etc.

[línea 142]: Devuelve la lista limpia de tokens útiles.




**STEMMING** --> Función apply_stemm(tokens):

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

[línea 171]: Importamos la librería nltk.

[línea 172]: Importamos el algoritmo SnowballStemmer, que mejora al clásico PorterStemmer.

[línea 173]: Creamos el objeto stemmer usando reglas morfológicas para el idioma inglés. 

[línea 174]: Definimos la función stemming(), que recibe una lista de tokens como str.

[línea 175]: Inicializamos la lista vacía que contendrá los stems resultantes.

[línea 176]: Recorre cada token de la lista de tokens.

[línea 177]: Aplicamos el método .stem()

[línea 178]: Guardamos el resultado en la lista stems

[línea 179]: Devolvemos la lista final con todas las raíces.




**LEMMATIZATION** --> Función apply_lemm(tokens):

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

[línea 211]: Importamos la librería nltk.

[línea 212]: Cargamos el modelo de inglés una vez (evitamos cargarlo en cada llamada).

[línea 213]: Definimos la función que recibe una lista de palabras (tokens) como strings.

[línea 214]: Unimos los tokens en una cadena y procesamos el texto con spaCy, lo que permite analizarlo con contexto gramatical.

[línea 215]: Inicializamos la lista vacía que contendrá los stems resultantes.

[línea 216]: Recorre cada token del doc de tokens.

[línea 217]: Usamos is_alpha para excluir signos y números.

[línea 218]: Extraemos el lema con .lemma_ y lo guardamos en la lista.

[línea 18]: Devolvemos la lista final con todos los lemas.
