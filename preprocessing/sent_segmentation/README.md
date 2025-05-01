Función apply_segment(raw):

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

[línea 10]: Importa la librería spaCy.

[línea 11]: Cargamos el modelo de procesamiento en inglés. Solo lo hacemos una vez (fuera de la función) para ahorrar tiempo y memoria si la función se llama muchas veces

[línea 12]: Definimos la función que toma el texto crudo (str) como entrada.

[línea 13]: Procesamos el texto con spaCy, obteniendo un objeto Doc con toda la estructura lingüística.

[línea 14]: Recorremos todas las frases (doc.sents), extraemos el texto y quitamos espacios en blanco.

[línea 15]:  Devolvemos la lista de oraciones