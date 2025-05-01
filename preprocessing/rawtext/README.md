Función apply_rt(ruta):

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