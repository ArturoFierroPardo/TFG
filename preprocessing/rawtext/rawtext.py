def apply_rt(ruta):
    with open(ruta, 'r', encoding='utf-8') as f:
        raw = f.read()
    return raw
