"""
Filtra el dataset KELM por tematica STEM y extrae un subconjunto de filas en
formato compatible con los pipelines (columnas Structured_Data / Human_Reference).

Requisitos:
    pip install datasets pandas tqdm

Uso:
    python kelm.py
"""
import pandas as pd
import random
from datasets import load_dataset
from tqdm import tqdm

TARGET_ROWS = 60000
SEED = 42
OUTPUT_FILE = "kelm_stem_60k.csv"

# Keywords STEM para filtrar (en inglés, que es el idioma de KELM)
STEM_KEYWORDS = [
    # Informática y Programación
    "computer", "software", "programming", "algorithm", "database",
    "operating system", "linux", "windows", "python", "java", "javascript",
    "compiler", "processor", "cpu", "gpu", "memory", "server", "cloud",
    "artificial intelligence", "machine learning", "neural network",
    "deep learning", "data structure", "binary", "boolean", "encryption",
    "cybersecurity", "firewall", "malware", "virus", "hacking",

    # Telecomunicaciones y Redes
    "network", "protocol", "internet", "wifi", "bluetooth", "ethernet",
    "router", "switch", "bandwidth", "frequency", "signal", "antenna",
    "satellite", "telecommunication", "5g", "4g", "lte", "wireless",
    "fiber optic", "tcp", "udp", "http", "dns", "ip address",
    "modulation", "spectrum", "radio", "broadcast", "transmission",
    "cellular", "mobile network", "base station",

    # Electrónica
    "circuit", "transistor", "diode", "resistor", "capacitor",
    "semiconductor", "microchip", "integrated circuit", "voltage",
    "current", "amplifier", "sensor", "arduino", "raspberry pi",
    "led", "lcd", "display", "battery", "power supply",

    # Ingeniería general
    "engineer", "engineering", "technology", "technical", "patent",
    "invention", "inventor", "laboratory", "experiment", "prototype",
    "manufacture", "industrial", "automation", "robot", "robotics",

    # Matemáticas y Física
    "mathematics", "equation", "theorem", "calculus", "algebra",
    "geometry", "statistics", "probability", "matrix", "vector",
    "physics", "quantum", "electromagnetic", "wavelength", "photon",
    "laser", "optics", "thermodynamic", "energy", "force",

    # Ciencias aplicadas
    "biomedical", "nanotechnology", "aerospace", "mechanical",
    "electrical", "chemical engineering", "material science",

    # Estándares y organizaciones técnicas
    "ieee", "iso", "ietf", "w3c", "ansi", "etsi",
    "standard", "specification", "protocol",

    # Empresas y productos tecnológicos
    "google", "microsoft", "apple", "intel", "amd", "nvidia",
    "cisco", "ibm", "samsung", "qualcomm", "huawei", "ericsson",
    "tesla", "spacex", "nasa", "esa",
]

if __name__ == "__main__":
    random.seed(SEED)

    print("Cargando dataset KELM (puede tardar unos minutos)...")
    dataset = load_dataset("kelm", split="train", trust_remote_code=True)
    print(f"KELM total: {len(dataset)} filas")

    # Crear set de keywords en minúsculas para búsqueda rápida
    keywords_lower = [kw.lower() for kw in STEM_KEYWORDS]

    # Filtrar por keywords STEM
    print("\nFiltrando por temática STEM...")
    stem_rows = []

    for item in tqdm(dataset, desc="Filtrando"):
        triple = item.get('triple', '')
        sentence = item.get('sentence', '')

        if not triple or not sentence:
            continue

        # Verificar si contiene alguna keyword STEM
        combined = (triple + " " + sentence).lower()

        for kw in keywords_lower:
            if kw in combined:
                # Limpiar
                triple_clean = triple.replace('\n', ' ').replace('\r', '').strip()
                sentence_clean = sentence.replace('\n', ' ').replace('\r', '').strip()

                # Filtrar filas demasiado cortas o largas
                if 20 < len(triple_clean) < 500 and 15 < len(sentence_clean) < 500:
                    stem_rows.append({
                        'Structured_Data': triple_clean,
                        'Human_Reference': sentence_clean,
                    })
                break  # No contar la misma fila dos veces

    print(f"\nFilas STEM encontradas: {len(stem_rows)}")

    if len(stem_rows) < TARGET_ROWS:
        print(f"AVISO: Solo hay {len(stem_rows)} filas STEM, menos de las {TARGET_ROWS} objetivo.")
        print("Se usarán todas las encontradas.")
        selected = stem_rows
    else:
        # Seleccionar aleatoriamente TARGET_ROWS filas
        selected = random.sample(stem_rows, TARGET_ROWS)
        print(f"Seleccionadas aleatoriamente: {len(selected)}")

    # Eliminar duplicados por Structured_Data
    seen = set()
    unique = []
    for row in selected:
        key = row['Structured_Data'][:200]
        if key not in seen:
            seen.add(key)
            unique.append(row)

    print(f"Tras eliminar duplicados: {len(unique)}")

    # Guardar CSV
    df = pd.DataFrame(unique)
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')

    print(f"\nGuardado: {OUTPUT_FILE}")
    print(f"Filas: {len(unique)}")

    # Mostrar ejemplos
    print("\n" + "="*60)
    print("EJEMPLOS:")
    print("="*60)
    samples = random.sample(unique, min(10, len(unique)))
    for i, row in enumerate(samples, 1):
        print(f"\n--- Ejemplo {i} ---")
        print(f"DATOS: {row['Structured_Data'][:200]}")
        print(f"TEXTO: {row['Human_Reference'][:200]}")

    # Estadísticas
    print("\n" + "="*60)
    print("ESTADÍSTICAS:")
    print("="*60)
    avg_triple_len = sum(len(r['Structured_Data']) for r in unique) / len(unique)
    avg_text_len = sum(len(r['Human_Reference']) for r in unique) / len(unique)
    print(f"Longitud media de tripletas: {avg_triple_len:.0f} caracteres")
    print(f"Longitud media de texto: {avg_text_len:.0f} caracteres")