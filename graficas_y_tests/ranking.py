# pip install pandas numpy
"""
Rankings de Modelos por Métrica × Dataset.
- Flag --modo: 'media' o 'mediana' para elegir el criterio de ranking.
- Grupo 1: 9 modelos (LLM + SLM + mini-SLM) en WebNLG, ToTTo, KELM.
- Grupo 2: 7 modelos (GAN + mini-SLM base + FT) en Teleco.
- Genera un .txt con los rankings y un .csv con los valores.

USO:
  python rankings.py --input-dir resultados --modo media
  python rankings.py --input-dir resultados --modo mediana
"""
import pandas as pd
import numpy as np
import glob
import os
import argparse

# =====================================================================
METRICAS = {
    'ROUGE_L':      ('ROUGE-L',           'Mayor es mejor'),
    'METEOR':       ('METEOR',            'Mayor es mejor'),
    'BERTScore':    ('BERTScore',         'Mayor es mejor'),
    'BLEU':         ('BLEU',              'Mayor es mejor'),
    'Time_seconds': ('Tiempo (Segundos)', 'Menor es mejor'),
    'CO2_gramos':   ('CO₂ (Gramos)',      'Menor es mejor'),
}

GRUPO_1 = {
    "nombre": "Grupo 1: LLM vs SLM vs Mini-SLM (9 modelos)",
    "datasets": ["WebNLG", "ToTTo", "KELM"],
    "modelos": [
        'DeepSeek', 'Llama 70B', 'Qwen 72B',
        'Gemma 9B', 'Llama 3B', 'Qwen 7B',
        'Gemma 3 1B', 'Llama 1B', 'Qwen3 1.7B',
    ],
}

GRUPO_2 = {
    "nombre": "Grupo 2: GAN vs Mini-SLM Base vs Mini-SLM FT (7 modelos)",
    "datasets": ["Teleco"],
    "modelos": [
        'GAN',
        'Gemma 3 1B', 'Llama 1B', 'Qwen3 1.7B',
        'Gemma 3 1B FT', 'Llama 1B FT', 'Qwen3 1.7B FT',
    ],
}

ARCHIVO_MAPA = [
    ('webNLG_Gemma_3_1B',       'Gemma 3 1B',    'WebNLG'),
    ('webNLG_Llama_1B',         'Llama 1B',      'WebNLG'),
    ('webNLG_Qwen3_1.7B',      'Qwen3 1.7B',    'WebNLG'),
    ('totto_Gemma_3_1B',        'Gemma 3 1B',    'ToTTo'),
    ('totto_Llama_1B',          'Llama 1B',      'ToTTo'),
    ('totto_Qwen3_1.7B',       'Qwen3 1.7B',    'ToTTo'),
    ('kelm_stem_Gemma_3_1B',   'Gemma 3 1B',    'KELM'),
    ('kelm_stem_Llama_1B',     'Llama 1B',      'KELM'),
    ('kelm_stem_Qwen3_1.7B',  'Qwen3 1.7B',    'KELM'),
    ('teleco_Gemma_3_1B',      'Gemma 3 1B',    'Teleco'),
    ('teleco_Llama_1B',        'Llama 1B',      'Teleco'),
    ('teleco_Qwen3_1.7B',     'Qwen3 1.7B',    'Teleco'),
    ('teleco_Qwen_7B',         None,            None),
    ('teleco_Gemma_9B',         None,            None),
    ('teleco_Llama_3B',         None,            None),
    ('GAN_valtest',            'GAN',           'Teleco'),
    ('Gemma_3_1B_FT_valtest',  'Gemma 3 1B FT', 'Teleco'),
    ('Llama_1B_FT_valtest',    'Llama 1B FT',   'Teleco'),
    ('Qwen3_1.7B_FT_valtest', 'Qwen3 1.7B FT', 'Teleco'),
    ('DeepSeek_WebNLG',   'DeepSeek',  'WebNLG'),
    ('DeepSeek_ToTTo',    'DeepSeek',  'ToTTo'),
    ('DeepSeek_KELM',     'DeepSeek',  'KELM'),
    ('Llama_70B_WebNLG',  'Llama 70B', 'WebNLG'),
    ('Llama_70B_ToTTo',   'Llama 70B', 'ToTTo'),
    ('Llama_70B_KELM',    'Llama 70B', 'KELM'),
    ('Qwen_72B_WebNLG',   'Qwen 72B',  'WebNLG'),
    ('Qwen_72B_ToTTo',    'Qwen 72B',  'ToTTo'),
    ('Qwen_72B_KELM',     'Qwen 72B',  'KELM'),
    ('Gemma_9B_WebNLG',   'Gemma 9B',  'WebNLG'),
    ('Gemma_9B_ToTTo',    'Gemma 9B',  'ToTTo'),
    ('Gemma_9B_KELM',     'Gemma 9B',  'KELM'),
    ('Llama_3B_WebNLG',   'Llama 3B',  'WebNLG'),
    ('Llama_3B_ToTTo',    'Llama 3B',  'ToTTo'),
    ('Llama_3B_KELM',     'Llama 3B',  'KELM'),
    ('Qwen_7B_WebNLG',    'Qwen 7B',   'WebNLG'),
    ('Qwen_7B_ToTTo',     'Qwen 7B',   'ToTTo'),
    ('Qwen_7B_KELM',      'Qwen 7B',   'KELM'),
]


def cargar_todos(input_dir):
    df_total = pd.DataFrame()
    archivos = glob.glob(os.path.join(input_dir, "metricas_por_fila_*.csv"))
    for archivo in archivos:
        nombre_base = os.path.basename(archivo).replace("metricas_por_fila_", "").replace(".csv", "")
        modelo, dataset = None, None
        for patron, m, d in ARCHIVO_MAPA:
            if nombre_base == patron:
                modelo, dataset = m, d
                break
        if modelo is None:
            continue
        try:
            df_temp = pd.read_csv(archivo)
            df_temp['Modelo'] = modelo
            df_temp['Dataset'] = dataset
            df_total = pd.concat([df_total, df_temp], ignore_index=True)
        except Exception as e:
            print(f"  [ERROR] {archivo}: {e}")

    for c in METRICAS.keys():
        if c in df_total.columns:
            df_total[c] = pd.to_numeric(df_total[c], errors='coerce')
    return df_total


def generar_rankings(df, grupo, modo, f, filas_csv):
    """Genera rankings por métrica × dataset para un grupo."""

    f.write(f"\n{'='*70}\n")
    f.write(f"  {grupo['nombre'].upper()}\n")
    f.write(f"  Criterio de ranking: {modo.upper()}\n")
    f.write(f"{'='*70}\n")

    for ds in grupo["datasets"]:
        f.write(f"\n{'─'*50}\n")
        f.write(f"  Dataset: {ds}\n")
        f.write(f"{'─'*50}\n")

        for col, (nombre, direccion) in METRICAS.items():
            if col not in df.columns:
                continue

            df_filtrado = df[
                (df['Dataset'] == ds) &
                (df['Modelo'].isin(grupo['modelos'])) &
                (df[col].notna())
            ]

            modelos_presentes = [m for m in grupo['modelos'] if m in df_filtrado['Modelo'].unique()]
            if len(modelos_presentes) < 2:
                continue

            # Calcular estadísticos por modelo
            filas_modelo = []
            for m in modelos_presentes:
                vals = df_filtrado[df_filtrado['Modelo'] == m][col].dropna().values
                if len(vals) == 0:
                    continue
                filas_modelo.append({
                    'modelo': m,
                    'media': np.mean(vals),
                    'mediana': np.median(vals),
                    'std': np.std(vals, ddof=1),
                    'min': np.min(vals),
                    'max': np.max(vals),
                    'n': len(vals),
                })

            if len(filas_modelo) < 2:
                continue

            # Ordenar según modo y dirección
            reverse = "Mayor" in direccion
            filas_modelo.sort(key=lambda x: x[modo], reverse=reverse)

            # Escribir ranking
            f.write(f"\n  ■ {nombre} ({direccion})\n")
            f.write(f"    {'Pos':<5s} {'Modelo':<20s} {'Media':>10s} {'Mediana':>10s} {'σ':>10s} {'Min':>10s} {'Max':>10s} {'N':>6s}\n")
            f.write(f"    {'─'*5} {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*6}\n")

            for i, r in enumerate(filas_modelo):
                f.write(f"    {i+1:<5d} {r['modelo']:<20s} {r['media']:>10.4f} {r['mediana']:>10.4f} {r['std']:>10.4f} {r['min']:>10.4f} {r['max']:>10.4f} {r['n']:>6d}\n")

                filas_csv.append({
                    'grupo': grupo['nombre'],
                    'dataset': ds,
                    'metrica': nombre,
                    'col': col,
                    'posicion': i + 1,
                    'modelo': r['modelo'],
                    'media': r['media'],
                    'mediana': r['mediana'],
                    'std': r['std'],
                    'min': r['min'],
                    'max': r['max'],
                    'n': r['n'],
                    'criterio': modo,
                })

            # ¿Cambia el ranking si usamos el otro criterio?
            otro_modo = 'mediana' if modo == 'media' else 'media'
            filas_otro = sorted(filas_modelo, key=lambda x: x[otro_modo], reverse=reverse)
            ranking_actual = [r['modelo'] for r in filas_modelo]
            ranking_otro = [r['modelo'] for r in filas_otro]

            if ranking_actual != ranking_otro:
                f.write(f"    ⚠ OJO: El ranking por {otro_modo} sería diferente:\n")
                for i, m in enumerate(ranking_otro):
                    val = filas_otro[i][otro_modo]
                    f.write(f"       {i+1}º {m} ({otro_modo}={val:.4f})\n")
            else:
                f.write(f"    ✓ Mismo ranking por {otro_modo}.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="resultados")
    parser.add_argument("--modo", choices=["media", "mediana"], default="media",
                        help="Criterio para ordenar: 'media' o 'mediana'")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: No se encuentra '{args.input_dir}'")
        exit(1)

    print(f"Cargando datos (modo: {args.modo})...")
    df = cargar_todos(args.input_dir)
    if df.empty:
        print("Error: No se han cargado datos.")
        exit(1)

    print(f"  {len(df)} filas | {df['Modelo'].nunique()} modelos | {df['Dataset'].nunique()} datasets\n")

    filas_csv = []
    nombre_txt = f"Rankings_{args.modo}.txt"
    nombre_csv = f"Rankings_{args.modo}.csv"

    with open(nombre_txt, "w", encoding="utf-8") as f:
        f.write("═"*70 + "\n")
        f.write(f"   RANKINGS DE MODELOS (criterio: {args.modo.upper()})\n")
        f.write("═"*70 + "\n")
        f.write(f"\nCriterio principal: {args.modo}\n")
        f.write(f"Se indica si el ranking cambiaría con el otro criterio.\n")

        print("─── Grupo 1: 9 modelos en bases genéricas ───")
        generar_rankings(df, GRUPO_1, args.modo, f, filas_csv)

        print("─── Grupo 2: 7 modelos en Teleco ───")
        generar_rankings(df, GRUPO_2, args.modo, f, filas_csv)

    df_csv = pd.DataFrame(filas_csv)
    df_csv.to_csv(nombre_csv, index=False, encoding="utf-8")

    print(f"\n✓ Reporte: {nombre_txt}")
    print(f"✓ CSV:     {nombre_csv}")