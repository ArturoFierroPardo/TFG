# pip install pandas numpy
"""
Intervalos de Confianza Bootstrap (95%).
Calcula el intervalo de confianza del percentil bootstrap para la media
de cada métrica × modelo × dataset.

- n_bootstrap = 10000 iteraciones
- Método: percentil (2.5% - 97.5%)
- Seed fijo para reproducibilidad

Grupo 1: 9 modelos en WebNLG, ToTTo, KELM.
Grupo 2: 7 modelos en Teleco.

Genera:
  - Reporte_Bootstrap.txt
  - Reporte_Bootstrap.csv

USO: python bootstrap_ci.py --input-dir resultados
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

ARCHIVO_MAPA_3COL = [
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

N_BOOTSTRAP = 10000
SEED = 42
ALPHA = 0.05  # 95% CI → percentiles 2.5% y 97.5%


def cargar_todos(input_dir):
    df_total = pd.DataFrame()
    archivos = glob.glob(os.path.join(input_dir, "metricas_por_fila_*.csv"))
    for archivo in archivos:
        nombre_base = os.path.basename(archivo).replace("metricas_por_fila_", "").replace(".csv", "")
        modelo, dataset = None, None
        for patron, m, d in ARCHIVO_MAPA_3COL:
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

    for c in ['ROUGE_L', 'METEOR', 'BERTScore', 'BLEU']:
        if c in df_total.columns:
            n_antes = len(df_total)
            df_total = df_total[(df_total[c].isna()) | ((df_total[c] >= 0) & (df_total[c] <= 1))]
            n_filtrado = n_antes - len(df_total)
            if n_filtrado > 0:
                print(f"  [FILTRO] {c}: eliminadas {n_filtrado} filas fuera de [0,1]")

    return df_total


def bootstrap_ci(datos, n_boot=N_BOOTSTRAP, seed=SEED):
    """Calcula intervalo de confianza bootstrap del percentil para la media."""
    rng = np.random.RandomState(seed)
    n = len(datos)
    medias_boot = np.empty(n_boot)

    for i in range(n_boot):
        muestra = datos[rng.randint(0, n, size=n)]
        medias_boot[i] = np.mean(muestra)

    ci_low = np.percentile(medias_boot, 100 * ALPHA / 2)      # 2.5%
    ci_high = np.percentile(medias_boot, 100 * (1 - ALPHA / 2))  # 97.5%
    media = np.mean(datos)
    ancho = ci_high - ci_low

    return media, ci_low, ci_high, ancho


def ejecutar_bootstrap(df, grupo, f, filas_csv):
    f.write(f"\n{'='*70}\n")
    f.write(f"  {grupo['nombre'].upper()}\n")
    f.write(f"{'='*70}\n")

    for ds in grupo["datasets"]:
        f.write(f"\n{'─'*50}\n")
        f.write(f"  Dataset: {ds}\n")
        f.write(f"{'─'*50}\n")

        for col, (nombre, direccion) in METRICAS.items():
            if col not in df.columns:
                continue

            f.write(f"\n  ■ {nombre} ({direccion})\n")
            f.write(f"    {'Modelo':<20s} {'N':>6s} {'Media':>8s} {'CI 2.5%':>10s} {'CI 97.5%':>10s} {'Ancho CI':>10s}\n")
            f.write(f"    {'─'*20} {'─'*6} {'─'*8} {'─'*10} {'─'*10} {'─'*10}\n")

            resultados_metrica = []

            for m in grupo["modelos"]:
                df_m = df[(df['Dataset'] == ds) & (df['Modelo'] == m)]
                vals = df_m[col].dropna().values

                if len(vals) < 10:
                    continue

                media, ci_low, ci_high, ancho = bootstrap_ci(vals)

                f.write(f"    {m:<20s} {len(vals):>6d} {media:>8.4f} [{ci_low:>9.4f}, {ci_high:>9.4f}] {ancho:>10.4f}\n")

                resultados_metrica.append({
                    'modelo': m, 'media': media,
                    'ci_low': ci_low, 'ci_high': ci_high
                })

                filas_csv.append({
                    'grupo': grupo['nombre'], 'dataset': ds, 'metrica': nombre,
                    'col': col, 'modelo': m, 'n': len(vals),
                    'media': media, 'ci_low': ci_low, 'ci_high': ci_high,
                    'ancho_ci': ancho,
                })

            # Detectar solapamientos de CI (modelos potencialmente equivalentes)
            if len(resultados_metrica) >= 2:
                solapados = []
                for i in range(len(resultados_metrica)):
                    for j in range(i + 1, len(resultados_metrica)):
                        a = resultados_metrica[i]
                        b = resultados_metrica[j]
                        # Solapamiento: el CI de uno contiene parte del CI del otro
                        if a['ci_low'] <= b['ci_high'] and b['ci_low'] <= a['ci_high']:
                            solapados.append((a['modelo'], b['modelo']))

                if solapados:
                    f.write(f"    Solapamientos CI (modelos potencialmente equivalentes):\n")
                    for m_a, m_b in solapados:
                        f.write(f"      ↔ {m_a} ≈ {m_b}\n")
                else:
                    f.write(f"    Sin solapamientos: todos los modelos son distinguibles.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="resultados")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: No se encuentra '{args.input_dir}'")
        exit(1)

    print("Cargando datos...")
    df = cargar_todos(args.input_dir)
    if df.empty:
        print("Error: No se han cargado datos.")
        exit(1)

    print(f"  {len(df)} filas | {df['Modelo'].nunique()} modelos | {df['Dataset'].nunique()} datasets")
    print(f"  Bootstrap: {N_BOOTSTRAP} iteraciones, IC {int((1-ALPHA)*100)}%, seed={SEED}\n")

    filas_csv = []

    with open("Reporte_Bootstrap.txt", "w", encoding="utf-8") as f:
        f.write("═"*70 + "\n")
        f.write("   INTERVALOS DE CONFIANZA BOOTSTRAP (95%)\n")
        f.write("═"*70 + "\n")
        f.write(f"\nMetodología:\n")
        f.write(f"  • Bootstrap del percentil: {N_BOOTSTRAP} iteraciones\n")
        f.write(f"  • Intervalo: [{ALPHA/2*100:.1f}%, {(1-ALPHA/2)*100:.1f}%]\n")
        f.write(f"  • Estadístico: media\n")
        f.write(f"  • Seed: {SEED}\n")
        f.write(f"  • Solapamiento de CI → modelos potencialmente equivalentes\n\n")

        print("─── Grupo 1: 9 modelos ───")
        ejecutar_bootstrap(df, GRUPO_1, f, filas_csv)

        print("─── Grupo 2: 7 modelos ───")
        ejecutar_bootstrap(df, GRUPO_2, f, filas_csv)

    df_csv = pd.DataFrame(filas_csv)
    df_csv.to_csv("Reporte_Bootstrap.csv", index=False, encoding="utf-8")

    print(f"\n✓ Reporte: Reporte_Bootstrap.txt")
    print(f"✓ CSV:     Reporte_Bootstrap.csv")