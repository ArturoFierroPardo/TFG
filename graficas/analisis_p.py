# pip install pandas numpy scipy
"""
Tests Estadísticos Globales (Muestras Pareadas).
- Todas las métricas: Friedman (no paramétrico).

- α = 0.05

Grupo 1: 9 modelos (LLM + SLM + mini-SLM) en WebNLG, ToTTo, KELM.
Grupo 2: 7 modelos (GAN + mini-SLM base + mini-SLM FT) en Teleco.

Genera: Reporte_Tests_Globales.txt

USO: python analisis_global.py --input-dir analisis
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
import glob
import os
import argparse

# =====================================================================
# CONFIGURACIÓN
# =====================================================================
METRICAS = {
    'ROUGE_L':      ('ROUGE-L',           'Mayor es mejor',  False),
    'METEOR':       ('METEOR',            'Mayor es mejor',  False),
    'BERTScore':    ('BERTScore',         'Mayor es mejor',  False),  # No paramétrico → Friedman
    'BLEU':         ('BLEU',              'Mayor es mejor',  False),
    'Time_seconds': ('Tiempo (Segundos)', 'Menor es mejor',  False),
    'CO2_gramos':   ('CO₂ (Gramos)',      'Menor es mejor',  False),
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
    "nombre": "Grupo 2: GAN vs Mini-SLM Base vs Mini-SLM Fine-Tuned (7 modelos)",
    "datasets": ["Teleco"],
    "modelos": [
        'GAN',
        'Gemma 3 1B', 'Llama 1B', 'Qwen3 1.7B',
        'Gemma 3 1B FT', 'Llama 1B FT', 'Qwen3 1.7B FT',
    ],
}

ARCHIVO_MAPA_3COL = [
    # Mini-SLMs en bases genéricas (formato invertido)
    ('webNLG_Gemma_3_1B',       'Gemma 3 1B',    'WebNLG'),
    ('webNLG_Llama_1B',         'Llama 1B',      'WebNLG'),
    ('webNLG_Qwen3_1.7B',      'Qwen3 1.7B',    'WebNLG'),
    ('totto_Gemma_3_1B',        'Gemma 3 1B',    'ToTTo'),
    ('totto_Llama_1B',          'Llama 1B',      'ToTTo'),
    ('totto_Qwen3_1.7B',       'Qwen3 1.7B',    'ToTTo'),
    ('kelm_stem_Gemma_3_1B',   'Gemma 3 1B',    'KELM'),
    ('kelm_stem_Llama_1B',     'Llama 1B',      'KELM'),
    ('kelm_stem_Qwen3_1.7B',  'Qwen3 1.7B',    'KELM'),
    # Teleco: mini-SLMs base
    ('teleco_Gemma_3_1B',      'Gemma 3 1B',    'Teleco'),
    ('teleco_Llama_1B',        'Llama 1B',      'Teleco'),
    ('teleco_Qwen3_1.7B',     'Qwen3 1.7B',    'Teleco'),
    # Teleco: excluidos (SLMs medianos en teleco, no entran)
    ('teleco_Qwen_7B',         None,            None),
    ('teleco_Gemma_9B',         None,            None),
    ('teleco_Llama_3B',         None,            None),
    # Teleco: GAN y FT
    ('GAN_valtest',            'GAN',           'Teleco'),
    ('Gemma_3_1B_FT_valtest',  'Gemma 3 1B FT', 'Teleco'),
    ('Llama_1B_FT_valtest',    'Llama 1B FT',   'Teleco'),
    ('Qwen3_1.7B_FT_valtest', 'Qwen3 1.7B FT', 'Teleco'),
    # Originales (6 modelos × 3 datasets)
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

    # Forzar numérico
    for c in METRICAS.keys():
        if c in df_total.columns:
            df_total[c] = pd.to_numeric(df_total[c], errors='coerce')

    # Filtrar valores imposibles en métricas de puntuación (deben ser 0-1)
    metricas_0_1 = ['ROUGE_L', 'METEOR', 'BERTScore', 'BLEU']
    for c in metricas_0_1:
        if c in df_total.columns:
            n_antes = len(df_total)
            df_total = df_total[(df_total[c].isna()) | ((df_total[c] >= 0) & (df_total[c] <= 1))]
            n_filtrado = n_antes - len(df_total)
            if n_filtrado > 0:
                print(f"  [FILTRO] {c}: eliminadas {n_filtrado} filas con valores fuera de [0, 1]")

    return df_total


def formatear_p(p):
    """Formatea p-valor para que siempre muestre un número legible."""
    if p == 0.0:
        return "< 1.0e-300"
    elif p < 1e-300:
        return "< 1.0e-300"
    elif p < 0.001:
        return f"{p:.2e}"
    else:
        return f"{p:.4f}"


def ejecutar_tests(df, grupo, f):
    """Ejecuta Friedman/ANOVA para un grupo y escribe los resultados."""

    f.write(f"\n{'='*70}\n")
    f.write(f"  {grupo['nombre'].upper()}\n")
    f.write(f"{'='*70}\n")

    resultados = []  # Para resumen CSV-friendly al final

    for ds in grupo["datasets"]:
        f.write(f"\n{'─'*50}\n")
        f.write(f"  Dataset: {ds}\n")
        f.write(f"{'─'*50}\n")

        for col, (nombre, direccion, es_parametrico) in METRICAS.items():
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

            # Alinear longitudes (muestras pareadas: misma fila = mismo dato)
            longitud_min = min(len(df_filtrado[df_filtrado['Modelo'] == m]) for m in modelos_presentes)
            if longitud_min < 10:
                f.write(f"\n  ⚠ {nombre}: insuficientes datos pareados ({longitud_min} filas)\n")
                continue

            datos_por_modelo = {}
            for m in modelos_presentes:
                datos_por_modelo[m] = df_filtrado[df_filtrado['Modelo'] == m][col].values[:longitud_min]

            listas = list(datos_por_modelo.values())

            # --- Test ---
            # --- Test (con subsampling si N > MAX_N para p-valores informativos) ---
            MAX_N = 500  # Submuestra para p-valores legibles
            if longitud_min > MAX_N:
                np.random.seed(42)
                idx = np.random.choice(longitud_min, size=MAX_N, replace=False)
                listas_test = [datos[idx] for datos in listas]
                n_test = MAX_N
                nota_subsample = f" (submuestra={MAX_N} de {longitud_min})"
            else:
                listas_test = listas
                n_test = longitud_min
                nota_subsample = ""

            try:
                if es_parametrico:
                    stat, p = stats.f_oneway(*listas_test)
                    test_nombre = "ANOVA (F)"
                else:
                    stat, p = stats.friedmanchisquare(*listas_test)
                    test_nombre = "Friedman (χ²)"
            except Exception as e:
                f.write(f"\n  ⚠ {nombre}: error en test — {e}\n")
                continue

            # --- Ranking ---
            resumen_modelos = []
            for m, datos in datos_por_modelo.items():
                media = np.mean(datos)
                mediana = np.median(datos)
                std = np.std(datos, ddof=1)
                resumen_modelos.append((m, media, mediana, std))

            reverse = "Mayor" in direccion
            resumen_modelos.sort(key=lambda x: x[1], reverse=reverse)

            # --- Escribir ---
            significativo = p < 0.05
            marca = "✓ SIGNIFICATIVO" if significativo else "✗ No significativo"

            f.write(f"\n  ■ {nombre} ({direccion})\n")
            f.write(f"    Test: {test_nombre}{nota_subsample}\n")
            f.write(f"    Estadístico = {stat:.4f}  |  p-valor = {formatear_p(p)}\n")
            f.write(f"    Resultado: {marca} (α = 0.05)\n")
            f.write(f"    N total = {longitud_min}  |  N test = {n_test}\n")
            f.write(f"    Ranking:\n")
            for i, (m, media, mediana, std) in enumerate(resumen_modelos):
                f.write(f"      {i+1}º  {m:<20s}  media={media:.4f}  mediana={mediana:.4f}  σ={std:.4f}\n")

            if significativo:
                f.write(f"    → Se requiere test post-hoc para identificar qué pares difieren.\n")
            else:
                f.write(f"    → Empate técnico global. No se requiere post-hoc.\n")

            resultados.append({
                'grupo': grupo['nombre'],
                'dataset': ds,
                'metrica': nombre,
                'col': col,
                'test': test_nombre,
                'estadistico': stat,
                'p_valor': p,
                'significativo': significativo,
                'n_pareado': longitud_min,
                'n_modelos': len(modelos_presentes),
                'mejor_modelo': resumen_modelos[0][0],
                'mejor_media': resumen_modelos[0][1],
            })

    return resultados


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="analisis")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: No se encuentra '{args.input_dir}'")
        exit(1)

    print("Cargando datos...")
    df = cargar_todos(args.input_dir)
    if df.empty:
        print("Error: No se han cargado datos.")
        exit(1)

    print(f"  {len(df)} filas | {df['Modelo'].nunique()} modelos | {df['Dataset'].nunique()} datasets\n")

    todos_resultados = []

    with open("Reporte_Tests_Globales.txt", "w", encoding="utf-8") as f:
        f.write("═"*70 + "\n")
        f.write("   REPORTE DE SIGNIFICANCIA ESTADÍSTICA GLOBAL\n")
        f.write("   (Muestras Pareadas — Friedman)\n")
        f.write("═"*70 + "\n")
        f.write(f"\nMetodología:\n")
        f.write(f"  • Todas las métricas → Test de Friedman (no paramétrico)\n")
        
        f.write(f"  • Nivel de significancia: α = 0.05\n")
        f.write(f"  • Datos pareados: se trunca al mínimo de filas entre modelos\n")

        print("─── Grupo 1: LLM vs SLM vs Mini-SLM (9 modelos) ───")
        r1 = ejecutar_tests(df, GRUPO_1, f)
        todos_resultados.extend(r1)

        print("─── Grupo 2: GAN vs Mini-SLM vs FT (7 modelos) ───")
        r2 = ejecutar_tests(df, GRUPO_2, f)
        todos_resultados.extend(r2)

        # Resumen final
        f.write(f"\n\n{'═'*70}\n")
        f.write("   RESUMEN RÁPIDO\n")
        f.write(f"{'═'*70}\n\n")

        sig_count = sum(1 for r in todos_resultados if r['significativo'])
        no_sig_count = sum(1 for r in todos_resultados if not r['significativo'])

        f.write(f"  Total de tests realizados: {len(todos_resultados)}\n")
        f.write(f"  Significativos (p < 0.05): {sig_count}\n")
        f.write(f"  No significativos:         {no_sig_count}\n\n")

        f.write(f"  {'Grupo':<55s} {'Dataset':<10s} {'Métrica':<12s} {'Test':<15s} {'p-valor':<12s} {'Sig?'}\n")
        f.write(f"  {'─'*55} {'─'*10} {'─'*12} {'─'*15} {'─'*12} {'─'*5}\n")
        for r in todos_resultados:
            sig_mark = "✓" if r['significativo'] else "✗"
            grupo_corto = "G1: 9 modelos" if "9" in r['grupo'] else "G2: 7 modelos"
            f.write(f"  {grupo_corto:<55s} {r['dataset']:<10s} {r['metrica']:<12s} {r['test']:<15s} {formatear_p(r['p_valor']):<12s} {sig_mark}\n")

    # También guardar CSV para el post-hoc
    df_resultados = pd.DataFrame(todos_resultados)
    df_resultados.to_csv("resultados_tests_globales.csv", index=False, encoding="utf-8")

    print(f"\n✓ Reporte: Reporte_Tests_Globales.txt")
    print(f"✓ CSV:     resultados_tests_globales.csv")
    print(f"\nTests significativos: {sig_count}/{len(todos_resultados)} → requieren post-hoc")