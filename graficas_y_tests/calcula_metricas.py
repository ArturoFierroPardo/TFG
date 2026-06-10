# pip install evaluate rouge_score bert_score sacrebleu nltk tqdm pandas tiktoken
"""
Calcula TODAS las métricas que falten para todos los results_*.csv.
Genera/actualiza metricas_por_fila_*.csv con las columnas:
  ROUGE_L, METEOR, BLEU, BERTScore, Time_seconds,
  Tokens_Input, Tokens_Output, Coste_USD, CO2_gramos

Si el metricas_por_fila ya existe, solo calcula las columnas faltantes.

USO:
  python calcula_metricas.py --input-dir resultados
  python calcula_metricas.py --input-dir resultados --solo results_teleco_Qwen3_1.7B.csv
"""
import csv, os, glob, argparse
import pandas as pd
import numpy as np
from io import StringIO
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════
# MAPEO: nombre base del results → (modelo_legible, dataset, precio_key)
# ══════════════════════════════════════════════════════════════════════════

# LLM / SLM (formato: results_{dataset}_{api-model-name}.csv)
MODELO_API = {
    "deepseek-chat":           ("DeepSeek",  "deepseek"),
    "llama-3.3-70b-instruct":  ("Llama_70B", "llama-70b"),
    "llama-3_3-70b-instruct":  ("Llama_70B", "llama-70b"),
    "qwen-2.5-72b-instruct":  ("Qwen_72B",  "qwen-72b"),
    "qwen-2_5-72b-instruct":  ("Qwen_72B",  "qwen-72b"),
    "gemma-2-9b-it":           ("Gemma_9B",  "gemma-9b"),
    "llama-3.2-3b-instruct":   ("Llama_3B",  "llama-3b"),
    "llama-3_2-3b-instruct":   ("Llama_3B",  "llama-3b"),
    "qwen-2.5-7b-instruct":   ("Qwen_7B",   "qwen-7b"),
    "qwen-2_5-7b-instruct":   ("Qwen_7B",   "qwen-7b"),
}

DATASET_PARSE = {
    "webNLG": "WebNLG", "webnlg": "WebNLG",
    "totto": "ToTTo",
    "kelm_stem": "KELM", "kelm": "KELM",
    "teleco": "Teleco",
}

# Mini-SLM / FT / GAN (formato directo: results_{patron}.csv)
MAPA_DIRECTO = {
    # Mini-SLM base — datasets genéricos
    'webNLG_Gemma_3_1B':       ('webNLG_Gemma_3_1B',       'mini-gemma'),
    'webNLG_Llama_1B':         ('webNLG_Llama_1B',         'mini-llama'),
    'webNLG_Qwen3_1.7B':      ('webNLG_Qwen3_1.7B',       'mini-qwen'),
    'totto_Gemma_3_1B':        ('totto_Gemma_3_1B',        'mini-gemma'),
    'totto_Llama_1B':          ('totto_Llama_1B',          'mini-llama'),
    'totto_Qwen3_1.7B':       ('totto_Qwen3_1.7B',        'mini-qwen'),
    'kelm_stem_Gemma_3_1B':   ('kelm_stem_Gemma_3_1B',    'mini-gemma'),
    'kelm_stem_Llama_1B':     ('kelm_stem_Llama_1B',      'mini-llama'),
    'kelm_stem_Qwen3_1.7B':  ('kelm_stem_Qwen3_1.7B',    'mini-qwen'),
    # Mini-SLM base — Teleco
    'teleco_Gemma_3_1B':       ('teleco_Gemma_3_1B',       'mini-gemma'),
    'teleco_Llama_1B':         ('teleco_Llama_1B',         'mini-llama'),
    'teleco_Qwen3_1.7B':      ('teleco_Qwen3_1.7B',       'mini-qwen'),
    # FT — valtest
    'Gemma_3_1B_FT_valtest':  ('Gemma_3_1B_FT_valtest',   'mini-gemma'),
    'Llama_1B_FT_valtest':    ('Llama_1B_FT_valtest',     'mini-llama'),
    'Qwen3_1.7B_FT_valtest': ('Qwen3_1.7B_FT_valtest',   'mini-qwen'),
    # FT — KELM
    'kelm_gemma-3-1b_ft':     ('kelm_gemma-3-1b_ft',      'mini-gemma'),
    'kelm_llama-1b_ft':       ('kelm_llama-1b_ft',        'mini-llama'),
    'kelm_qwen-1.7b_ft':     ('kelm_qwen-1.7b_ft',       'mini-qwen'),
    # GAN
    'GAN_valtest':             ('GAN_valtest',              'gan'),
    'kelm_GAN':                ('kelm_GAN',                 'gan'),
}

# ══════════════════════════════════════════════════════════════════════════
# PRECIOS Y CO2
# ══════════════════════════════════════════════════════════════════════════
PRECIOS = {
    "deepseek":  {"input": 0.0000008,  "output": 0.000002},
    "llama-70b": {"input": 0.00000059, "output": 0.00000079},
    "qwen-72b":  {"input": 0.00000059, "output": 0.00000079},
    "gemma-9b":  {"input": 0.0000002,  "output": 0.0000002},
    "llama-3b":  {"input": 0.00000006, "output": 0.00000006},
    "qwen-7b":   {"input": 0.0000002,  "output": 0.0000002},
    "mini-gemma": {"input": 0.00000003, "output": 0.00000003},
    "mini-llama": {"input": 0.00000001, "output": 0.00000002},
    "mini-qwen":  {"input": 0.00000005, "output": 0.00000010},
    "gan":        {"input": 0.0,         "output": 0.0},
}

KWH_POR_TOKEN = {
    "deepseek": 0.0000005, "llama-70b": 0.0000008, "qwen-72b": 0.0000008,
    "gemma-9b": 0.0000002, "llama-3b": 0.00000008, "qwen-7b": 0.0000002,
    "mini-gemma": 0.00000003, "mini-llama": 0.00000003, "mini-qwen": 0.00000004,
    "gan": 0.00000001,
}

CO2_POR_KWH = 475
TOKENS_PROMPT_SISTEMA = 45


# ══════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════════════════
def contar_tokens(texto):
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(str(texto)))
    except ImportError:
        return len(str(texto)) // 4


def detectar_dataset(nombre_metricas):
    """Detecta el dataset a partir del nombre del fichero de métricas."""
    n = nombre_metricas.lower()
    if 'teleco' in n or 'ft_valtest' in n or 'gan_valtest' in n:
        return 'Teleco'
    if 'webnlg' in n:
        return 'WebNLG'
    if 'totto' in n:
        return 'ToTTo'
    if 'kelm' in n:
        return 'KELM'
    return 'unknown'


def bert_config(dataset):
    """BERTScore: inglés para datasets genéricos, español para Teleco."""
    if dataset == 'Teleco':
        return {"lang": "es", "model_type": "xlm-roberta-base"}
    return {"lang": "en", "model_type": "distilbert-base-uncased"}


def parsear_results(filename):
    """
    Dado un results_X.csv, devuelve (nombre_metricas, precio_key) o (None, None).
    nombre_metricas es el sufijo para metricas_por_fila_{nombre_metricas}.csv
    """
    base = os.path.basename(filename).replace('.csv', '').replace('results_', '')

    # 1) Intentar mapa directo (Mini-SLM, FT, GAN)
    if base in MAPA_DIRECTO:
        return MAPA_DIRECTO[base]

    # 2) Intentar parseo LLM/SLM (results_{dataset}_{api-model}.csv)
    for modelo_csv, (modelo_legible, precio_key) in MODELO_API.items():
        if base.endswith(modelo_csv):
            dataset_part = base[:-(len(modelo_csv) + 1)]
            dataset_label = DATASET_PARSE.get(dataset_part, dataset_part)
            nombre = f"{modelo_legible}_{dataset_label}"
            return nombre, precio_key

    return None, None


def leer_results(archivo):
    """Lee results CSV, devuelve lista de filas [input, ref, gen, time]."""
    with open(archivo, 'r', encoding='utf-8') as f:
        contenido = f.read()
    reader = csv.reader(StringIO(contenido))
    next(reader)  # header
    filas = []
    for row in reader:
        if len(row) < 4:
            continue
        if row[0].startswith('---'):
            continue
        if row[2].startswith('Error:'):
            continue
        filas.append(row)
    return filas


# ══════════════════════════════════════════════════════════════════════════
# CÁLCULO DE MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════
def calcular_faltantes(df_met, filas, nombre_metricas, batch_bert=1000):
    """
    Dado un DataFrame de métricas (puede estar vacío) y las filas del results,
    calcula las columnas que falten y devuelve el DataFrame actualizado.
    """
    n = len(filas)
    preds = [row[2] for row in filas]
    refs = [row[1] for row in filas]
    pregs = [row[0] for row in filas]
    dataset = detectar_dataset(nombre_metricas)
    bcfg = bert_config(dataset)

    # Si df_met está vacío, crear esqueleto
    if df_met is None or len(df_met) == 0:
        df_met = pd.DataFrame(index=range(n))

    # Ajustar longitud si no coincide
    if len(df_met) != n:
        print(f"    [WARN] métricas={len(df_met)} vs results={n}, recalculando todo")
        df_met = pd.DataFrame(index=range(n))

    columnas_necesarias = ['ROUGE_L', 'METEOR', 'BLEU', 'BERTScore', 'Time_seconds',
                           'Tokens_Input', 'Tokens_Output', 'Coste_USD', 'CO2_gramos']
    faltantes = [c for c in columnas_necesarias if c not in df_met.columns]

    if not faltantes:
        return df_met, False

    print(f"    Columnas faltantes: {', '.join(faltantes)}")

    # Time_seconds
    if 'Time_seconds' in faltantes:
        tiempos = []
        for row in filas:
            try:
                tiempos.append(float(str(row[3]).replace('"', '').replace(';', '').strip()))
            except:
                tiempos.append(1.5)
        df_met['Time_seconds'] = tiempos

    # ROUGE-L
    if 'ROUGE_L' in faltantes:
        import evaluate
        met = evaluate.load('rouge')
        rouges = []
        for i in tqdm(range(n), desc="    ROUGE-L"):
            try:
                r = met.compute(predictions=[preds[i]], references=[refs[i]])
                rouges.append(round(r['rougeL'], 4))
            except:
                rouges.append(0.0)
        df_met['ROUGE_L'] = rouges

    # METEOR
    if 'METEOR' in faltantes:
        import evaluate
        met = evaluate.load('meteor')
        meteors = []
        for i in tqdm(range(n), desc="    METEOR"):
            try:
                r = met.compute(predictions=[preds[i]], references=[refs[i]])
                meteors.append(round(r['meteor'], 4))
            except:
                meteors.append(0.0)
        df_met['METEOR'] = meteors

    # BLEU
    if 'BLEU' in faltantes:
        import evaluate
        met = evaluate.load('bleu')
        bleus = []
        for i in tqdm(range(n), desc="    BLEU"):
            try:
                r = met.compute(predictions=[preds[i]], references=[[refs[i]]])
                bleus.append(round(r['bleu'], 4))
            except:
                bleus.append(0.0)
        df_met['BLEU'] = bleus

    # BERTScore
    if 'BERTScore' in faltantes:
        import evaluate
        met = evaluate.load('bertscore')
        print(f"    BERTScore ({bcfg['model_type']}, lang={bcfg['lang']})...")
        berts = []
        for i in tqdm(range(0, n, batch_bert), desc="    BERTScore"):
            try:
                r = met.compute(
                    predictions=preds[i:i + batch_bert],
                    references=refs[i:i + batch_bert],
                    **bcfg,
                )
                berts.extend([round(f, 4) for f in r['f1']])
            except Exception as e:
                print(f"    [WARN] BERTScore batch {i} error: {e}")
                berts.extend([0.0] * len(preds[i:i + batch_bert]))
        df_met['BERTScore'] = berts

    # Tokens + Coste + CO2
    if any(c in faltantes for c in ['Tokens_Input', 'Tokens_Output', 'Coste_USD', 'CO2_gramos']):
        tok_in, tok_out, costes, co2s = [], [], [], []
        for i in range(n):
            ti = contar_tokens(pregs[i]) + TOKENS_PROMPT_SISTEMA
            to = contar_tokens(preds[i])
            tok_in.append(ti)
            tok_out.append(to)

        if 'Tokens_Input' in faltantes:
            df_met['Tokens_Input'] = tok_in
        if 'Tokens_Output' in faltantes:
            df_met['Tokens_Output'] = tok_out

        # Coste y CO2 necesitan precio_key (se pasa desde fuera)
        # Lo marcamos como pendiente y se rellena en el caller
        if 'Coste_USD' in faltantes:
            df_met['_tok_in'] = tok_in
            df_met['_tok_out'] = tok_out
        if 'CO2_gramos' in faltantes:
            if '_tok_in' not in df_met.columns:
                df_met['_tok_in'] = tok_in
                df_met['_tok_out'] = tok_out

    return df_met, True


def rellenar_coste_co2(df_met, precio_key):
    """Rellena Coste_USD y CO2_gramos si están pendientes."""
    if '_tok_in' not in df_met.columns:
        return

    precios = PRECIOS.get(precio_key, {"input": 0, "output": 0})
    kwh = KWH_POR_TOKEN.get(precio_key, 0.0000002)

    if 'Coste_USD' not in df_met.columns or df_met['Coste_USD'].isna().all():
        df_met['Coste_USD'] = [
            round(ti * precios['input'] + to * precios['output'], 8)
            for ti, to in zip(df_met['_tok_in'], df_met['_tok_out'])
        ]

    if 'CO2_gramos' not in df_met.columns or df_met['CO2_gramos'].isna().all():
        df_met['CO2_gramos'] = [
            round((ti + to) * kwh * CO2_POR_KWH, 6)
            for ti, to in zip(df_met['_tok_in'], df_met['_tok_out'])
        ]

    df_met.drop(columns=['_tok_in', '_tok_out'], inplace=True, errors='ignore')


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="resultados",
                        help="Carpeta con results_*.csv y metricas_por_fila_*.csv")
    parser.add_argument("--batch-bert", type=int, default=1000)
    parser.add_argument("--solo", default=None, help="Procesar solo este results_*.csv")
    args = parser.parse_args()

    if args.solo:
        csvs = [os.path.join(args.input_dir, args.solo)]
    else:
        csvs = sorted(glob.glob(os.path.join(args.input_dir, "results_*.csv")))

    if not csvs:
        print(f"No se encontraron results_*.csv en {args.input_dir}")
        exit(1)

    print(f"Encontrados {len(csvs)} results CSVs\n")
    actualizados = 0

    for csv_path in csvs:
        filename = os.path.basename(csv_path)
        nombre_metricas, precio_key = parsear_results(csv_path)

        if nombre_metricas is None:
            print(f"⚠ No se pudo parsear: {filename} — saltando")
            continue

        metricas_path = os.path.join(args.input_dir, f"metricas_por_fila_{nombre_metricas}.csv")
        dataset = detectar_dataset(nombre_metricas)

        print(f"{'='*60}")
        print(f"{filename}")
        print(f"  → metricas_por_fila_{nombre_metricas}.csv  [{dataset}]")

        # Leer results
        filas = leer_results(csv_path)
        if not filas:
            print("  Sin filas válidas, saltando")
            continue
        print(f"  {len(filas)} filas válidas")

        # Cargar métricas existentes si las hay
        df_met = None
        if os.path.exists(metricas_path):
            df_met = pd.read_csv(metricas_path)
            existentes = [c for c in ['ROUGE_L', 'METEOR', 'BLEU', 'BERTScore',
                                       'Time_seconds', 'Coste_USD', 'CO2_gramos']
                          if c in df_met.columns]
            print(f"  Ya tiene: {', '.join(existentes)}")

        # Calcular faltantes
        df_met, cambio = calcular_faltantes(df_met, filas, nombre_metricas, args.batch_bert)

        if not cambio:
            print("  ✓ Completo, nada que hacer")
            continue

        # Rellenar coste y CO2
        rellenar_coste_co2(df_met, precio_key)

        # Ordenar columnas
        col_orden = ['ROUGE_L', 'METEOR', 'BLEU', 'BERTScore', 'Time_seconds',
                     'Tokens_Input', 'Tokens_Output', 'Coste_USD', 'CO2_gramos']
        cols_final = [c for c in col_orden if c in df_met.columns]
        extras = [c for c in df_met.columns if c not in cols_final]
        df_met = df_met[cols_final + extras]

        df_met.to_csv(metricas_path, index=False)
        actualizados += 1

        # Resumen
        for col in ['ROUGE_L', 'METEOR', 'BLEU', 'BERTScore']:
            if col in df_met.columns:
                m = df_met[col].mean()
                print(f"    {col:10s}: {m:.4f}")
        print(f"  ✓ Guardado: {metricas_path}")

    print(f"\n{'='*60}")
    print(f"COMPLETADO — {actualizados}/{len(csvs)} ficheros actualizados")