"""
Inferencia y evaluacion de Mini-SLMs (Llama 3.2 1B, Gemma 3 1B, Qwen3 1.7B) sobre
el dataset propio de Telecomunicaciones (24.680 pares pregunta-respuesta en
espanol). En Qwen3 se desactiva el modo de razonamiento (enable_thinking).

Con --split splits_por_subtema.json la evaluacion se restringe a val+test.
Calcula metricas globales y por fila (ROUGE-L, METEOR, BLEU, BERTScore con
xlm-roberta-base), tiempo, tokens, coste y CO2. Procesa en paralelo, reanuda
ejecuciones interrumpidas y dispone de un modo --fix para reprocesar errores.

Las claves se leen de las variables de entorno OPENROUTER_API_KEY y TOGETHER_API_KEY.

Requisitos:
    pip install openai tqdm pandas evaluate rouge_score bert_score sacrebleu nltk tiktoken

Uso:
    python pipelineminiSLMbbdd.py --model llama --evaluar
    python pipelineminiSLMbbdd.py --model qwen --evaluar --split resultados_arturo/splits/splits_por_subtema.json
    python pipelineminiSLMbbdd.py --model gemma --fix
"""

import sys
# Forzar UTF-8 en stdout/stderr (necesario en Windows con cp1252)
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


import pandas as pd
import time
import os
import csv
import argparse
import concurrent.futures
import threading
from openai import OpenAI
from tqdm import tqdm
import evaluate
from io import StringIO

csv_lock = threading.Lock()

TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

DATASET_PATH = "dataset_teleco.csv"
OUTDIR = "resultados_arturo/minislm_base"

MODELS_CONFIG = {
    "llama":  "meta-llama/llama-3.2-1b-instruct",
    "gemma":  "arturofierrop2_c6c6/google/gemma-3-1b-it-98eee583",
    "qwen":   "arturofierrop2_c6c6/Qwen/Qwen3-1.7B-416ad4dd",
}

# Qué proveedor usar para cada modelo
PROVIDER_CONFIG = {
    "llama": {"base_url": "https://openrouter.ai/api/v1",  "api_key_var": "OPENROUTER"},
    "qwen":  {"base_url": "https://api.together.xyz/v1",   "api_key_var": "TOGETHER"},
    "gemma": {"base_url": "https://api.together.xyz/v1",   "api_key_var": "TOGETHER"},
}

def get_api_key(model_key):
    var = PROVIDER_CONFIG[model_key]["api_key_var"]
    if var == "OPENROUTER": return OPENROUTER_API_KEY
    if var == "TOGETHER":   return TOGETHER_API_KEY
    return OPENROUTER_API_KEY

def get_base_url(model_key):
    return PROVIDER_CONFIG[model_key]["base_url"]

# Mapeo clave -> nombre legible para los CSVs (compatible con master.py)
NOMBRES_LEGIBLES = {
    "llama":  "Llama_1B",
    "gemma":  "Gemma_3_1B",
    "qwen":   "Qwen3_1.7B",
}

# Prompt en español adaptado a Telecomunicaciones
PROMPT_SISTEMA = ("Eres un experto en Telecomunicaciones. Responde de forma "
                  "técnica y concisa a la pregunta del usuario.")

MAX_TOKENS    = 256
TEMPERATURE   = 0.0   # greedy
MAX_HILOS     = 20
BATCH_BERT    = 1000

# Precios USD por token
PRECIOS = {
    "meta-llama/llama-3.2-1b-instruct":                          {"input": 0.00000001, "output": 0.00000002},
    "arturofierrop2_c6c6/google/gemma-3-1b-it-98eee583":         {"input": 0.00000003, "output": 0.00000003},
    "arturofierrop2_c6c6/Qwen/Qwen3-1.7B-416ad4dd":             {"input": 0.00000005, "output": 0.00000010},
}
PRECIO_DEFAULT = {"input": 0.0000002, "output": 0.0000002}

# kWh/token estimado para cálculo de CO2 (modelos pequeños)
KWH_POR_TOKEN = {
    "meta-llama/llama-3.2-1b-instruct":                          0.00000003,
    "arturofierrop2_c6c6/google/gemma-3-1b-it-98eee583":         0.00000003,
    "arturofierrop2_c6c6/Qwen/Qwen3-1.7B-416ad4dd":             0.00000004,
}
CO2_POR_KWH = 475
TOKENS_PROMPT_SISTEMA = 45


def limpiar_csv_final(archivo, total_filas):
    print(f"\nLimpiando {archivo}...")
    with open(archivo, 'r', encoding='utf-8') as f:
        contenido = f.read()
    reader = csv.reader(StringIO(contenido))
    header = next(reader)

    filas_ok = []
    filas_error = []
    metricas = []

    for row in reader:
        if len(row) < 4: continue
        if str(row[0]).startswith('---'):
            metricas.append(row)
            continue
        row = [campo.replace('\n', ' ').replace('\r', '') for campo in row]
        if len(row) > 4:
            row = [row[0], row[1], ",".join(row[2:-1]), row[-1]]
        if len(row) == 4:
            firma = (row[0].strip(), row[1].strip())
            if row[2].startswith("Error:"):
                if not any(f == firma for f, _ in [(r[0].strip(), r[1].strip()) for r in filas_ok if len(r) >= 2]):
                    filas_error.append(row)
            else:
                filas_error = [e for e in filas_error if (e[0].strip(), e[1].strip()) != firma]
                filas_ok.append(row)

    datos = filas_ok[:total_filas]
    if len(datos) < total_filas:
        datos.extend(filas_error[:total_filas - len(datos)])

    metricas = metricas[-7:]

    with open(archivo, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(datos)
        writer.writerows(metricas)

    errores_restantes = sum(1 for d in datos if d[2].startswith("Error:"))
    total_lineas = 1 + len(datos) + len(metricas)
    print(f"   {len(datos)} datos ({errores_restantes} errores) + {len(metricas)} métricas + 1 cabecera = {total_lineas} líneas totales")


def cargar_dataset_teleco(csv_path, limit=None, split_json=None):
    """Lee dataset_teleco.csv y devuelve lista de pares (pregunta, respuesta).
    Si split_json apunta a splits_por_subtema.json, filtra solo val+test."""
    if not os.path.exists(csv_path):
        print(f"ERROR: No se encontró {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    # Filtrar filas válidas (sin NaN ni vacíos)
    df = df.dropna(subset=['pregunta', 'respuesta'])
    df['pregunta']  = df['pregunta'].astype(str).str.strip()
    df['respuesta'] = df['respuesta'].astype(str).str.strip()
    df = df[df['pregunta'].str.len() > 0]
    df = df[df['respuesta'].str.len() > 0]

    # Aplicar split val+test si se indica
    if split_json:
        import json
        with open(split_json, 'r', encoding='utf-8') as f:
            splits = json.load(f)
        valtest_ids = set(splits['val_ids'] + splits['test_ids'])
        df = df[df.index.isin(valtest_ids)]
        print(f"Filtrado val+test: {len(df)} filas (de {splits['n_filas_total']})")

    pares = list(zip(df['pregunta'].tolist(), df['respuesta'].tolist()))

    if limit:
        pares = pares[:limit]

    print(f"Cargado {csv_path}: {len(pares)} pares")
    return pares


def run_pipeline(model_key, limit=None, csv_path=DATASET_PATH, evaluar=False, split_json=None):
    api_key = get_api_key(model_key)
    base_url = get_base_url(model_key)
    if not api_key:
        print(f"API key no configurada para {model_key}. Edita la línea correspondiente al inicio del archivo.")
        return

    model_id = MODELS_CONFIG[model_key]
    nombre_modelo = NOMBRES_LEGIBLES[model_key]   # nombre legible para archivos

    os.makedirs(OUTDIR, exist_ok=True)

    sufijo = "_valtest" if split_json else ""
    if limit:
        output_name = os.path.join(OUTDIR, f"results_teleco_{limit}_{nombre_modelo}{sufijo}.csv")
    else:
        output_name = os.path.join(OUTDIR, f"results_teleco_{nombre_modelo}{sufijo}.csv")

    if evaluar:
        print("Cargando métricas en memoria...")
        metrica_rouge  = evaluate.load('rouge')
        metrica_meteor = evaluate.load('meteor')
        metrica_bleu   = evaluate.load('bleu')
        metrica_bert   = evaluate.load('bertscore')
    else:
        print("Modo solo inferencia (sin --evaluar)")

    pares = cargar_dataset_teleco(csv_path, limit, split_json=split_json)
    if pares is None:
        return

    total_filas = len(pares)
    client = OpenAI(base_url=base_url, api_key=api_key)

    print("\n" + "=" * 50)
    print(f"DATASET: TELECO (CEU) | MODELO: {nombre_modelo.upper()}")
    if limit: print(f"LÍMITE: {limit} filas")
    print(f"Guardado en: {output_name}")
    print(f"Hilos concurrentes: {MAX_HILOS}")
    print("=" * 50)

    dict_procesados = {}

    if os.path.exists(output_name):
        lineas_limpias = []
        with open(output_name, 'r', encoding='utf-8') as f:
            for linea in f:
                linea = linea.rstrip('\r\n')
                while linea.endswith(';'):
                    linea = linea[:-1]
                lineas_limpias.append(linea + '\n')

        reader = csv.reader(StringIO("".join(lineas_limpias)))
        cabecera = next(reader)

        datos_validos_para_guardar = []
        errores_encontrados = 0

        for row in reader:
            if len(row) == 1 and ',' in row[0]:
                sub = list(csv.reader(StringIO(row[0])))
                if sub:
                    row = sub[0]
            if len(row) > 0 and str(row[0]).startswith('---'): continue
            if len(row) >= 4:
                row[-1] = row[-1].replace('"', '').replace(';', '').strip()
            if len(row) > 4:
                row = [row[0], row[1], ",".join(row[2:-1]), row[-1]]

            if len(row) == 4:
                firma = (str(row[0]).strip(), str(row[1]).strip())
                texto_generado = str(row[2])
                tiempo = str(row[3])

                if texto_generado.startswith("Error:"):
                    errores_encontrados += 1
                    continue

                if firma not in dict_procesados:
                    dict_procesados[firma] = []
                dict_procesados[firma].append((texto_generado, tiempo))

                datos_validos_para_guardar.append(row)

        with open(output_name, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(cabecera)
            writer.writerows(datos_validos_para_guardar)

        total_recuperadas = sum(len(lista) for lista in dict_procesados.values())
        print(f"Encontradas {total_recuperadas} filas válidas ya procesadas. Retomando...")
        if errores_encontrados > 0:
            print(f"Eliminados {errores_encontrados} errores previos del CSV. Se reprocesarán automáticamente.")
    else:
        pd.DataFrame(columns=["Structured_Data", "Human_Reference", "LLM_Generated", "Time_per_row_seconds"]).to_csv(output_name, index=False)

    preds_validas = []
    refs_validas = []
    pregs_validas = []   # preguntas en paralelo (para cálculo de tokens correcto)
    tiempos = []
    items_a_procesar = []
    filas_procesadas_total = 0

    print("\nEvaluando filas guardadas y preparando envíos...")
    for pregunta, respuesta in pares:
        if filas_procesadas_total >= total_filas:
            break

        # Limpiar saltos de línea ANTES de calcular firma (para coincidir con CSV)
        pregunta_norm = str(pregunta).replace('\n', ' ').replace('\r', ' ').strip()
        respuesta_norm = str(respuesta).replace('\n', ' ').replace('\r', ' ').strip()
        firma_actual = (pregunta_norm, respuesta_norm)

        if firma_actual in dict_procesados and len(dict_procesados[firma_actual]) > 0:
            saved_text, time_raw = dict_procesados[firma_actual].pop(0)
            try:
                saved_time = float(time_raw)
            except ValueError:
                saved_time = 1.5
            if not saved_text.startswith("Error:"):
                preds_validas.append(saved_text)
                refs_validas.append(respuesta_norm)
                pregs_validas.append(pregunta_norm)
            tiempos.append(saved_time)
        else:
            items_a_procesar.append((pregunta_norm, respuesta_norm))

        filas_procesadas_total += 1

    def es_modelo_gemma(mid):
        return "gemma" in str(mid).lower()

    def es_modelo_qwen3(mid):
        return "qwen3" in str(mid).lower()

    def procesar_fila_api(datos_tarea):
        pregunta_limpia, respuesta = datos_tarea  # ya viene normalizada

        # Gemma no acepta role 'system' -> combinar
        if es_modelo_gemma(model_id):
            messages = [{"role": "user", "content": f"{PROMPT_SISTEMA}\n\n{pregunta_limpia}"}]
        elif es_modelo_qwen3(model_id):
            messages = [
                {"role": "system", "content": PROMPT_SISTEMA},
                {"role": "user",   "content": f"{pregunta_limpia}\n/no_think"},
            ]
        else:
            messages = [
                {"role": "system", "content": PROMPT_SISTEMA},
                {"role": "user",   "content": pregunta_limpia},
            ]

        # Qwen 3 tiene "thinking mode" - desactivar para respuestas directas
        kwargs_extra = {}
        if es_modelo_qwen3(model_id):
            kwargs_extra["extra_body"] = {
                "enable_thinking": False
            }

        generated_text = ""
        start_row_time = time.time()
        backoff_sec = 2.0

        for attempt in range(5):  # Aumentado de 3 a 5
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    **kwargs_extra,
                )
                content = response.choices[0].message.content
                generated_text = content.strip() if content else "Error: modelo devolvió respuesta vacía"
                # Limpiar bloques <think>...</think> por si acaso
                if "<think>" in generated_text:
                    import re
                    generated_text = re.sub(r"<think>.*?</think>", "", generated_text, flags=re.DOTALL).strip()
                    generated_text = re.sub(r"<think>.*", "", generated_text, flags=re.DOTALL).strip()
                generated_text = generated_text.replace('\n', ' ').replace('\r', '')
                # Si la respuesta no es vacía, salir
                if generated_text and not generated_text.startswith("Error:"):
                    break
                # Si fue vacío, esperar antes de reintentar
                time.sleep(backoff_sec)
                backoff_sec = min(backoff_sec * 2, 30)
            except Exception as e:
                error_str = str(e)
                # Detectar rate limit (429) y esperar más
                if "429" in error_str or "rate" in error_str.lower():
                    espera = min(backoff_sec * 2, 60)
                else:
                    espera = backoff_sec
                time.sleep(espera)
                backoff_sec = min(backoff_sec * 2, 60)
                generated_text = f"Error: {error_str[:200]}"

        end_row_time = time.time()
        row_time = round(end_row_time - start_row_time, 4)

        with csv_lock:
            nueva_fila = pd.DataFrame([{
                "Structured_Data": pregunta_limpia,
                "Human_Reference": respuesta,
                "LLM_Generated":   generated_text,
                "Time_per_row_seconds": row_time
            }])
            nueva_fila.to_csv(output_name, mode='a', header=False, index=False)

        return generated_text, respuesta, row_time, pregunta_limpia

    if len(items_a_procesar) > 0:
        print(f"\nLanzando {MAX_HILOS} hilos para las {len(items_a_procesar)} filas restantes...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_HILOS) as executor:
            resultados = list(tqdm(
                executor.map(procesar_fila_api, items_a_procesar),
                total=len(items_a_procesar),
                desc=f"Generando ({nombre_modelo})"
            ))

        for gen_text, ref, r_time, preg in resultados:
            if not gen_text.startswith("Error:"):
                preds_validas.append(gen_text)
                refs_validas.append(ref)
                pregs_validas.append(preg)
            tiempos.append(r_time)
    else:
        print("\nTodas las filas ya estaban procesadas.")

    total_time = sum(tiempos)
    avg_time   = sum(tiempos) / len(tiempos) if tiempos else 0

    # MÉTRICAS (solo si --evaluar)
    if evaluar and len(preds_validas) > 0:
        print(f"\nEvaluando las {len(preds_validas)} respuestas válidas...")

        # Globales
        try:
            res_rouge  = metrica_rouge.compute(predictions=preds_validas, references=refs_validas)
            rouge_score  = round(res_rouge['rougeL'], 4)
        except Exception as e:
            print(f"ROUGE global falló: {e}")
            rouge_score = 0.0
        try:
            res_meteor = metrica_meteor.compute(predictions=preds_validas, references=refs_validas)
            meteor_score = round(res_meteor['meteor'], 4)
        except Exception as e:
            print(f"METEOR global falló: {e}")
            meteor_score = 0.0
        try:
            bleu_score = round(metrica_bleu.compute(
                predictions=preds_validas,
                references=[[r] for r in refs_validas]
            )['bleu'], 4)
        except Exception as e:
            print(f"BLEU global falló: {e}")
            bleu_score = 0.0

        # BERTScore en español
        print(f"Calculando BERTScore en batches de {BATCH_BERT}...")
        all_f1 = []
        total_batches = (len(preds_validas) + BATCH_BERT - 1) // BATCH_BERT
        for i in tqdm(range(0, len(preds_validas), BATCH_BERT), desc="BERTScore", total=total_batches):
            batch_preds = preds_validas[i:i + BATCH_BERT]
            batch_refs  = refs_validas[i:i + BATCH_BERT]
            try:
                res = metrica_bert.compute(
                    predictions=batch_preds,
                    references=batch_refs,
                    lang="es",
                    model_type="xlm-roberta-base",
                )
                all_f1.extend(res['f1'])
            except Exception as e:
                print(f"BERTScore batch {i} falló: {e}")
                all_f1.extend([0.0] * len(batch_preds))

        bert_score = round(sum(all_f1) / len(all_f1), 4) if all_f1 else 0.0

        # Por fila
        print(f"\nCalculando métricas por fila ({len(preds_validas)} filas)...")
        rouges_fila = []
        for i in tqdm(range(len(preds_validas)), desc="ROUGE-L por fila"):
            try:
                r = metrica_rouge.compute(predictions=[preds_validas[i]], references=[refs_validas[i]])
                rouges_fila.append(round(r['rougeL'], 4))
            except: rouges_fila.append(0.0)

        meteors_fila = []
        for i in tqdm(range(len(preds_validas)), desc="METEOR por fila"):
            try:
                r = metrica_meteor.compute(predictions=[preds_validas[i]], references=[refs_validas[i]])
                meteors_fila.append(round(r['meteor'], 4))
            except: meteors_fila.append(0.0)

        bleus_fila = []
        for i in tqdm(range(len(preds_validas)), desc="BLEU por fila"):
            try:
                r = metrica_bleu.compute(predictions=[preds_validas[i]], references=[[refs_validas[i]]])
                bleus_fila.append(round(r['bleu'], 4))
            except:
                bleus_fila.append(0.0)

        bertscores_fila = [round(f, 4) for f in all_f1]

        # Costes y CO2 por fila
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            def n_tokens(t): return len(enc.encode(str(t)))
        except ImportError:
            def n_tokens(t): return len(str(t)) // 4

        precios = PRECIOS.get(model_id, PRECIO_DEFAULT)
        kwh = KWH_POR_TOKEN.get(model_id, 0.0000002)

        tok_in_list, tok_out_list, costes, co2s = [], [], [], []
        for i in range(len(preds_validas)):
            ti = TOKENS_PROMPT_SISTEMA + n_tokens(pregs_validas[i])
            to = n_tokens(preds_validas[i])
            tok_in_list.append(ti)
            tok_out_list.append(to)
            costes.append(round(ti * precios['input'] + to * precios['output'], 8))
            co2s.append(round((ti + to) * kwh * CO2_POR_KWH, 6))

        # Guardar métricas por fila
        metricas_fila_name = output_name.replace("results_", "metricas_por_fila_")
        pd.DataFrame({
            'ROUGE_L':       rouges_fila,
            'METEOR':        meteors_fila,
            'BLEU':          bleus_fila,
            'BERTScore':     bertscores_fila,
            'Time_seconds':  tiempos[:len(rouges_fila)],
            'Tokens_Input':  tok_in_list,
            'Tokens_Output': tok_out_list,
            'Coste_USD':     costes,
            'CO2_gramos':    co2s,
        }).to_csv(metricas_fila_name, index=False)
        print(f"Métricas por fila guardadas: {metricas_fila_name}")

        metricas_finales = pd.DataFrame([
            {"Structured_Data": "--- TOTAL TIME ---",         "Human_Reference": "", "LLM_Generated": "", "Time_per_row_seconds": round(total_time, 4)},
            {"Structured_Data": "--- AVERAGE TIME ---",       "Human_Reference": "", "LLM_Generated": "", "Time_per_row_seconds": round(avg_time, 4)},
            {"Structured_Data": "--- METRICA: ROUGE-L ---",   "Human_Reference": "", "LLM_Generated": rouge_score,  "Time_per_row_seconds": ""},
            {"Structured_Data": "--- METRICA: METEOR ---",    "Human_Reference": "", "LLM_Generated": meteor_score, "Time_per_row_seconds": ""},
            {"Structured_Data": "--- METRICA: BLEU ---",      "Human_Reference": "", "LLM_Generated": bleu_score,   "Time_per_row_seconds": ""},
            {"Structured_Data": "--- METRICA: BERTSCORE ---", "Human_Reference": "", "LLM_Generated": bert_score,   "Time_per_row_seconds": ""},
            {"Structured_Data": "--- RESPUESTAS VALIDAS ---", "Human_Reference": "", "LLM_Generated": f"{len(preds_validas)}/{len(tiempos)}", "Time_per_row_seconds": ""}
        ])
        metricas_finales.to_csv(output_name, mode='a', header=False, index=False)

        print(f"\n{'=' * 50}")
        print(f"COMPLETADO {nombre_modelo.upper()} en TELECO")
        print(f"ROUGE-L: {rouge_score} | METEOR: {meteor_score} | BLEU: {bleu_score} | BERTScore: {bert_score}")
        print(f"Respuestas válidas: {len(preds_validas)}/{filas_procesadas_total}")
        print(f"{'=' * 50}")

    else:
        # Solo inferencia: resumen sin métricas
        errores = len(tiempos) - len(preds_validas)
        print(f"\n{'=' * 50}")
        print(f"INFERENCIA COMPLETADA: {nombre_modelo.upper()} en TELECO")
        print(f"Respuestas válidas: {len(preds_validas)}/{filas_procesadas_total}")
        print(f"Errores: {errores}")
        print(f"Tiempo total: {total_time:.2f}s | Media: {avg_time:.4f}s/fila")
        if not evaluar:
            print(f"Usa --evaluar para calcular métricas")
        print(f"{'=' * 50}")

    limpiar_csv_final(output_name, total_filas)


# MODO FIX: arregla errores y recalcula métricas
def fix_errors(model_key, limit=None, csv_path=DATASET_PATH):
    api_key = get_api_key(model_key)
    base_url = get_base_url(model_key)
    if not api_key:
        print(f"API key no configurada para {model_key}.")
        return

    model_id = MODELS_CONFIG[model_key]
    nombre_modelo = NOMBRES_LEGIBLES[model_key]

    if limit:
        output_name = os.path.join(OUTDIR, f"results_teleco_{limit}_{nombre_modelo}.csv")
    else:
        output_name = os.path.join(OUTDIR, f"results_teleco_{nombre_modelo}.csv")

    if not os.path.exists(output_name):
        print(f"No se encontró {output_name}")
        return

    print(f"\nLeyendo {output_name}...")
    with open(output_name, 'r', encoding='utf-8') as f:
        contenido = f.read()

    reader = csv.reader(StringIO(contenido))
    header = next(reader)

    filas = []
    for row in reader:
        if len(row) < 4: continue
        if row[0].startswith('---'): continue
        filas.append(row)

    errores = sum(1 for r in filas if r[2].startswith('Error:'))
    print(f"Filas: {len(filas)} | Errores: {errores}")

    def es_gemma(mid): return "gemma" in str(mid).lower()
    def es_qwen3(mid): return "qwen3" in str(mid).lower()

    if errores > 0:
        print(f"\nReintentando {errores} errores (3 intentos progresivos)...")
        client = OpenAI(base_url=base_url, api_key=api_key)
        arreglados = 0

        prompt_reforzado = (PROMPT_SISTEMA +
                            "\nDEBES generar exactamente una respuesta en español. "
                            "No devuelvas respuesta vacía.")

        for i, row in enumerate(filas):
            if not row[2].startswith('Error:'):
                continue

            pregunta = row[0]
            if len(pregunta) > 5000:
                pregunta = pregunta[:5000] + "..."
            pregunta = pregunta.replace('\n', ' ').replace('\r', '')

            print(f"  Fila {i+1}: {pregunta[:60]}...")

            intentos = [
                (0.3, prompt_reforzado, pregunta),
                (0.5, prompt_reforzado, pregunta),
                (0.7, prompt_reforzado, f"Pregunta de Telecomunicaciones (responde técnicamente): {pregunta}"),
            ]

            generated = None
            for attempt, (temp, sys_prompt, user_content) in enumerate(intentos, 1):
                try:
                    print(f"    Intento {attempt} (temp={temp})...", end=" ")
                    start_time = time.time()
                    # Añadir /no_think para Qwen 3
                    uc = f"{user_content}\n/no_think" if es_qwen3(model_id) else user_content
                    if es_gemma(model_id):
                        msgs = [{"role": "user", "content": f"{sys_prompt}\n\n{uc}"}]
                    else:
                        msgs = [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user",   "content": uc},
                        ]
                    kwargs_extra = {}
                    if es_qwen3(model_id):
                        kwargs_extra["extra_body"] = {
                            "enable_thinking": False
                        }
                    response = client.chat.completions.create(
                        model=model_id, messages=msgs,
                        max_tokens=MAX_TOKENS, temperature=temp,
                        **kwargs_extra,
                    )
                    content = response.choices[0].message.content
                    row_time = round(time.time() - start_time, 4)

                    if content and content.strip():
                        generated = content.strip().replace('\n', ' ').replace('\r', '')
                        # Limpiar bloques <think>
                        if "<think>" in generated:
                            import re
                            generated = re.sub(r"<think>.*?</think>", "", generated, flags=re.DOTALL).strip()
                            generated = re.sub(r"<think>.*", "", generated, flags=re.DOTALL).strip()
                        print(f"OK: {generated[:60]}")
                        filas[i] = [row[0], row[1], generated, str(row_time)]
                        arreglados += 1
                        break
                    else:
                        print("vacío")
                        time.sleep(1)
                except Exception as e:
                    print(f"error: {str(e)[:50]}")
                    time.sleep(2)

            if generated is None:
                print(f"    FALLO TOTAL: no se pudo arreglar")

        print(f"Arreglados: {arreglados}/{errores}")
    else:
        print("No hay errores que arreglar.")

    # Recalcular métricas (igual que run_pipeline)
    print("\nRecalculando métricas...")
    metrica_rouge  = evaluate.load('rouge')
    metrica_meteor = evaluate.load('meteor')
    metrica_bleu   = evaluate.load('bleu')
    metrica_bert   = evaluate.load('bertscore')

    preds, refs, pregs, tiempos_lista = [], [], [], []
    for row in filas:
        if not row[2].startswith('Error:'):
            preds.append(row[2])
            refs.append(row[1])
            pregs.append(row[0])  # row[0] = pregunta (Structured_Data)
        try:
            tiempos_lista.append(float(str(row[3]).replace('"', '').replace(';', '').strip()))
        except:
            tiempos_lista.append(1.5)

    total_time = sum(tiempos_lista)
    avg_time   = total_time / len(tiempos_lista) if tiempos_lista else 0
    print(f"Respuestas válidas: {len(preds)}/{len(filas)}")

    try:
        rouge_score = round(metrica_rouge.compute(predictions=preds, references=refs)['rougeL'], 4)
    except Exception as e:
        print(f"ROUGE global falló: {e}"); rouge_score = 0.0
    try:
        meteor_score = round(metrica_meteor.compute(predictions=preds, references=refs)['meteor'], 4)
    except Exception as e:
        print(f"METEOR global falló: {e}"); meteor_score = 0.0
    try:
        bleu_score = round(metrica_bleu.compute(predictions=preds, references=[[r] for r in refs])['bleu'], 4)
    except Exception as e:
        print(f"BLEU global falló: {e}"); bleu_score = 0.0
    print(f"  ROUGE-L: {rouge_score} | METEOR: {meteor_score} | BLEU: {bleu_score}")

    print(f"Calculando BERTScore en batches de {BATCH_BERT}...")
    all_f1 = []
    total_batches = (len(preds) + BATCH_BERT - 1) // BATCH_BERT
    for i in tqdm(range(0, len(preds), BATCH_BERT), desc="BERTScore", total=total_batches):
        try:
            res = metrica_bert.compute(
                predictions=preds[i:i+BATCH_BERT], references=refs[i:i+BATCH_BERT],
                lang="es", model_type="xlm-roberta-base",
            )
            all_f1.extend(res['f1'])
        except Exception as e:
            print(f"BERTScore batch {i} falló: {e}")
            all_f1.extend([0.0] * len(preds[i:i+BATCH_BERT]))
    bert_score = round(sum(all_f1) / len(all_f1), 4) if all_f1 else 0.0
    print(f"  BERTScore: {bert_score}")

    # Métricas por fila
    print(f"\nCalculando métricas por fila ({len(preds)} filas)...")
    rouges_fila = []
    for i in tqdm(range(len(preds)), desc="ROUGE-L por fila"):
        try:
            r = metrica_rouge.compute(predictions=[preds[i]], references=[refs[i]])
            rouges_fila.append(round(r['rougeL'], 4))
        except: rouges_fila.append(0.0)

    meteors_fila = []
    for i in tqdm(range(len(preds)), desc="METEOR por fila"):
        try:
            r = metrica_meteor.compute(predictions=[preds[i]], references=[refs[i]])
            meteors_fila.append(round(r['meteor'], 4))
        except: meteors_fila.append(0.0)

    bleus_fila = []
    for i in tqdm(range(len(preds)), desc="BLEU por fila"):
        try:
            r = metrica_bleu.compute(predictions=[preds[i]], references=[[refs[i]]])
            bleus_fila.append(round(r['bleu'], 4))
        except:
            bleus_fila.append(0.0)

    bertscores_fila = [round(f, 4) for f in all_f1]

    # Costes
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        def n_tokens(t): return len(enc.encode(str(t)))
    except ImportError:
        def n_tokens(t): return len(str(t)) // 4

    precios = PRECIOS.get(model_id, PRECIO_DEFAULT)
    kwh = KWH_POR_TOKEN.get(model_id, 0.0000002)

    tok_in_list, tok_out_list, costes, co2s = [], [], [], []
    for i in range(len(preds)):
        ti = TOKENS_PROMPT_SISTEMA + n_tokens(pregs[i])
        to = n_tokens(preds[i])
        tok_in_list.append(ti)
        tok_out_list.append(to)
        costes.append(round(ti * precios['input'] + to * precios['output'], 8))
        co2s.append(round((ti + to) * kwh * CO2_POR_KWH, 6))

    metricas_fila_name = output_name.replace("results_", "metricas_por_fila_")
    pd.DataFrame({
        'ROUGE_L': rouges_fila, 'METEOR': meteors_fila, 'BLEU': bleus_fila,
        'BERTScore': bertscores_fila, 'Time_seconds': tiempos_lista[:len(rouges_fila)],
        'Tokens_Input': tok_in_list, 'Tokens_Output': tok_out_list,
        'Coste_USD': costes, 'CO2_gramos': co2s,
    }).to_csv(metricas_fila_name, index=False)
    print(f"Métricas por fila guardadas: {metricas_fila_name}")

    metricas_nuevas = [
        ["--- TOTAL TIME ---",         "", "",                str(round(total_time, 4))],
        ["--- AVERAGE TIME ---",       "", "",                str(round(avg_time, 4))],
        ["--- METRICA: ROUGE-L ---",   "", str(rouge_score),  ""],
        ["--- METRICA: METEOR ---",    "", str(meteor_score), ""],
        ["--- METRICA: BLEU ---",      "", str(bleu_score),   ""],
        ["--- METRICA: BERTSCORE ---", "", str(bert_score),   ""],
        ["--- RESPUESTAS VALIDAS ---", "", f"{len(preds)}/{len(filas)}", ""],
    ]

    with open(output_name, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(filas)
        writer.writerows(metricas_nuevas)

    errores_final = sum(1 for r in filas if r[2].startswith('Error:'))
    print(f"\n{'=' * 50}")
    print(f"COMPLETADO FIX: {nombre_modelo.upper()}")
    print(f"ROUGE-L: {rouge_score} | METEOR: {meteor_score} | BLEU: {bleu_score} | BERTScore: {bert_score}")
    print(f"Filas: {len(filas)} | Errores restantes: {errores_final}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline OpenRouter Mini-SLM (Teleco, multihilo)")
    parser.add_argument("--model", required=True, choices=list(MODELS_CONFIG.keys()),
                        help="Modelo a usar: llama (1B), gemma (3-1B), qwen (3-1.7B)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Número de filas a procesar (por defecto: todas)")
    parser.add_argument("--evaluar", action="store_true",
                        help="Calcular métricas tras inferencia (ROUGE, METEOR, BLEU, BERTScore)")
    parser.add_argument("--fix", action="store_true",
                        help="Solo arreglar errores y recalcular métricas")
    parser.add_argument("--csv", default=DATASET_PATH,
                        help="Ruta al dataset_teleco.csv")
    parser.add_argument("--split", default=None,
                        help="Ruta a splits_por_subtema.json (filtra solo val+test)")
    args = parser.parse_args()

    if args.fix:
        print(f"\nModo FIX: TELECO + {args.model.upper()}")
        fix_errors(args.model, args.limit, args.csv)
    else:
        print(f"\nPipeline OpenRouter Mini-SLM: TELECO + {args.model.upper()}", end="")
        if args.limit:
            print(f" | Límite: {args.limit} filas")
        else:
            print(" | Dataset completo")
        if args.split:
            print(f" | Split: val+test ({args.split})")
        run_pipeline(args.model, args.limit, args.csv, evaluar=args.evaluar, split_json=args.split)