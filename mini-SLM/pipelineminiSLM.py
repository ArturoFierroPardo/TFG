# pip install datasets openai tqdm pandas evaluate rouge_score bert_score sacrebleu nltk tiktoken "datasets<3.0.0"
"""
Pipeline Mini-SLMs sobre 3 BBDD públicas (totto, webnlg, kelm).

Procesa los 3 Mini-SLMs (1-2B parámetros) sobre las 3 BBDD del benchmark:
  - Llama 3.2 1B Instruct (OpenRouter)
  - Gemma 3 1B IT          (Together)
  - Qwen 3 1.7B            (Together)

Saca métricas por fila para boxplots:
  - ROUGE-L, METEOR, BLEU, BERTScore
  - Tiempo por fila
  - Tokens input/output
  - Coste USD por fila
  - CO2 (gramos) por fila

ROBUSTO: Multihilo (20 hilos), resume anti-duplicados, modo --fix de errores.

Uso:
    python pipeline_minislm_3bbdd.py --dataset totto  --model llama
    python pipeline_minislm_3bbdd.py --dataset webnlg --model gemma
    python pipeline_minislm_3bbdd.py --dataset kelm   --model qwen
    python pipeline_minislm_3bbdd.py --dataset totto  --model llama --limit 100
    python pipeline_minislm_3bbdd.py --dataset totto  --model llama --fix

    # Lanzar TODAS las combinaciones (9 = 3 modelos x 3 bbdd):
    python pipeline_minislm_3bbdd.py --all
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
import re
import argparse
import concurrent.futures
import threading
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
import evaluate
from io import StringIO

# Tokens con tiktoken (fallback si no está instalado)
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def n_tokens(text):
        try:    return len(_enc.encode(text or ""))
        except: return max(1, len(text or "") // 4)
except ImportError:
    def n_tokens(text):
        return max(1, len(text or "") // 4)


# =========================================================
# CANDADO DE SEGURIDAD PARA MULTIHILO
# =========================================================
csv_lock = threading.Lock()

# =========================================================
# API KEYS  (PEGA AQUÍ TUS CLAVES)
# =========================================================
TOGETHER_API_KEY  = "REDACTED"
OPENROUTER_API_KEY = "REDACTED"

# =========================================================
# CONFIGURACIÓN DE DATASETS Y MODELOS
# =========================================================
DATASETS_CONFIG = {
    "totto": {
        "hf_name":    "totto",
        "hf_version": None,
        "split":      "train",
        "max_tokens": 150,
    },
    "webnlg": {
        "hf_name":    "web_nlg",
        "hf_version": "release_v2.1",
        "split":      "train",
        "max_tokens": 100,
    },
    "kelm": {
        "hf_name":    "local_csv",
        "hf_version": None,
        "split":      "train",
        "max_tokens": 100,
        "csv_file":   "kelm_stem_60k.csv",
    },
}

MODELS_CONFIG = {
    "llama":  "meta-llama/llama-3.2-1b-instruct",
    "gemma":  "arturofierrop2_c6c6/google/gemma-3-4b-it-8b4dc37a",
    "qwen":   "arturofierrop2_c6c6/Qwen/Qwen3-1.7B-416ad4dd",
}

PROVIDER_CONFIG = {
    "llama": {"base_url": "https://openrouter.ai/api/v1", "api_key_var": "OPENROUTER"},
    "qwen":  {"base_url": "https://api.together.xyz/v1",  "api_key_var": "TOGETHER"},
    "gemma": {"base_url": "https://api.together.xyz/v1",  "api_key_var": "TOGETHER"},
}

def get_api_key(model_key):
    var = PROVIDER_CONFIG[model_key]["api_key_var"]
    if var == "OPENROUTER": return OPENROUTER_API_KEY
    if var == "TOGETHER":   return TOGETHER_API_KEY
    return OPENROUTER_API_KEY

def get_base_url(model_key):
    return PROVIDER_CONFIG[model_key]["base_url"]

# Mapeo clave -> nombre legible para los CSVs
NOMBRES_LEGIBLES = {
    "llama":  "Llama_1B",
    "gemma":  "Gemma_3_1B",
    "qwen":   "Qwen3_1.7B",
}

PROMPTS = {
    "totto":  "You are a Data-to-Text generation API. Convert the provided table metadata and highlighted cell data into a single, highly fluent English sentence.\nRULE: Output ONLY the final English sentence without any extra comments, conversational filler, or greetings.",
    "webnlg": "You are a Data-to-Text generation API. Convert the provided table metadata and highlighted cell data into a single, highly fluent English sentence. Output ONLY the final English sentence without any extra comments, conversational filler, or greetings.",
    "kelm":   "You are a Data-to-Text generation API. Convert the provided knowledge graph triples into a single, highly fluent English sentence.\nRULE: Output ONLY the final English sentence without any extra comments, conversational filler, or greetings.",
}

# Tokens del prompt sistema (aprox.)
TOKENS_PROMPT_SISTEMA = {
    "totto":  n_tokens(PROMPTS["totto"]),
    "webnlg": n_tokens(PROMPTS["webnlg"]),
    "kelm":   n_tokens(PROMPTS["kelm"]),
}

# Precios USD por token (input/output)
PRECIOS = {
    "meta-llama/llama-3.2-1b-instruct":                       {"input": 0.00000001, "output": 0.00000002},
    "arturofierrop2_c6c6/google/gemma-3-1b-it-98eee583":      {"input": 0.00000003, "output": 0.00000003},
    "arturofierrop2_c6c6/Qwen/Qwen3-1.7B-416ad4dd":          {"input": 0.00000005, "output": 0.00000010},
}
PRECIO_DEFAULT = {"input": 0.0000002, "output": 0.0000002}

# kWh/token estimado para cálculo de CO2 (modelos pequeños)
KWH_POR_TOKEN = {
    "meta-llama/llama-3.2-1b-instruct":                       0.00000003,
    "arturofierrop2_c6c6/google/gemma-3-1b-it-98eee583":      0.00000003,
    "arturofierrop2_c6c6/Qwen/Qwen3-1.7B-416ad4dd":          0.00000004,
}
KWH_DEFAULT     = 0.0000002
CO2_POR_KWH     = 400.0   # gramos CO2 por kWh (media UE)

MAX_HILOS    = 20
BATCH_BERT   = 1000

# =========================================================
# EXTRACTORES POR DATASET
# =========================================================
def extraer_totto(item):
    titulo = item.get("table_page_title", "")
    seccion = item.get("table_section_title", "")
    tabla = item["table"]
    celdas_resaltadas = item["highlighted_cells"]

    valores_resaltados = []
    for coordenada in celdas_resaltadas:
        fila = coordenada[0]
        columna = coordenada[1]
        try:
            valor = tabla[fila][columna]["value"]
            valores_resaltados.append(str(valor))
        except Exception:
            continue

    datos = f"Title: {titulo} | Section: {seccion} | Highlighted Data: {' | '.join(valores_resaltados)}"

    try:
        if isinstance(item["sentence_annotations"], dict):
            ref = item["sentence_annotations"]["final_sentence"][0]
        else:
            ref = item["sentence_annotations"][0]["final_sentence"]
    except Exception:
        ref = "ERROR"

    return datos, ref


def extraer_webnlg(item):
    triples = " | ".join(item['modified_triple_sets']['mtriple_set'][0])
    reference = item['lex']['text'][0]
    return triples, reference


def extraer_kelm(item):
    datos = item['Structured_Data'].replace('\n', ' ').replace('\r', '')
    reference = item['Human_Reference'].replace('\n', ' ').replace('\r', '')
    return datos, reference


EXTRACTORES = {
    "totto":  extraer_totto,
    "webnlg": extraer_webnlg,
    "kelm":   extraer_kelm,
}


# =========================================================
# LIMPIEZA FINAL DEL CSV
# =========================================================
def limpiar_csv_final(archivo, total_filas):
    print(f"\n[CLEAN] Limpiando {archivo}...")
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
    print(f"  [OK] {len(datos)} datos ({errores_restantes} errores) + {len(metricas)} métricas + 1 cabecera = {total_lineas} líneas")


# =========================================================
# PIPELINE PRINCIPAL
# =========================================================
def run_pipeline(dataset_key, model_key, limit=None, evaluar=False):
    api_key = get_api_key(model_key)
    base_url = get_base_url(model_key)

    if not api_key or "PEGA_AQUI" in api_key:
        print(f"[X] API key no configurada para {model_key} (proveedor: {PROVIDER_CONFIG[model_key]['api_key_var']})")
        print(f"   Edita la cabecera de este archivo con tus keys reales.")
        return

    ds_config = DATASETS_CONFIG[dataset_key]
    model_id = MODELS_CONFIG[model_key]
    nombre_modelo = NOMBRES_LEGIBLES[model_key]
    extraer = EXTRACTORES[dataset_key]
    prompt = PROMPTS[dataset_key]

    # Nombres bonitos para cada dataset
    dataset_names = {
        "totto":  "totto",
        "webnlg": "webNLG",
        "kelm":   "kelm_stem",
    }
    ds_label = dataset_names[dataset_key]

    # Nombre del CSV: results_dataset_modelo o results_dataset_N_modelo
    if limit:
        output_name = f"results_{ds_label}_{limit}_{nombre_modelo}.csv"
    else:
        output_name = f"results_{ds_label}_{nombre_modelo}.csv"

    print(f"\n{'='*60}")
    print(f"PIPELINE: {nombre_modelo} sobre {ds_label.upper()}")
    print(f"Proveedor: {base_url}")
    print(f"Modelo:    {model_id}")
    print(f"Output:    {output_name}")
    print(f"{'='*60}")

    if evaluar:
        print("\nCargando métricas en memoria...")
        metrica_rouge  = evaluate.load('rouge')
        metrica_meteor = evaluate.load('meteor')
        metrica_bleu   = evaluate.load('bleu')
        metrica_bert   = evaluate.load('bertscore')
    else:
        print("\n[INFO] Modo solo inferencia (sin --evaluar)")

    print(f"\nCargando dataset: {ds_config['hf_name']}...")
    if ds_config['hf_name'] == 'local_csv':
        # Cargar desde CSV local (KELM)
        csv_path = ds_config['csv_file']
        posibles = [
            csv_path,
            os.path.join('base kelm', csv_path),
            os.path.join('..', 'base kelm', csv_path),
            os.path.join('..', '..', 'base kelm', csv_path),
            os.path.join(os.path.expanduser('~'), 'Desktop', 'TFG', 'base kelm', csv_path),
        ]
        csv_real = None
        for p in posibles:
            if os.path.exists(p):
                csv_real = p
                break
        if not csv_real:
            print(f"[X] No se encontró {csv_path}")
            return
        print(f"  Leyendo: {csv_real}")
        df_kelm = pd.read_csv(csv_real)
        dataset = df_kelm.to_dict(orient='records')
    else:
        if ds_config['hf_version']:
            dataset = load_dataset(ds_config['hf_name'], ds_config['hf_version'],
                                   split=ds_config['split'])
        else:
            dataset = load_dataset(ds_config['hf_name'], split=ds_config['split'])

    total_filas = limit if limit else len(dataset)
    print(f"  Total filas a procesar: {total_filas}")

    client = OpenAI(base_url=base_url, api_key=api_key)

    # =========================================================
    # RESUME: cargar CSV previo
    # =========================================================
    cabecera = ["Structured_Data", "Human_Reference", "LLM_Generated", "Time_per_row_seconds"]
    dict_procesados = {}

    if os.path.exists(output_name):
        print(f"\n[READ] CSV previo encontrado, retomando: {output_name}")
        with open(output_name, 'r', encoding='utf-8') as f:
            contenido = f.read()
        reader = csv.reader(StringIO(contenido))
        try: next(reader)
        except: pass

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

        # Reescribir CSV solo con filas validas
        with open(output_name, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(cabecera)
            writer.writerows(datos_validos_para_guardar)

        total_recuperadas = sum(len(lista) for lista in dict_procesados.values())
        print(f"  Filas válidas recuperadas: {total_recuperadas}")
        if errores_encontrados > 0:
            print(f"  Errores previos eliminados: {errores_encontrados} (se reprocesarán)")
    else:
        pd.DataFrame(columns=cabecera).to_csv(output_name, index=False)

    preds_validas  = []
    refs_validas   = []
    datos_validos  = []   # para tokens correctos
    tiempos        = []
    items_a_procesar = []
    filas_procesadas_total = 0

    print("\nEvaluando filas guardadas y preparando envíos...")
    for item in dataset:
        datos_estructurados, referencia = extraer(item)

        if referencia == "ERROR":
            continue

        if filas_procesadas_total >= total_filas:
            break

        # Truncar ANTES de calcular la firma para que coincida con el CSV
        if len(datos_estructurados) > 5000:
            datos_estructurados = datos_estructurados[:5000] + "..."

        firma_actual = (str(datos_estructurados).strip(), str(referencia).strip())

        if firma_actual in dict_procesados and len(dict_procesados[firma_actual]) > 0:
            saved_text, time_raw = dict_procesados[firma_actual].pop(0)
            try:
                saved_time = float(time_raw)
            except ValueError:
                saved_time = 1.5
            if not saved_text.startswith("Error:"):
                preds_validas.append(saved_text)
                refs_validas.append(referencia)
                datos_validos.append(datos_estructurados)
            tiempos.append(saved_time)
        else:
            items_a_procesar.append((datos_estructurados, referencia))

        filas_procesadas_total += 1

    # =========================================================
    # FUNCIÓN QUE EJECUTARÁ CADA HILO
    # =========================================================
    # Detectar Qwen 3 para desactivar thinking
    def es_qwen3(mid):
        return "qwen3" in mid.lower() or "Qwen3" in mid

    def limpiar_think(text):
        """Elimina bloques <think>...</think> cerrados Y abiertos (sin cierre)."""
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        text = re.sub(r'<think>.*', '', text, flags=re.DOTALL).strip()
        return text

    def procesar_fila_api(datos_tarea):
        datos_estructurados, referencia = datos_tarea
        datos_estructurados = datos_estructurados.replace('\n', ' ').replace('\r', '')
        user_content = f"Data: {datos_estructurados}"

        # Qwen 3: /no_think en el mensaje para desactivar thinking a nivel de modelo
        if es_qwen3(model_id):
            user_content += "\n/no_think"

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user",   "content": user_content}
        ]

        # Qwen 3: también extra_body como doble seguridad
        kwargs_extra = {}
        if es_qwen3(model_id):
            kwargs_extra["extra_body"] = {"enable_thinking": False}

        generated_text = ""
        start_row_time = time.time()
        backoff_sec = 2.0

        for attempt in range(5):
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=ds_config['max_tokens'],
                    temperature=0.1,
                    **kwargs_extra,
                )
                
                content = response.choices[0].message.content
                generated_text = content.strip() if content else "Error: modelo devolvio respuesta vacia"
                # Limpiar <think> por si acaso (doble seguridad)
                if "<think>" in generated_text:
                    generated_text = limpiar_think(generated_text)
                generated_text = generated_text.replace('\n', ' ').replace('\r', '')
                if not generated_text:
                    generated_text = "Error: modelo solo genero thinking sin respuesta"
                
                # ¡Sin waits ni pausas artificiales aquí!
                break
            except Exception as e:
                err_str = str(e).lower()
                generated_text = f"Error: {str(e)[:200]}"
                
                if "rate" in err_str or "429" in err_str:
                    time.sleep(backoff_sec)
                    backoff_sec = min(backoff_sec * 2, 30)
                else:
                    time.sleep(backoff_sec)
                    backoff_sec = min(backoff_sec * 2, 60)

        end_row_time = time.time()
        row_time = round(end_row_time - start_row_time, 4)

        # GUARDADO PROTEGIDO POR EL CANDADO
        with csv_lock:
            nueva_fila = pd.DataFrame([{
                "Structured_Data":      datos_estructurados,
                "Human_Reference":      referencia,
                "LLM_Generated":        generated_text,
                "Time_per_row_seconds": row_time
            }])
            nueva_fila.to_csv(output_name, mode='a', header=False, index=False)

        return generated_text, referencia, datos_estructurados, row_time
    # =========================================================
    # EJECUCIÓN CONCURRENTE (MULTIHILO)
    # =========================================================
    if len(items_a_procesar) > 0:
        print(f"\n[GO] Lanzando {MAX_HILOS} hilos para {len(items_a_procesar)} filas restantes...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_HILOS) as executor:
            resultados = list(tqdm(
                executor.map(procesar_fila_api, items_a_procesar),
                total=len(items_a_procesar),
                desc=f"Generando ({nombre_modelo})"
            ))

        for gen_text, ref, datos, r_time in resultados:
            if not gen_text.startswith("Error:"):
                preds_validas.append(gen_text)
                refs_validas.append(ref)
                datos_validos.append(datos)
            tiempos.append(r_time)
    else:
        print("\n[OK] Todas las filas ya estaban procesadas.")

    total_time = sum(tiempos)
    avg_time   = total_time / len(tiempos) if tiempos else 0

    # =========================================================
    # MÉTRICAS (solo si --evaluar)
    # =========================================================
    if evaluar and len(preds_validas) > 0:
        print(f"\nEvaluando las {len(preds_validas)} respuestas válidas...")

        # Métricas globales
        try:
            rouge_score = round(metrica_rouge.compute(predictions=preds_validas, references=refs_validas)['rougeL'], 4)
        except Exception as e:
            print(f"[!] ROUGE global falló: {e}"); rouge_score = 0.0
        try:
            meteor_score = round(metrica_meteor.compute(predictions=preds_validas, references=refs_validas)['meteor'], 4)
        except Exception as e:
            print(f"[!] METEOR global falló: {e}"); meteor_score = 0.0
        try:
            bleu_score = round(metrica_bleu.compute(predictions=preds_validas, references=[[r] for r in refs_validas])['bleu'], 4)
        except Exception as e:
            print(f"[!] BLEU global falló: {e}"); bleu_score = 0.0

        # BERTScore global en batches
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
                    lang="en",
                    model_type="distilbert-base-uncased"
                )
                all_f1.extend(res['f1'])
            except Exception as e:
                print(f"[!] BERTScore batch {i} falló: {e}")
                all_f1.extend([0.0] * len(batch_preds))

        bert_score = round(sum(all_f1) / len(all_f1), 4) if all_f1 else 0.0

        # Métricas POR FILA para boxplots
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
            except: bleus_fila.append(0.0)

        bertscores_fila = [round(f, 4) for f in all_f1]

        # Coste y CO2 por fila
        print(f"Calculando coste y CO2 por fila...")
        precios = PRECIOS.get(model_id, PRECIO_DEFAULT)
        kwh     = KWH_POR_TOKEN.get(model_id, KWH_DEFAULT)
        tokens_prompt = TOKENS_PROMPT_SISTEMA[dataset_key]

        tok_in_list, tok_out_list, costes, co2s = [], [], [], []
        for i in range(len(preds_validas)):
            ti = tokens_prompt + n_tokens(datos_validos[i])
            to = n_tokens(preds_validas[i])
            tok_in_list.append(ti)
            tok_out_list.append(to)
            costes.append(round(ti * precios['input'] + to * precios['output'], 8))
            co2s.append(round((ti + to) * kwh * CO2_POR_KWH, 6))

        # Tiempos paralelos a las predicciones válidas
        n_validas = len(preds_validas)
        tiempos_validos = tiempos[:n_validas] if len(tiempos) >= n_validas else tiempos + [avg_time] * (n_validas - len(tiempos))

        # Guardar métricas por fila
        metricas_fila_name = output_name.replace("results_", "metricas_por_fila_")
        pd.DataFrame({
            'ROUGE_L':       rouges_fila,
            'METEOR':        meteors_fila,
            'BLEU':          bleus_fila,
            'BERTScore':     bertscores_fila,
            'Time_seconds':  tiempos_validos[:n_validas],
            'Tokens_Input':  tok_in_list,
            'Tokens_Output': tok_out_list,
            'Coste_USD':     costes,
            'CO2_gramos':    co2s,
        }).to_csv(metricas_fila_name, index=False)
        print(f"[OK] Métricas por fila guardadas: {metricas_fila_name}")

        # Totales agregados
        coste_total = sum(costes)
        co2_total   = sum(co2s)

        # Guardar resumen al final del CSV
        metricas_finales = pd.DataFrame([
            {"Structured_Data": "--- TOTAL TIME ---",        "Human_Reference": "", "LLM_Generated": "", "Time_per_row_seconds": round(total_time, 4)},
            {"Structured_Data": "--- AVERAGE TIME ---",      "Human_Reference": "", "LLM_Generated": "", "Time_per_row_seconds": round(avg_time, 4)},
            {"Structured_Data": "--- METRICA: ROUGE-L ---",  "Human_Reference": "", "LLM_Generated": rouge_score,  "Time_per_row_seconds": ""},
            {"Structured_Data": "--- METRICA: METEOR ---",   "Human_Reference": "", "LLM_Generated": meteor_score, "Time_per_row_seconds": ""},
            {"Structured_Data": "--- METRICA: BLEU ---",     "Human_Reference": "", "LLM_Generated": bleu_score,   "Time_per_row_seconds": ""},
            {"Structured_Data": "--- METRICA: BERTSCORE ---","Human_Reference": "", "LLM_Generated": bert_score,   "Time_per_row_seconds": ""},
            {"Structured_Data": "--- COSTE TOTAL USD ---",   "Human_Reference": "", "LLM_Generated": round(coste_total, 6),  "Time_per_row_seconds": ""},
            {"Structured_Data": "--- CO2 TOTAL GRAMOS ---",  "Human_Reference": "", "LLM_Generated": round(co2_total, 4),    "Time_per_row_seconds": ""},
            {"Structured_Data": "--- RESPUESTAS VALIDAS ---","Human_Reference": "", "LLM_Generated": f"{len(preds_validas)}/{len(tiempos)}", "Time_per_row_seconds": ""}
        ])
        metricas_finales.to_csv(output_name, mode='a', header=False, index=False)

        print(f"\n{'='*60}")
        print(f"COMPLETADO {nombre_modelo} en {dataset_key.upper()}")
        print(f"ROUGE-L: {rouge_score} | METEOR: {meteor_score} | BLEU: {bleu_score} | BERTScore: {bert_score}")
        print(f"Coste:   {coste_total:.6f} USD | CO2: {co2_total:.4f} g")
        print(f"Respuestas válidas: {len(preds_validas)}/{filas_procesadas_total}")
        print(f"{'='*60}")

    else:
        # Solo inferencia: resumen sin métricas
        errores = len(tiempos) - len(preds_validas)
        print(f"\n{'='*60}")
        print(f"INFERENCIA COMPLETADA: {nombre_modelo} en {dataset_key.upper()}")
        print(f"Respuestas válidas: {len(preds_validas)}/{filas_procesadas_total}")
        print(f"Errores: {errores}")
        print(f"Tiempo total: {total_time:.2f}s | Media: {avg_time:.4f}s/fila")
        if not evaluar:
            print(f"[INFO] Usa --evaluar para calcular métricas")
        print(f"{'='*60}")

    # Limpieza final
    limpiar_csv_final(output_name, total_filas)


# =========================================================
# MODO FIX: arregla errores
# =========================================================
def fix_errors(dataset_key, model_key, limit=None):
    """Reprocesa SOLO las filas con Error: del CSV existente."""
    api_key = get_api_key(model_key)
    base_url = get_base_url(model_key)

    if not api_key or "PEGA_AQUI" in api_key:
        print(f"[X] API key no configurada.")
        return

    ds_config = DATASETS_CONFIG[dataset_key]
    model_id = MODELS_CONFIG[model_key]
    nombre_modelo = NOMBRES_LEGIBLES[model_key]
    prompt = PROMPTS[dataset_key]

    dataset_names = {"totto": "totto", "webnlg": "webNLG", "kelm": "kelm_stem"}
    ds_label = dataset_names[dataset_key]

    if limit:
        output_name = f"results_{ds_label}_{limit}_{nombre_modelo}.csv"
    else:
        output_name = f"results_{ds_label}_{nombre_modelo}.csv"

    if not os.path.exists(output_name):
        print(f"[X] No existe {output_name}. Lanza primero el modo normal.")
        return

    print(f"\n[FIX] Modo arreglar errores: {output_name}")
    df = pd.read_csv(output_name)
    # Filtrar las filas con error
    df_err = df[df['LLM_Generated'].astype(str).str.startswith('Error:')].copy()
    print(f"  Filas con error a reprocesar: {len(df_err)}")
    if len(df_err) == 0:
        print(f"  [OK] No hay errores.")
        return

    client = OpenAI(base_url=base_url, api_key=api_key)

    # Qwen 3: desactivar thinking
    def es_qwen3(mid):
        return "qwen3" in mid.lower() or "Qwen3" in mid

    kwargs_extra = {}
    if es_qwen3(model_id):
        kwargs_extra["extra_body"] = {"enable_thinking": False}

    def procesar_fila(idx_y_fila):
        idx, row = idx_y_fila
        datos = str(row['Structured_Data'])
        ref   = str(row['Human_Reference'])
        user_content = f"Data: {datos}"
        if es_qwen3(model_id):
            user_content += "\n/no_think"
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user",   "content": user_content}
        ]
        start = time.time()
        gen = ""
        for attempt in range(5):
            try:
                response = client.chat.completions.create(
                    model=model_id, messages=messages,
                    max_tokens=ds_config['max_tokens'], temperature=0.1,
                    **kwargs_extra,
                )
                content = response.choices[0].message.content
                gen = content.strip() if content else "Error: vacio"
                if "<think>" in gen:
                    gen = re.sub(r'<think>.*?</think>', '', gen, flags=re.DOTALL).strip()
                    gen = re.sub(r'<think>.*', '', gen, flags=re.DOTALL).strip()
                gen = gen.replace('\n', ' ').replace('\r', '')
                if not gen:
                    gen = "Error: modelo solo genero thinking sin respuesta"
                break
            except Exception as e:
                gen = f"Error: {str(e)[:200]}"
                time.sleep(2 * (attempt + 1))
        elapsed = round(time.time() - start, 4)
        return idx, gen, elapsed

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_HILOS) as ex:
        for idx, gen, elapsed in tqdm(ex.map(procesar_fila, df_err.iterrows()),
                                       total=len(df_err), desc="Fixing"):
            df.at[idx, 'LLM_Generated'] = gen
            df.at[idx, 'Time_per_row_seconds'] = elapsed

    df.to_csv(output_name, index=False)
    print(f"[OK] CSV actualizado. Relanza el pipeline normal para recalcular métricas.")


# =========================================================
# CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["totto", "webnlg", "kelm"],
                        help="Dataset a procesar")
    parser.add_argument("--model",   choices=["llama", "gemma", "qwen"],
                        help="Modelo a usar")
    parser.add_argument("--limit",   type=int, default=None,
                        help="Limitar a N filas (test rápido)")
    parser.add_argument("--evaluar", action="store_true",
                        help="Calcular métricas tras inferencia (ROUGE, METEOR, BLEU, BERTScore)")
    parser.add_argument("--fix",     action="store_true",
                        help="Solo reprocesar errores")
    parser.add_argument("--all",     action="store_true",
                        help="Procesar TODAS las combinaciones (3 modelos x 3 bbdd = 9)")
    args = parser.parse_args()

    if args.all:
        # Procesar las 9 combinaciones en orden
        combinaciones = [
            (ds, m) for ds in ["totto", "webnlg", "kelm"]
                    for m in ["llama", "gemma", "qwen"]
        ]
        print(f"[GO] Lanzando {len(combinaciones)} combinaciones (3 bbdd x 3 modelos):")
        for ds, m in combinaciones:
            print(f"  - {NOMBRES_LEGIBLES[m]} en {ds}")
        print()
        for ds, m in combinaciones:
            try:
                run_pipeline(ds, m, args.limit, evaluar=args.evaluar)
            except KeyboardInterrupt:
                print("\n[!] Interrumpido")
                raise
            except Exception as e:
                print(f"[X] Fallo en {m}/{ds}: {e}. Continuando...")
                import traceback
                traceback.print_exc()
        print("\n[OK] Todas las combinaciones procesadas")
        return

    if not args.dataset or not args.model:
        parser.error("Especifica --dataset y --model, o usa --all")

    if args.fix:
        fix_errors(args.dataset, args.model, args.limit)
    else:
        run_pipeline(args.dataset, args.model, args.limit, evaluar=args.evaluar)


if __name__ == "__main__":
    main()
