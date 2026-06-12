"""
Inferencia y evaluacion de SLMs (Llama 3.2 3B, Qwen 2.5 7B, Gemma 2 9B) sobre los
benchmarks ToTTo, WebNLG y KELM a traves de OpenRouter.

Calcula metricas globales y por fila (ROUGE-L, METEOR, BLEU, BERTScore), tiempo,
tokens, coste y CO2. Procesa en paralelo, reanuda ejecuciones interrumpidas y
dispone de un modo --fix para reprocesar errores.

La clave de OpenRouter se lee de la variable de entorno OPENROUTER_API_KEY.

Requisitos:
    pip install datasets openai tqdm pandas evaluate rouge_score bert_score sacrebleu nltk tiktoken "datasets<3.0.0"

Uso:
    python pipelineSLM.py --dataset totto --model llama
    python pipelineSLM.py --dataset kelm --model qwen --limit 100
    python pipelineSLM.py --all
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


csv_lock = threading.Lock()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

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
    "llama": "meta-llama/llama-3.2-3b-instruct",
    "qwen":  "qwen/qwen-2.5-7b-instruct",
    "gemma": "google/gemma-2-9b-it",
}

# Mapeo clave -> nombre legible para los CSVs
NOMBRES_LEGIBLES = {
    "llama": "Llama_3B",
    "qwen":  "Qwen_7B",
    "gemma": "Gemma_9B",
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

# Precios USD por token (input/output) según OpenRouter
PRECIOS = {
    "meta-llama/llama-3.2-3b-instruct": {"input": 0.00000003, "output": 0.00000005},
    "qwen/qwen-2.5-7b-instruct":        {"input": 0.00000004, "output": 0.00000010},
    "google/gemma-2-9b-it":             {"input": 0.00000006, "output": 0.00000006},
}
PRECIO_DEFAULT = {"input": 0.0000002, "output": 0.0000002}

# kWh/token estimado para cálculo de CO2 (modelos medianos)
KWH_POR_TOKEN = {
    "meta-llama/llama-3.2-3b-instruct": 0.00000005,
    "qwen/qwen-2.5-7b-instruct":        0.00000010,
    "google/gemma-2-9b-it":             0.00000012,
}
KWH_DEFAULT     = 0.0000002
CO2_POR_KWH     = 400.0   # gramos CO2 por kWh (media UE)

MAX_HILOS    = 20
BATCH_BERT   = 1000

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
    print(f"  {len(datos)} datos ({errores_restantes} errores) + {len(metricas)} métricas + 1 cabecera = {total_lineas} líneas")


def run_pipeline(dataset_key, model_key, limit=None):
    if not OPENROUTER_API_KEY:
        print(f"OPENROUTER_API_KEY no configurada.")
        print(f"   Edita la cabecera de este archivo con tu key real.")
        return

    ds_config = DATASETS_CONFIG[dataset_key]
    model_id = MODELS_CONFIG[model_key]
    nombre_modelo = NOMBRES_LEGIBLES[model_key]
    extraer = EXTRACTORES[dataset_key]
    prompt = PROMPTS[dataset_key]

    dataset_names = {"totto": "totto", "webnlg": "webNLG", "kelm": "kelm_stem"}
    ds_label = dataset_names[dataset_key]

    if limit:
        output_name = f"results_{ds_label}_{limit}_{nombre_modelo}.csv"
    else:
        output_name = f"results_{ds_label}_{nombre_modelo}.csv"

    print(f"\n{'='*60}")
    print(f"PIPELINE: {nombre_modelo} sobre {ds_label.upper()}")
    print(f"Modelo:    {model_id}")
    print(f"Output:    {output_name}")
    print(f"{'='*60}")

    print("\nCargando métricas en memoria...")
    metrica_rouge  = evaluate.load('rouge')
    metrica_meteor = evaluate.load('meteor')
    metrica_bleu   = evaluate.load('bleu')
    metrica_bert   = evaluate.load('bertscore')

    print(f"\nCargando dataset: {ds_config['hf_name']}...")
    if ds_config['hf_name'] == 'local_csv':
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
            print(f"No se encontró {csv_path}")
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

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

    # RESUME: cargar CSV previo
    cabecera = ["Structured_Data", "Human_Reference", "LLM_Generated", "Time_per_row_seconds"]
    dict_procesados = {}

    if os.path.exists(output_name):
        print(f"\nCSV previo encontrado, retomando: {output_name}")
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

    def procesar_fila_api(datos_tarea):
        datos_estructurados, referencia = datos_tarea
        datos_estructurados = datos_estructurados.replace('\n', ' ').replace('\r', '')
        user_content = f"Data: {datos_estructurados}"

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user",   "content": user_content}
        ]

        generated_text = ""
        start_row_time = time.time()
        backoff_sec = 2.0

        for attempt in range(5):
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=ds_config['max_tokens'],
                    temperature=0.1
                )
                content = response.choices[0].message.content
                generated_text = content.strip() if content else "Error: modelo devolvio respuesta vacia"
                generated_text = generated_text.replace('\n', ' ').replace('\r', '')
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

        with csv_lock:
            nueva_fila = pd.DataFrame([{
                "Structured_Data":      datos_estructurados,
                "Human_Reference":      referencia,
                "LLM_Generated":        generated_text,
                "Time_per_row_seconds": row_time
            }])
            nueva_fila.to_csv(output_name, mode='a', header=False, index=False)

        return generated_text, referencia, datos_estructurados, row_time

    if len(items_a_procesar) > 0:
        print(f"\nLanzando {MAX_HILOS} hilos para {len(items_a_procesar)} filas restantes...")

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
        print("\nTodas las filas ya estaban procesadas.")

    total_time = sum(tiempos)
    avg_time   = total_time / len(tiempos) if tiempos else 0

    print(f"\nEvaluando las {len(preds_validas)} respuestas válidas...")

    if len(preds_validas) > 0:
        # Métricas globales
        try:
            rouge_score = round(metrica_rouge.compute(predictions=preds_validas, references=refs_validas)['rougeL'], 4)
        except Exception as e:
            print(f"ROUGE global falló: {e}"); rouge_score = 0.0
        try:
            meteor_score = round(metrica_meteor.compute(predictions=preds_validas, references=refs_validas)['meteor'], 4)
        except Exception as e:
            print(f"METEOR global falló: {e}"); meteor_score = 0.0
        try:
            bleu_score = round(metrica_bleu.compute(predictions=preds_validas, references=[[r] for r in refs_validas])['bleu'], 4)
        except Exception as e:
            print(f"BLEU global falló: {e}"); bleu_score = 0.0

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
                print(f"BERTScore batch {i} falló: {e}")
                all_f1.extend([0.0] * len(batch_preds))

        bert_score = round(sum(all_f1) / len(all_f1), 4) if all_f1 else 0.0

        # Métricas POR FILA
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
        print(f"Métricas por fila guardadas: {metricas_fila_name}")

        # Totales agregados
        coste_total = sum(costes)
        co2_total   = sum(co2s)
    else:
        rouge_score = meteor_score = bleu_score = bert_score = 0.0
        coste_total = co2_total = 0.0

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

    limpiar_csv_final(output_name, total_filas)


# MODO FIX: arregla errores
def fix_errors(dataset_key, model_key, limit=None):
    """Reprocesa SOLO las filas con Error: del CSV existente."""
    if not OPENROUTER_API_KEY:
        print(f"OPENROUTER_API_KEY no configurada.")
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
        print(f"No existe {output_name}. Lanza primero el modo normal.")
        return

    print(f"\nModo arreglar errores: {output_name}")
    df = pd.read_csv(output_name)
    df_err = df[df['LLM_Generated'].astype(str).str.startswith('Error:')].copy()
    print(f"  Filas con error a reprocesar: {len(df_err)}")
    if len(df_err) == 0:
        print(f"  No hay errores.")
        return

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

    def procesar_fila(idx_y_fila):
        idx, row = idx_y_fila
        datos = str(row['Structured_Data'])
        ref   = str(row['Human_Reference'])
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user",   "content": f"Data: {datos}"}
        ]
        start = time.time()
        gen = ""
        for attempt in range(5):
            try:
                response = client.chat.completions.create(
                    model=model_id, messages=messages,
                    max_tokens=ds_config['max_tokens'], temperature=0.1
                )
                content = response.choices[0].message.content
                gen = content.strip() if content else "Error: vacio"
                gen = gen.replace('\n', ' ').replace('\r', '')
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
    print(f"CSV actualizado. Relanza el pipeline normal para recalcular métricas.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["totto", "webnlg", "kelm"],
                        help="Dataset a procesar")
    parser.add_argument("--model",   choices=["llama", "gemma", "qwen"],
                        help="Modelo a usar")
    parser.add_argument("--limit",   type=int, default=None,
                        help="Limitar a N filas (test rápido)")
    parser.add_argument("--fix",     action="store_true",
                        help="Solo reprocesar errores")
    parser.add_argument("--all",     action="store_true",
                        help="Procesar TODAS las combinaciones (3 modelos x 3 bbdd = 9)")
    args = parser.parse_args()

    if args.all:
        combinaciones = [
            (ds, m) for ds in ["totto", "webnlg", "kelm"]
                    for m in ["llama", "gemma", "qwen"]
        ]
        print(f"Lanzando {len(combinaciones)} combinaciones (3 bbdd x 3 modelos):")
        for ds, m in combinaciones:
            print(f"  - {NOMBRES_LEGIBLES[m]} en {ds}")
        print()
        for ds, m in combinaciones:
            try:
                run_pipeline(ds, m, args.limit)
            except KeyboardInterrupt:
                print("\nInterrumpido")
                raise
            except Exception as e:
                print(f"Fallo en {m}/{ds}: {e}. Continuando...")
                import traceback
                traceback.print_exc()
        print("\nTodas las combinaciones procesadas")
        return

    if not args.dataset or not args.model:
        parser.error("Especifica --dataset y --model, o usa --all")

    if args.fix:
        fix_errors(args.dataset, args.model, args.limit)
    else:
        run_pipeline(args.dataset, args.model, args.limit)


if __name__ == "__main__":
    main()