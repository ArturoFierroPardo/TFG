#pip install datasets openai tqdm pandas evaluate rouge_score bert_score sacrebleu nltk "datasets<3.0.0"
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

# =========================================================
# CANDADO DE SEGURIDAD PARA MULTIHILO
# =========================================================
csv_lock = threading.Lock()

# =========================================================
# CONFIGURACIÓN DE DATASETS Y MODELOS
# =========================================================
DATASETS_CONFIG = {
    "totto": {
        "hf_name": "totto",
        "hf_version": None,
        "split": "train",
        "max_tokens": 150,
    },
    "webnlg": {
        "hf_name": "web_nlg",
        "hf_version": "release_v2.1",
        "split": "train",
        "max_tokens": 100,
    },
    "kelm": {
        "hf_name": "local_csv",
        "hf_version": None,
        "split": "train",
        "max_tokens": 100,
        "csv_file": "kelm_stem_60k.csv",
    },
}

MODELS_CONFIG = {
    "deepseek": "deepseek/deepseek-chat",
    "llama": "meta-llama/llama-3.3-70b-instruct",
    "qwen": "qwen/qwen-2.5-72b-instruct",
}

PROMPTS = {
    "totto": "You are a Data-to-Text generation API. Convert the provided table metadata and highlighted cell data into a single, highly fluent English sentence.\nRULE: Output ONLY the final English sentence without any extra comments, conversational filler, or greetings.",
    "webnlg": "You are a Data-to-Text generation API. Convert the provided table metadata and highlighted cell data into a single, highly fluent English sentence. Output ONLY the final English sentence without any extra comments, conversational filler, or greetings.",
    "kelm": "You are a Data-to-Text generation API. Convert the provided knowledge graph triples into a single, highly fluent English sentence.\nRULE: Output ONLY the final English sentence without any extra comments, conversational filler, or greetings.",
}

# =========================================================
# FUNCIONES DE EXTRACCIÓN POR DATASET
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
    "totto": extraer_totto,
    "webnlg": extraer_webnlg,
    "kelm": extraer_kelm,
}

# =========================================================
# LIMPIEZA FINAL DEL CSV
# =========================================================
def limpiar_csv_final(archivo, total_filas):
    print(f"\n🧹 Limpiando {archivo}...")
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
                # Solo guardar el error si no tenemos ya una versión correcta
                if not any(f == firma for f, _ in [(r[0].strip(), r[1].strip()) for r in filas_ok if len(r) >= 2]):
                    filas_error.append(row)
            else:
                # Quitar cualquier error previo con la misma firma
                filas_error = [e for e in filas_error if (e[0].strip(), e[1].strip()) != firma]
                filas_ok.append(row)
    
    # Coger las correctas (hasta el total), y si faltan, rellenar con errores
    datos = filas_ok[:total_filas]
    if len(datos) < total_filas:
        datos.extend(filas_error[:total_filas - len(datos)])
    
    metricas = metricas[-6:]
    
    with open(archivo, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(datos)
        writer.writerows(metricas)
    
    errores_restantes = sum(1 for d in datos if d[2].startswith("Error:"))
    total_lineas = 1 + len(datos) + len(metricas)
    print(f"   ✅ {len(datos)} datos ({errores_restantes} errores) + {len(metricas)} métricas + 1 cabecera = {total_lineas} líneas totales")

# =========================================================
# PIPELINE PRINCIPAL
# =========================================================
def run_pipeline(dataset_key, model_key, limit=None):
    OPENROUTER_API_KEY = "REDACTED"
    BATCH_SIZE_BERT = 1000
    MAX_HILOS = 20
    
    ds_config = DATASETS_CONFIG[dataset_key]
    model_id = MODELS_CONFIG[model_key]
    nombre_modelo = model_id.split('/')[-1].lower()
    extraer = EXTRACTORES[dataset_key]
    prompt = PROMPTS[dataset_key]
    
    # Nombres bonitos para cada dataset
    dataset_names = {
        "totto": "totto",
        "webnlg": "webNLG",
        "kelm": "kelm_stem",
    }
    ds_label = dataset_names[dataset_key]
    
    # Nombre del CSV: results_dataset_modelo o results_dataset_N_modelo
    if limit:
        output_name = f"results_{ds_label}_{limit}_{nombre_modelo}.csv"
    else:
        output_name = f"results_{ds_label}_{nombre_modelo}.csv"
    
    print("Cargando métricas en memoria...")
    metrica_rouge = evaluate.load('rouge')
    metrica_meteor = evaluate.load('meteor')
    metrica_bleu = evaluate.load('bleu')
    metrica_bert = evaluate.load('bertscore')
    
    print(f"\nCargando dataset: {ds_config['hf_name']}...")
    if ds_config['hf_name'] == 'local_csv':
        import pandas as pd_load
        csv_path = ds_config['csv_file']
        # Buscar en varias ubicaciones posibles
        posibles = [
            csv_path,
            os.path.join('base kelm', csv_path),
            os.path.join('..', 'base kelm', csv_path),
            os.path.join('..', '..', 'base kelm', csv_path),
            os.path.join(os.path.expanduser('~'), 'Desktop', 'TFG', 'base kelm', csv_path),
        ]
        csv_encontrado = None
        for p in posibles:
            if os.path.exists(p):
                csv_encontrado = p
                break
        if csv_encontrado is None:
            print(f"ERROR: No se encontro {csv_path}")
            print(f"Buscado en: {posibles}")
            return
        csv_path = csv_encontrado
        df_local = pd_load.read_csv(csv_path)
        dataset = df_local.to_dict('records')
        print(f"Cargado CSV local: {csv_path}")
    elif ds_config['hf_version']:
        dataset = load_dataset(ds_config['hf_name'], ds_config['hf_version'], split=ds_config['split'], trust_remote_code=True)
    else:
        dataset = load_dataset(ds_config['hf_name'], split=ds_config['split'], trust_remote_code=True)
    
    total_filas_dataset = len(dataset)
    total_filas = min(limit, total_filas_dataset) if limit else total_filas_dataset
    print(f"Dataset tiene {total_filas_dataset} filas | Procesando: {total_filas}")
    
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    
    print("\n" + "="*50)
    print(f"DATASET: {dataset_key.upper()} | MODELO: {nombre_modelo.upper()}")
    if limit:
        print(f"LÍMITE: {limit} filas")
    print(f"Guardado en: {output_name}")
    print(f"Hilos concurrentes: {MAX_HILOS}")
    print("="*50)
    
    # =========================================================
    # LECTOR CSV AUTO-REPARABLE Y ANTI-DUPLICADOS
    # =========================================================
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
        
        # Reescribir CSV solo con filas validas
        with open(output_name, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(cabecera)
            writer.writerows(datos_validos_para_guardar)
            
        total_recuperadas = sum(len(lista) for lista in dict_procesados.values())
        print(f"Encontradas {total_recuperadas} filas validas ya procesadas. Retomando...")
        if errores_encontrados > 0:
            print(f"Eliminados {errores_encontrados} errores previos del CSV. Se reprocesaran automaticamente.")
    else:
        pd.DataFrame(columns=["Structured_Data", "Human_Reference", "LLM_Generated", "Time_per_row_seconds"]).to_csv(output_name, index=False)
    
    preds_validas = []
    refs_validas = []
    tiempos = []
    items_a_procesar = []
    filas_procesadas_total = 0
    
    print("\nEvaluando filas guardadas y preparando envios...")
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
            tiempos.append(saved_time)
        else:
            items_a_procesar.append((datos_estructurados, referencia))
            
        filas_procesadas_total += 1

    # =========================================================
    # FUNCIÓN QUE EJECUTARÁ CADA HILO
    # =========================================================
    def procesar_fila_api(datos_tarea):
        datos_estructurados, referencia = datos_tarea
        
        # Limpiar saltos de línea
        datos_estructurados = datos_estructurados.replace('\n', ' ').replace('\r', '')
        
        user_content = f"Data: {datos_estructurados}"
            
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content}
        ]
        
        generated_text = ""
        start_row_time = time.time()
        
        for attempt in range(3):
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
                time.sleep(2)
                generated_text = f"Error: {str(e)}"
        
        end_row_time = time.time()
        row_time = round(end_row_time - start_row_time, 4)
        
        # GUARDADO PROTEGIDO POR EL CANDADO
        with csv_lock:
            nueva_fila = pd.DataFrame([{
                "Structured_Data": datos_estructurados,
                "Human_Reference": referencia,
                "LLM_Generated": generated_text,
                "Time_per_row_seconds": row_time
            }])
            nueva_fila.to_csv(output_name, mode='a', header=False, index=False)
            
        return generated_text, referencia, row_time

    # =========================================================
    # EJECUCIÓN CONCURRENTE (MULTIHILO)
    # =========================================================
    
    if len(items_a_procesar) > 0:
        print(f"\n🚀 Lanzando {MAX_HILOS} hilos para las {len(items_a_procesar)} filas restantes...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_HILOS) as executor:
            resultados = list(tqdm(
                executor.map(procesar_fila_api, items_a_procesar),
                total=len(items_a_procesar),
                desc=f"Generando ({nombre_modelo})"
            ))
            
        for gen_text, ref, r_time in resultados:
            if not gen_text.startswith("Error:"):
                preds_validas.append(gen_text)
                refs_validas.append(ref)
            tiempos.append(r_time)
    else:
        print("\n✅ Todas las filas ya estaban procesadas.")
    
    total_time = sum(tiempos)
    avg_time = sum(tiempos) / len(tiempos) if tiempos else 0
    
    # =========================================================
    # MÉTRICAS GLOBALES + POR FILA
    # =========================================================
    print(f"\nEvaluando las {len(preds_validas)} respuestas válidas...")
    
    if len(preds_validas) > 0:
        # Métricas globales
        res_rouge = metrica_rouge.compute(predictions=preds_validas, references=refs_validas)
        res_meteor = metrica_meteor.compute(predictions=preds_validas, references=refs_validas)
        
        rouge_score = round(res_rouge['rougeL'], 4)
        meteor_score = round(res_meteor['meteor'], 4)
        
        # BLEU global
        bleu_score = round(metrica_bleu.compute(
            predictions=preds_validas,
            references=[[r] for r in refs_validas]
        )['bleu'], 4)
        
        # BERTScore global en batches
        print(f"Calculando BERTScore en batches de {BATCH_SIZE_BERT}...")
        all_f1 = []
        total_batches = (len(preds_validas) + BATCH_SIZE_BERT - 1) // BATCH_SIZE_BERT
        for i in tqdm(range(0, len(preds_validas), BATCH_SIZE_BERT), desc="BERTScore", total=total_batches):
            batch_preds = preds_validas[i:i + BATCH_SIZE_BERT]
            batch_refs = refs_validas[i:i + BATCH_SIZE_BERT]
            res = metrica_bert.compute(
                predictions=batch_preds,
                references=batch_refs,
                lang="en",
                model_type="distilbert-base-uncased"
            )
            all_f1.extend(res['f1'])
        
        bert_score = round(sum(all_f1) / len(all_f1), 4)
        
        # Métricas POR FILA para boxplots
        print(f"\nCalculando métricas por fila ({len(preds_validas)} filas)...")
        rouges_fila = []
        for i in tqdm(range(len(preds_validas)), desc="ROUGE-L por fila"):
            r = metrica_rouge.compute(predictions=[preds_validas[i]], references=[refs_validas[i]])
            rouges_fila.append(round(r['rougeL'], 4))
        
        meteors_fila = []
        for i in tqdm(range(len(preds_validas)), desc="METEOR por fila"):
            r = metrica_meteor.compute(predictions=[preds_validas[i]], references=[refs_validas[i]])
            meteors_fila.append(round(r['meteor'], 4))
        
        bleus_fila = []
        for i in tqdm(range(len(preds_validas)), desc="BLEU por fila"):
            try:
                r = metrica_bleu.compute(predictions=[preds_validas[i]], references=[[refs_validas[i]]])
                bleus_fila.append(round(r['bleu'], 4))
            except:
                bleus_fila.append(0.0)
        
        bertscores_fila = [round(f, 4) for f in all_f1]
        
        # Guardar métricas por fila
        metricas_fila_name = output_name.replace("results_", "metricas_por_fila_")
        pd.DataFrame({
            'ROUGE_L': rouges_fila,
            'METEOR': meteors_fila,
            'BLEU': bleus_fila,
            'BERTScore': bertscores_fila,
            'Time_seconds': tiempos[:len(rouges_fila)]
        }).to_csv(metricas_fila_name, index=False)
        print(f"Métricas por fila guardadas: {metricas_fila_name}")
    else:
        rouge_score = meteor_score = bleu_score = bert_score = 0.0
    
    metricas_finales = pd.DataFrame([
        {"Structured_Data": "--- TOTAL TIME ---", "Human_Reference": "", "LLM_Generated": "", "Time_per_row_seconds": round(total_time, 4)},
        {"Structured_Data": "--- AVERAGE TIME ---", "Human_Reference": "", "LLM_Generated": "", "Time_per_row_seconds": round(avg_time, 4)},
        {"Structured_Data": "--- METRICA: ROUGE-L ---", "Human_Reference": "", "LLM_Generated": rouge_score, "Time_per_row_seconds": ""},
        {"Structured_Data": "--- METRICA: METEOR ---", "Human_Reference": "", "LLM_Generated": meteor_score, "Time_per_row_seconds": ""},
        {"Structured_Data": "--- METRICA: BLEU ---", "Human_Reference": "", "LLM_Generated": bleu_score, "Time_per_row_seconds": ""},
        {"Structured_Data": "--- METRICA: BERTSCORE ---", "Human_Reference": "", "LLM_Generated": bert_score, "Time_per_row_seconds": ""},
        {"Structured_Data": "--- RESPUESTAS VALIDAS ---", "Human_Reference": "", "LLM_Generated": f"{len(preds_validas)}/{len(tiempos)}", "Time_per_row_seconds": ""}
    ])
    metricas_finales.to_csv(output_name, mode='a', header=False, index=False)
    
    print(f"\n{'='*50}")
    print(f"COMPLETADO {nombre_modelo.upper()} en {dataset_key.upper()}")
    print(f"ROUGE-L: {rouge_score} | METEOR: {meteor_score} | BLEU: {bleu_score} | BERTScore: {bert_score}")
    print(f"Respuestas válidas: {len(preds_validas)}/{filas_procesadas_total}")
    print(f"{'='*50}")
    
    # Limpieza final
    limpiar_csv_final(output_name, total_filas)

# =========================================================
# MODO FIX: arregla errores y recalcula metricas
# =========================================================
def fix_errors(dataset_key, model_key, limit=None):
    OPENROUTER_API_KEY = "REDACTED"
    BATCH_SIZE_BERT = 1000
    
    ds_config = DATASETS_CONFIG[dataset_key]
    model_id = MODELS_CONFIG[model_key]
    nombre_modelo = model_id.split('/')[-1].lower()
    prompt = PROMPTS[dataset_key]
    
    dataset_names = {"totto": "totto", "webnlg": "webNLG", "kelm": "kelm_stem"}
    ds_label = dataset_names[dataset_key]
    
    if limit:
        output_name = f"results_{ds_label}_{limit}_{nombre_modelo}.csv"
    else:
        output_name = f"results_{ds_label}_{nombre_modelo}.csv"
    
    if not os.path.exists(output_name):
        print(f"No se encontro {output_name}")
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
    
    # Reintentar errores con estrategia progresiva
    if errores > 0:
        print(f"\nReintentando {errores} errores (3 intentos progresivos)...")
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
        arreglados = 0
        
        prompt_reforzado = prompt.replace(
            "Output ONLY the final English sentence",
            "You MUST output exactly one English sentence. Never output an empty response.\nOutput ONLY the final English sentence"
        )
        
        for i, row in enumerate(filas):
            if not row[2].startswith('Error:'):
                continue
            
            datos = row[0]
            if len(datos) > 5000:
                datos = datos[:5000] + "..."
            datos = datos.replace('\n', ' ').replace('\r', '')
            
            print(f"  Fila {i+1}: {datos[:60]}...")
            
            # 3 intentos progresivos: temp creciente + prompt reforzado + reformulación
            intentos = [
                (0.3, prompt_reforzado, f"Data: {datos}"),
                (0.5, prompt_reforzado, f"Data: {datos}"),
                (0.7, prompt_reforzado, f"Data: The following is structured data. Please describe it in one sentence: {datos}"),
            ]
            
            generated = None
            for attempt, (temp, sys_prompt, user_content) in enumerate(intentos, 1):
                try:
                    print(f"    Intento {attempt} (temp={temp})...", end=" ")
                    start_time = time.time()
                    response = client.chat.completions.create(
                        model=model_id,
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_content}
                        ],
                        max_tokens=ds_config['max_tokens'],
                        temperature=temp,
                    )
                    content = response.choices[0].message.content
                    row_time = round(time.time() - start_time, 4)
                    
                    if content and content.strip():
                        generated = content.strip().replace('\n', ' ').replace('\r', '')
                        print(f"OK: {generated[:60]}")
                        filas[i] = [row[0], row[1], generated, str(row_time)]
                        arreglados += 1
                        break
                    else:
                        print("vacio")
                        time.sleep(1)
                except Exception as e:
                    print(f"error: {str(e)[:50]}")
                    time.sleep(2)
            
            if generated is None:
                print(f"    FALLO TOTAL: no se pudo arreglar")
        
        print(f"Arreglados: {arreglados}/{errores}")
    else:
        print("No hay errores que arreglar.")
    
    # Recalcular metricas
    print("\nRecalculando metricas...")
    metrica_rouge = evaluate.load('rouge')
    metrica_meteor = evaluate.load('meteor')
    metrica_bleu = evaluate.load('bleu')
    metrica_bert = evaluate.load('bertscore')
    
    preds = []
    refs = []
    tiempos_lista = []
    
    for row in filas:
        if not row[2].startswith('Error:'):
            preds.append(row[2])
            refs.append(row[1])
        try:
            tiempos_lista.append(float(str(row[3]).replace('"', '').replace(';', '').strip()))
        except:
            tiempos_lista.append(1.5)
    
    total_time = sum(tiempos_lista)
    avg_time = total_time / len(tiempos_lista) if tiempos_lista else 0
    
    print(f"Respuestas validas: {len(preds)}/{len(filas)}")
    
    rouge_score = round(metrica_rouge.compute(predictions=preds, references=refs)['rougeL'], 4)
    meteor_score = round(metrica_meteor.compute(predictions=preds, references=refs)['meteor'], 4)
    bleu_score = round(metrica_bleu.compute(predictions=preds, references=[[r] for r in refs])['bleu'], 4)
    print(f"  ROUGE-L: {rouge_score} | METEOR: {meteor_score} | BLEU: {bleu_score}")
    
    print(f"Calculando BERTScore en batches de {BATCH_SIZE_BERT}...")
    all_f1 = []
    total_batches = (len(preds) + BATCH_SIZE_BERT - 1) // BATCH_SIZE_BERT
    for i in tqdm(range(0, len(preds), BATCH_SIZE_BERT), desc="BERTScore", total=total_batches):
        res = metrica_bert.compute(
            predictions=preds[i:i+BATCH_SIZE_BERT], references=refs[i:i+BATCH_SIZE_BERT],
            lang="en", model_type="distilbert-base-uncased"
        )
        all_f1.extend(res['f1'])
    bert_score = round(sum(all_f1) / len(all_f1), 4)
    print(f"  BERTScore: {bert_score}")
    
    # Métricas por fila
    print(f"\nCalculando métricas por fila ({len(preds)} filas)...")
    rouges_fila = []
    for i in tqdm(range(len(preds)), desc="ROUGE-L por fila"):
        r = metrica_rouge.compute(predictions=[preds[i]], references=[refs[i]])
        rouges_fila.append(round(r['rougeL'], 4))
    
    meteors_fila = []
    for i in tqdm(range(len(preds)), desc="METEOR por fila"):
        r = metrica_meteor.compute(predictions=[preds[i]], references=[refs[i]])
        meteors_fila.append(round(r['meteor'], 4))
    
    bleus_fila = []
    for i in tqdm(range(len(preds)), desc="BLEU por fila"):
        try:
            r = metrica_bleu.compute(predictions=[preds[i]], references=[[refs[i]]])
            bleus_fila.append(round(r['bleu'], 4))
        except:
            bleus_fila.append(0.0)
    
    bertscores_fila = [round(f, 4) for f in all_f1]
    
    metricas_fila_name = output_name.replace("results_", "metricas_por_fila_")
    pd.DataFrame({
        'ROUGE_L': rouges_fila, 'METEOR': meteors_fila, 'BLEU': bleus_fila,
        'BERTScore': bertscores_fila, 'Time_seconds': tiempos_lista[:len(rouges_fila)]
    }).to_csv(metricas_fila_name, index=False)
    print(f"Métricas por fila guardadas: {metricas_fila_name}")
    
    # Reescribir CSV completo
    metricas_nuevas = [
        ["--- TOTAL TIME ---", "", "", str(round(total_time, 4))],
        ["--- AVERAGE TIME ---", "", "", str(round(avg_time, 4))],
        ["--- METRICA: ROUGE-L ---", "", str(rouge_score), ""],
        ["--- METRICA: METEOR ---", "", str(meteor_score), ""],
        ["--- METRICA: BLEU ---", "", str(bleu_score), ""],
        ["--- METRICA: BERTSCORE ---", "", str(bert_score), ""],
        ["--- RESPUESTAS VALIDAS ---", "", f"{len(preds)}/{len(filas)}", ""],
    ]
    
    with open(output_name, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(filas)
        writer.writerows(metricas_nuevas)
    
    errores_final = sum(1 for r in filas if r[2].startswith('Error:'))
    print(f"\n{'='*50}")
    print(f"COMPLETADO FIX: {nombre_modelo.upper()} en {dataset_key.upper()}")
    print(f"ROUGE-L: {rouge_score} | METEOR: {meteor_score} | BLEU: {bleu_score} | BERTScore: {bert_score}")
    print(f"Filas: {len(filas)} | Errores restantes: {errores_final}")
    print(f"{'='*50}")

# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline LLM Data-to-Text (Multihilo)")
    parser.add_argument("--dataset", required=True, choices=list(DATASETS_CONFIG.keys()),
                        help="Dataset a usar: totto, webnlg, kelm")
    parser.add_argument("--model", required=True, choices=list(MODELS_CONFIG.keys()),
                        help="Modelo a usar: deepseek, llama, qwen")
    parser.add_argument("--limit", type=int, default=None,
                        help="Numero de filas a procesar (por defecto: todas)")
    parser.add_argument("--fix", action="store_true",
                        help="Solo arreglar errores y recalcular metricas")
    
    args = parser.parse_args()
    
    if args.fix:
        print(f"\nModo FIX: {args.dataset.upper()} + {args.model.upper()}")
        fix_errors(args.dataset, args.model, args.limit)
    else:
        print(f"\nPipeline LLM: {args.dataset.upper()} + {args.model.upper()}", end="")
        if args.limit:
            print(f" | Limite: {args.limit} filas")
        else:
            print(" | Dataset completo")
        run_pipeline(args.dataset, args.model, args.limit)