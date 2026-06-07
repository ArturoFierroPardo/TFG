"""
Pipeline Fine-tuning LoRA - adapta Mini-SLMs al dominio de teleco.

Para cada Mini-SLM (Qwen 1.5B, Llama 1B, Gemma 3 1B):
  1. Carga el modelo base
  2. Aplica LoRA con hiperparámetros estándar de literatura
  3. Entrena sobre el split TRAIN (subtemas exclusivos, no leakage)
  4. Valida sobre el split VAL al final de cada época
  5. Guarda el adapter LoRA + curva de aprendizaje
  6. Inferencia sobre TODO el dataset con el modelo fine-tuneado
  7. Calcula métricas + costes + CO2

HIPERPARÁMETROS (estándar literatura LoRA):
  - LR = 2e-4
  - Rank = 8
  - Alpha = 16
  - Dropout = 0.05
  - Épocas = 3
  - Batch size = adaptado a GPU

ROBUSTO PARA 4 DÍAS SIN SUPERVISIÓN:
  - Estado en estado_finetuning.json
  - Si fine-tuning de un modelo se cae, reanuda desde checkpoint de Trainer
  - Si inferencia se cae, reanuda fila a fila
  - Si un modelo entero falla, salta al siguiente

Uso:
    python pipeline_finetuning.py
    python pipeline_finetuning.py --csv dataset_teleco.csv --skip llama-1b
"""

import sys
# Forzar UTF-8 en stdout/stderr (necesario en Windows con cp1252)
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


import os, sys, json, csv, time, gc, argparse, datetime as dt

# --- HUGGINGFACE TOKEN ------------------------------------
HF_TOKEN = "REDACTED"

def hf_login():
    if not HF_TOKEN or HF_TOKEN == "hf_PEGA_AQUI_TU_TOKEN":
        print("[!] HF_TOKEN no configurado")
        return
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN, add_to_git_credential=False)
        print("[OK] Login HuggingFace OK")
    except Exception as e:
        print(f"[!] Error login HF: {e}")

hf_login()

# --- Config -----------------------------------------------
OUTDIR_BASE = "resultados_arturo"
SPLITS_PATH = os.path.join(OUTDIR_BASE, "splits", "splits_por_subtema.json")

MINI_SLMS = [
    ("qwen-1.7b",  "Qwen/Qwen3-1.7B",                   "Qwen3_1.7B_FT"),
    ("llama-1b",   "meta-llama/Llama-3.2-1B-Instruct",  "Llama_1B_FT"),
    ("gemma-3-1b", "google/gemma-3-1b-it",              "Gemma_3_1B_FT"),
]

# Hiperparámetros LoRA (estándar literatura)
LORA_LR        = 2e-4
LORA_RANK      = 8
LORA_ALPHA     = 16
LORA_DROPOUT   = 0.05
NUM_EPOCAS_FT  = 3
MAX_LEN        = 512   # longitud máxima de secuencia

KWH_POR_TOKEN = {"qwen-1.7b": 4e-8, "llama-1b": 3e-8, "gemma-3-1b": 3e-8}
CO2_POR_KWH = 475
TOKENS_PROMPT_SISTEMA = 45

ESTADO_PATH = os.path.join(OUTDIR_BASE, "estado_finetuning.json")
LOG_PATH    = os.path.join(OUTDIR_BASE, "pipeline_finetuning.log")


# =========================================================
# UTILIDADES
# =========================================================

def log(msg, level="INFO"):
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def cargar_estado():
    """Carga estado anterior. Si hay claves nuevas en MINI_SLMS, las añade."""
    estado_default = {clave: {"finetuning": "pendiente",
                              "inferencia": "pendiente",
                              "metricas":   "pendiente"} for clave, *_ in MINI_SLMS}
    if not os.path.exists(ESTADO_PATH):
        return estado_default

    try:
        with open(ESTADO_PATH, encoding="utf-8") as f:
            estado = json.load(f)
    except Exception:
        return estado_default

    # Asegurar que todas las claves actuales están en el estado
    for clave in estado_default:
        if clave not in estado:
            estado[clave] = estado_default[clave]
        else:
            # Asegurar que cada sub-key existe
            for sub in ["finetuning", "inferencia", "metricas"]:
                if sub not in estado[clave]:
                    estado[clave][sub] = "pendiente"
    return estado


def guardar_estado(estado):
    os.makedirs(os.path.dirname(ESTADO_PATH), exist_ok=True)
    with open(ESTADO_PATH, "w", encoding="utf-8") as f:
        json.dump(estado, f, indent=2)


def detectar_batch_size():
    try:
        import torch
        if not torch.cuda.is_available():
            return 1
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram >= 11: return 4
        elif vram >= 7: return 2
        else: return 1
    except: return 1


def detectar_dtype():
    """
    Detecta el dtype óptimo según capacidad de la GPU.
    - Ampere (RTX 30xx/40xx) o más nuevo: bfloat16
    - Turing (RTX 20xx) o anterior: float16 (no soporta bfloat16)
    - CPU: float32
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return torch.float32
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            return torch.float16
    except Exception:
        try:
            import torch
            return torch.float16
        except:
            return None


def soporta_bf16():
    """True si la GPU soporta bfloat16 nativamente."""
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except:
        return False


def liberar_gpu():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =========================================================
# DATASET PARA FINE-TUNING
# =========================================================

def es_modelo_gemma(model_id_or_clave):
    """Detecta si es Gemma (no acepta role 'system')."""
    return "gemma" in str(model_id_or_clave).lower()


def es_modelo_qwen3(model_id_or_clave):
    """Detecta si es Qwen 3 (necesita enable_thinking=False)."""
    s = str(model_id_or_clave).lower()
    # Detección por nombre HF completo
    if "qwen3" in s or "qwen-3" in s:
        return True
    # Detección por clave corta (qwen-1.7b corresponde a Qwen3 en este TFG)
    if "qwen-1.7b" in s or "qwen-1_7b" in s or "qwen-0.6b" in s:
        return True
    return False


def construir_dataset_ft(csv_path, splits_path, tokenizer, modelo_clave):
    """
    Construye datasets train/val tokenizados con plantilla de chat.
    Detecta Gemma para combinar sistema+usuario.
    """
    import pandas as pd
    from datasets import Dataset

    df = pd.read_csv(csv_path)
    with open(splits_path, encoding='utf-8') as f:
        splits = json.load(f)

    train_ids = set(splits["train_ids"])
    val_ids   = set(splits["val_ids"])

    df_train = df[df['id'].isin(train_ids)].reset_index(drop=True)
    df_val   = df[df['id'].isin(val_ids)].reset_index(drop=True)

    # Filtrar filas con NaN o vacías (defensivo)
    n_train_before = len(df_train)
    n_val_before = len(df_val)

    df_train = df_train.dropna(subset=['pregunta', 'respuesta'])
    df_train['pregunta']  = df_train['pregunta'].astype(str).str.strip()
    df_train['respuesta'] = df_train['respuesta'].astype(str).str.strip()
    df_train = df_train[df_train['pregunta'].str.len() > 0]
    df_train = df_train[df_train['respuesta'].str.len() > 0]
    df_train = df_train.reset_index(drop=True)

    df_val = df_val.dropna(subset=['pregunta', 'respuesta'])
    df_val['pregunta']  = df_val['pregunta'].astype(str).str.strip()
    df_val['respuesta'] = df_val['respuesta'].astype(str).str.strip()
    df_val = df_val[df_val['pregunta'].str.len() > 0]
    df_val = df_val[df_val['respuesta'].str.len() > 0]
    df_val = df_val.reset_index(drop=True)

    if len(df_train) < n_train_before:
        log(f"  [!] Train: descartadas {n_train_before - len(df_train)} filas con NaN/vacías", "WARN")
    if len(df_val) < n_val_before:
        log(f"  [!] Val: descartadas {n_val_before - len(df_val)} filas con NaN/vacías", "WARN")

    log(f"  Train: {len(df_train)} | Val: {len(df_val)}")

    sistema = ("Eres un experto en Telecomunicaciones. Responde de forma "
               "técnica y concisa a la pregunta del usuario.")

    is_gemma = es_modelo_gemma(modelo_clave)
    is_qwen3 = es_modelo_qwen3(modelo_clave) or es_modelo_qwen3(tokenizer.name_or_path if hasattr(tokenizer, 'name_or_path') else "")

    def formatear(ejemplo):
        # Cast defensivo a str
        pregunta  = str(ejemplo['pregunta']).strip()
        respuesta = str(ejemplo['respuesta']).strip()

        if is_gemma:
            msgs = [
                {"role": "user",      "content": f"{sistema}\n\n{pregunta}"},
                {"role": "assistant", "content": respuesta},
            ]
        else:
            msgs = [
                {"role": "system",    "content": sistema},
                {"role": "user",      "content": pregunta},
                {"role": "assistant", "content": respuesta},
            ]
        # Aplicar chat template con manejo de Qwen 3 thinking mode
        text = None
        if is_qwen3:
            # Qwen 3 -> enable_thinking=False (modo non-thinking)
            try:
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, enable_thinking=False)
            except (TypeError, ValueError):
                # transformers viejo no soporta enable_thinking -> usar normal
                try:
                    text = tokenizer.apply_chat_template(msgs, tokenize=False)
                except Exception:
                    text = None
        else:
            try:
                text = tokenizer.apply_chat_template(msgs, tokenize=False)
            except Exception:
                text = None

        if text is None:
            # Fallback simple si el tokenizer no tiene chat template (improbable)
            text = (f"Sistema: {sistema}\n"
                    f"Usuario: {pregunta}\n"
                    f"Asistente: {respuesta}")
        return {"text": text}

    def tokenizar(batch):
        return tokenizer(batch["text"],
                         truncation=True, max_length=MAX_LEN,
                         padding=False)

    ds_train = Dataset.from_pandas(df_train[['pregunta', 'respuesta']])
    ds_val   = Dataset.from_pandas(df_val[['pregunta', 'respuesta']])
    ds_train = ds_train.map(formatear, remove_columns=ds_train.column_names)
    ds_val   = ds_val.map(  formatear, remove_columns=ds_val.column_names)
    ds_train = ds_train.map(tokenizar, batched=True, remove_columns=["text"])
    ds_val   = ds_val.map(  tokenizar, batched=True, remove_columns=["text"])
    return ds_train, ds_val


# =========================================================
# FINE-TUNING DE UN MODELO
# =========================================================

def finetune_modelo(clave, hf_id, args):
    log("=" * 60)
    log(f"FINE-TUNING: {clave} ({hf_id})")
    log("=" * 60)

    outdir = os.path.join(OUTDIR_BASE, "minislm_finetuned", clave)
    os.makedirs(outdir, exist_ok=True)
    adapter_dir = os.path.join(outdir, "lora_adapter")
    log_path    = os.path.join(outdir, "training_log.json")
    curve_png   = os.path.join(outdir, "learning_curve.png")

    # Si ya existe el adapter, saltamos
    if os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        log("  [OK] Adapter ya existe, saltando fine-tuning")
        return True

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              TrainingArguments, Trainer,
                              DataCollatorForLanguageModeling)
    from peft import LoraConfig, get_peft_model, TaskType

    log(f"  Cargando tokenizer y modelo...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = detectar_dtype()
    use_bf16 = soporta_bf16()
    log(f"  Usando dtype: {dtype} (bf16={use_bf16})")

    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=dtype, device_map="auto",
    )

    log(f"  Aplicando LoRA (rank={LORA_RANK}, alpha={LORA_ALPHA})...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log(f"  Parámetros entrenables: {n_train:,} / {n_total:,} ({100*n_train/n_total:.2f}%)")

    log(f"  Construyendo dataset...")
    ds_train, ds_val = construir_dataset_ft(args.csv, SPLITS_PATH, tokenizer, clave)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    batch_size = detectar_batch_size()
    grad_accum = max(1, 16 // batch_size)
    log(f"  Batch size: {batch_size}, grad accum: {grad_accum}")

    # En CPU no se debe usar fp16 ni bf16
    import torch as _torch
    has_gpu = _torch.cuda.is_available()
    use_fp16 = (not use_bf16) and has_gpu
    use_bf16_safe = use_bf16 and has_gpu

    training_args = TrainingArguments(
        output_dir=os.path.join(outdir, "trainer_ckpt"),
        num_train_epochs=NUM_EPOCAS_FT,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=LORA_LR,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=50,
        # Checkpoints frecuentes para resume robusto en caso de caída.
        # eval_strategy y save_strategy deben coincidir si load_best_model_at_end=True.
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,    # solo mantenemos los 2 últimos checkpoints
        bf16=use_bf16_safe,
        fp16=use_fp16,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=collator,
    )

    # --- RESUME del Trainer si hay checkpoint previo --------
    trainer_ckpt_dir = os.path.join(outdir, "trainer_ckpt")
    has_checkpoint = False
    if os.path.exists(trainer_ckpt_dir):
        try:
            entries = os.listdir(trainer_ckpt_dir)
            has_checkpoint = any(d.startswith("checkpoint-") for d in entries)
        except Exception:
            has_checkpoint = False

    log(f"  [GO] Empezando entrenamiento{' (RESUME)' if has_checkpoint else ''}...")
    t0 = time.time()
    try:
        if has_checkpoint:
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
    except Exception as e:
        log(f"[X] Entrenamiento falló: {e}", "ERROR")
        log(f"  Al relanzar, reanudará desde el último checkpoint del Trainer", "INFO")
        del model, tokenizer, trainer
        liberar_gpu()
        raise
    elapsed = (time.time() - t0) / 60
    log(f"  [OK] Entrenamiento completado en {elapsed:.1f} min")

    # Guardar adapter LoRA y log
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    log(f"  [OK] Adapter guardado: {adapter_dir}")

    # Extraer log de entrenamiento de Trainer
    history = trainer.state.log_history
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    # Curva de aprendizaje
    try:
        import matplotlib.pyplot as plt
        train_losses, val_losses, epochs_t, epochs_v = [], [], [], []
        for entry in history:
            if 'loss' in entry and 'eval_loss' not in entry:
                train_losses.append(entry['loss'])
                epochs_t.append(entry.get('epoch', 0))
            if 'eval_loss' in entry:
                val_losses.append(entry['eval_loss'])
                epochs_v.append(entry.get('epoch', 0))

        fig, ax = plt.subplots(figsize=(8, 5))
        if train_losses:
            ax.plot(epochs_t, train_losses, label="Train", marker='.', alpha=0.7)
        if val_losses:
            ax.plot(epochs_v, val_losses, label="Val", marker='s', linewidth=2)
        ax.set_xlabel("Época")
        ax.set_ylabel("Loss")
        ax.set_title(f"Curva de aprendizaje LoRA - {clave}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.savefig(curve_png, dpi=200, bbox_inches='tight')
        plt.close(fig)
        log(f"  [OK] Curva guardada: {curve_png}")
    except Exception as e:
        log(f"  [!] Error generando curva: {e}", "WARN")

    # Limpiar trainer checkpoints SOLO SI el adapter está bien guardado
    # (si el adapter no se guardó, mantenemos los checkpoints para recuperar)
    import shutil
    adapter_ok = os.path.exists(os.path.join(adapter_dir, "adapter_config.json"))
    trainer_ckpt_dir = os.path.join(outdir, "trainer_ckpt")
    if adapter_ok and os.path.exists(trainer_ckpt_dir):
        shutil.rmtree(trainer_ckpt_dir, ignore_errors=True)
        log(f"  Limpiados checkpoints intermedios de Trainer")
    elif not adapter_ok:
        log(f"  [!] Adapter no encontrado, manteniendo checkpoints por seguridad", "WARN")

    del model, tokenizer, trainer
    liberar_gpu()
    return True


# =========================================================
# INFERENCIA CON MODELO FINE-TUNEADO
# =========================================================

def _generar_inferencia_split(split_name, split_ids, df, clave, hf_id, args,
                                model, tokenizer, is_gemma, is_qwen3, sistema):
    """Genera inferencia para un split concreto (val o test) y guarda CSV.
    Devuelve True si OK."""
    import torch
    from tqdm import tqdm

    outdir = os.path.join(OUTDIR_BASE, "minislm_finetuned", clave)
    results_csv = os.path.join(outdir, f"results_teleco_{clave}_ft_{split_name}.csv")

    df_split = df[df['id'].isin(split_ids)].reset_index(drop=True)
    n_total = len(df_split)
    log(f"  [{split_name}] {n_total} pares a procesar")

    # Resume robusto
    n_done = 0
    if os.path.exists(results_csv):
        try:
            import pandas as pd
            df_done = pd.read_csv(results_csv, encoding='utf-8')
            len_actual = len(df_done)
            if len_actual >= n_total:
                n_done = n_total
            elif len_actual > 0:
                n_done = len_actual - 1
                df_done.head(n_done).to_csv(results_csv, index=False)
            else:
                n_done = 0
            log(f"  [{split_name}] Reanudando desde fila {n_done}")
        except Exception as e:
            log(f"  [{split_name}] Error leyendo CSV: {e}", "WARN")
            n_done = 0
            with open(results_csv, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Structured_Data', 'Human_Reference',
                                 'LLM_Generated', 'Time_per_row_seconds'])
    else:
        with open(results_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Structured_Data', 'Human_Reference',
                             'LLM_Generated', 'Time_per_row_seconds'])

    if n_done >= n_total:
        log(f"  [{split_name}] [OK] Ya procesado por completo")
        return True

    def generar(pregunta, max_tokens=256):
        if is_gemma:
            msgs = [{"role": "user", "content": f"{sistema}\n\n{pregunta}"}]
        else:
            msgs = [{"role": "system", "content": sistema},
                    {"role": "user",   "content": pregunta}]
        inputs = None
        if is_qwen3:
            try:
                inputs = tokenizer.apply_chat_template(
                    msgs, return_tensors="pt", add_generation_prompt=True,
                    enable_thinking=False,
                ).to(model.device)
            except (TypeError, ValueError):
                # transformers viejo no soporta enable_thinking
                inputs = tokenizer.apply_chat_template(
                    msgs, return_tensors="pt", add_generation_prompt=True,
                ).to(model.device)
        else:
            inputs = tokenizer.apply_chat_template(
                msgs, return_tensors="pt", add_generation_prompt=True,
            ).to(model.device)
        with torch.no_grad():
            out = model.generate(inputs, max_new_tokens=max_tokens,
                                 do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        new_tokens = out[0][inputs.shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        # Limpiar restos de <think>
        if "<think>" in text:
            import re
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        # Limpiar saltos de línea para no romper el CSV
        text = text.replace('\n', ' ').replace('\r', ' ').strip()
        return text

    log(f"  [{split_name}] Procesando {n_done} -> {n_total}...")
    with open(results_csv, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for idx in tqdm(range(n_done, n_total), desc=f"{clave} FT {split_name}"):
            row = df_split.iloc[idx]
            pregunta  = str(row['pregunta'])
            respuesta = str(row['respuesta'])
            t0 = time.time()
            try:
                generado = generar(pregunta)
            except Exception as e:
                generado = f"Error: {str(e)[:80]}"
            elapsed = time.time() - t0
            # Limpiar saltos de línea para no romper el CSV
            preg_limpia = pregunta.replace('\n', ' ').replace('\r', ' ').strip()
            resp_limpia = respuesta.replace('\n', ' ').replace('\r', ' ').strip()
            writer.writerow([preg_limpia, resp_limpia, generado, round(elapsed, 4)])
            f.flush()
    return True


def inferencia_modelo_ft(clave, hf_id, args):
    """Inferencia post-FT sobre val + test + valtest combinado.
    Genera 3 CSVs: results_teleco_<clave>_ft_val.csv,
                   results_teleco_<clave>_ft_test.csv,
                   results_teleco_<clave>_ft_valtest.csv (combinado)
    """
    log("=" * 60)
    log(f"INFERENCIA FT: {clave}")
    log("=" * 60)

    outdir = os.path.join(OUTDIR_BASE, "minislm_finetuned", clave)
    adapter_dir = os.path.join(outdir, "lora_adapter")

    if not os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        log(f"  [X] No existe adapter, ¿se hizo el fine-tuning?", "ERROR")
        return False

    if not os.path.exists(SPLITS_PATH):
        log(f"  [X] No existe {SPLITS_PATH}", "ERROR")
        return False

    with open(SPLITS_PATH, encoding='utf-8') as f:
        splits = json.load(f)
    val_ids  = set(splits["val_ids"])
    test_ids = set(splits["test_ids"])
    log(f"  Val: {len(val_ids)} | Test: {len(test_ids)}")

    import pandas as pd
    df = pd.read_csv(args.csv)

    # Cargar modelo + adapter UNA SOLA VEZ
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    log(f"  Cargando modelo base + adapter LoRA (UNA SOLA VEZ)...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype_inf = detectar_dtype()
    base = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=dtype_inf, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    sistema = ("Eres un experto en Telecomunicaciones. Responde de forma "
               "técnica y concisa a la pregunta del usuario.")
    is_gemma = es_modelo_gemma(clave)
    is_qwen3 = es_modelo_qwen3(clave) or es_modelo_qwen3(hf_id)

    try:
        # 1) Inferencia VAL
        ok_val = _generar_inferencia_split("val", val_ids, df, clave, hf_id, args,
                                            model, tokenizer, is_gemma, is_qwen3, sistema)
        # 2) Inferencia TEST
        ok_test = _generar_inferencia_split("test", test_ids, df, clave, hf_id, args,
                                             model, tokenizer, is_gemma, is_qwen3, sistema)
    finally:
        del model, base, tokenizer
        liberar_gpu()

    # 3) Combinar val+test en un único CSV
    if ok_val and ok_test:
        results_val  = os.path.join(outdir, f"results_teleco_{clave}_ft_val.csv")
        results_test = os.path.join(outdir, f"results_teleco_{clave}_ft_test.csv")
        results_vt   = os.path.join(outdir, f"results_teleco_{clave}_ft_valtest.csv")
        try:
            df_v = pd.read_csv(results_val)
            df_t = pd.read_csv(results_test)
            df_vt = pd.concat([df_v, df_t], ignore_index=True)
            df_vt.to_csv(results_vt, index=False)
            log(f"  [OK] Combinado val+test: {len(df_vt)} filas -> {results_vt}")
        except Exception as e:
            log(f"  [!] Error combinando val+test: {e}", "WARN")

    return ok_val and ok_test


# =========================================================
# MÉTRICAS POST-FT
# =========================================================

def _metricas_un_split(clave, split_name, results_csv, metricas_csv, nombre_legible):
    """Calcula métricas para un CSV de results y las guarda en otro CSV."""
    if not os.path.exists(results_csv):
        log(f"  [{split_name}] [X] No existe {results_csv}", "WARN")
        return False
    if os.path.exists(metricas_csv):
        log(f"  [{split_name}] [OK] Métricas ya existen, saltando")
        return True

    import pandas as pd
    from tqdm import tqdm
    import evaluate

    df = pd.read_csv(results_csv)
    df_ok = df[~df['LLM_Generated'].astype(str).str.startswith('Error:')].reset_index(drop=True)
    n = len(df_ok)
    log(f"  [{split_name}] Filas válidas: {n}")
    if n == 0: return False

    preds = df_ok['LLM_Generated'].astype(str).tolist()
    refs  = df_ok['Human_Reference'].astype(str).tolist()
    structured = df_ok['Structured_Data'].astype(str).tolist()
    tiempos = df_ok['Time_per_row_seconds'].astype(float).tolist()

    rouge = evaluate.load('rouge')
    rouges = []
    for i in tqdm(range(n), desc=f"ROUGE-L {split_name}"):
        try:
            r = rouge.compute(predictions=[preds[i]], references=[refs[i]])
            rouges.append(round(r['rougeL'], 4))
        except: rouges.append(0.0)

    meteor = evaluate.load('meteor')
    meteors = []
    for i in tqdm(range(n), desc=f"METEOR {split_name}"):
        try:
            r = meteor.compute(predictions=[preds[i]], references=[refs[i]])
            meteors.append(round(r['meteor'], 4))
        except: meteors.append(0.0)

    bleu = evaluate.load('bleu')
    bleus = []
    for i in tqdm(range(n), desc=f"BLEU {split_name}"):
        try:
            bleus.append(round(bleu.compute(predictions=[preds[i]], references=[[refs[i]]])['bleu'], 4))
        except: bleus.append(0.0)

    bert = evaluate.load('bertscore')
    BATCH = 64
    bertscores = []
    for i in tqdm(range(0, n, BATCH), desc=f"BERTScore {split_name}"):
        try:
            r = bert.compute(predictions=preds[i:i+BATCH],
                             references=refs[i:i+BATCH],
                             lang="es", model_type="xlm-roberta-base")
            bertscores.extend([round(x, 4) for x in r['f1']])
        except:
            bertscores.extend([0.0] * len(preds[i:i+BATCH]))

    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        def n_tokens(t): return len(enc.encode(str(t)))
    except:
        def n_tokens(t): return len(str(t)) // 4

    kwh = KWH_POR_TOKEN.get(clave, 4e-8)
    tok_in_list, tok_out_list, costes, co2s = [], [], [], []
    for i in range(n):
        ti = n_tokens(structured[i]) + TOKENS_PROMPT_SISTEMA
        to = n_tokens(preds[i])
        tok_in_list.append(ti)
        tok_out_list.append(to)
        costes.append(0.0)   # local, sin coste
        co2s.append(round((ti+to) * kwh * CO2_POR_KWH, 6))

    df_out = pd.DataFrame({
        'ROUGE_L': rouges, 'METEOR': meteors, 'BLEU': bleus,
        'BERTScore': bertscores, 'Time_seconds': tiempos,
        'Tokens_Input': tok_in_list, 'Tokens_Output': tok_out_list,
        'Coste_USD': costes, 'CO2_gramos': co2s,
    })
    df_out.to_csv(metricas_csv, index=False)
    log(f"  [{split_name}] [OK] Guardado: {metricas_csv}")

    with open(metricas_csv, 'a', encoding='utf-8') as f:
        f.write(f"\n--- RESUMEN {nombre_legible} ({split_name}) ---\n")
        f.write(f"Filas,{n}\n")
        f.write(f"ROUGE-L medio,{round(sum(rouges)/n, 4)}\n")
        f.write(f"METEOR medio,{round(sum(meteors)/n, 4)}\n")
        f.write(f"BLEU medio,{round(sum(bleus)/n, 4)}\n")
        f.write(f"BERTScore medio,{round(sum(bertscores)/n, 4)}\n")
        f.write(f"Tiempo medio (s),{round(sum(tiempos)/n, 4)}\n")
        f.write(f"CO2 total (g),{round(sum(co2s), 4)}\n")

    return True


def metricas_ft(clave, args):
    """Calcula métricas para val, test y valtest (3 CSVs)."""
    log("=" * 60)
    log(f"MÉTRICAS FT: {clave}")
    log("=" * 60)

    outdir = os.path.join(OUTDIR_BASE, "minislm_finetuned", clave)
    nombre = next(n for c, _, n in MINI_SLMS if c == clave)

    ok_total = True
    for split in ["val", "test", "valtest"]:
        results_csv  = os.path.join(outdir, f"results_teleco_{clave}_ft_{split}.csv")
        metricas_csv = os.path.join(outdir, f"metricas_por_fila_{nombre}_{split}.csv")
        ok = _metricas_un_split(clave, split, results_csv, metricas_csv, nombre)
        ok_total = ok_total and ok

    return ok_total


# =========================================================
# MAIN
# =========================================================
def verificar_versiones():
    """Verifica versiones críticas (transformers, peft, etc)."""
    issues = []
    try:
        import transformers
        ver = transformers.__version__
        major, minor = ver.split('.')[:2]
        # Qwen 3 requiere >= 4.51.0
        if int(major) < 4 or (int(major) == 4 and int(minor) < 51):
            issues.append(f"transformers {ver} es vieja. Qwen 3 requiere >= 4.51.0. "
                         f"Ejecuta: pip install -U transformers")
    except Exception as e:
        issues.append(f"No pude verificar transformers: {e}")

    try:
        import peft
        log(f"  peft {peft.__version__}")
    except ImportError:
        issues.append("peft NO instalado. El fine-tuning fallará. Ejecuta: pip install peft")

    try:
        import datasets
        log(f"  datasets {datasets.__version__}")
    except ImportError:
        issues.append("datasets NO instalado. Ejecuta: pip install datasets")

    if issues:
        for i in issues: log(f"  [!] {i}", "WARN")
    return len(issues) == 0


def preparar_nltk():
    """Descarga recursos de NLTK necesarios para METEOR."""
    try:
        import nltk
        for pkg in ["wordnet", "omw-1.4", "punkt", "punkt_tab"]:
            try:
                nltk.data.find(f"corpora/{pkg}")
            except LookupError:
                try:
                    nltk.download(pkg, quiet=True)
                    log(f"  NLTK: descargado '{pkg}'")
                except Exception as e:
                    log(f"  [!] NLTK: no pude descargar '{pkg}': {e}", "WARN")
    except Exception as e:
        log(f"  [!] NLTK no disponible: {e}", "WARN")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",  default="dataset_teleco.csv")
    parser.add_argument("--skip", nargs="*", default=[])
    parser.add_argument("--solo", nargs="*", default=None)
    args = parser.parse_args()

    if not os.path.exists(SPLITS_PATH):
        log(f"[X] No existe {SPLITS_PATH}. Ejecuta primero hacer_splits.py", "ERROR")
        return

    log("#" * 60)
    log("PIPELINE FINE-TUNING - INICIO")
    log("#" * 60)

    log("Verificando versiones de librerías...")
    verificar_versiones()

    log("Preparando recursos NLTK para METEOR...")
    preparar_nltk()

    estado = cargar_estado()

    for clave, hf_id, _ in MINI_SLMS:
        if args.solo and clave not in args.solo: continue
        if clave in args.skip:
            log(f"  Saltando {clave} (--skip)")
            continue

        try:
            ok = finetune_modelo(clave, hf_id, args)
            if ok: estado[clave]["finetuning"] = "completado"; guardar_estado(estado)
        except Exception as e:
            log(f"[X] Fine-tuning {clave} falló: {e}", "ERROR")
            import traceback; log(traceback.format_exc(), "ERROR")
            continue

        try:
            ok = inferencia_modelo_ft(clave, hf_id, args)
            if ok: estado[clave]["inferencia"] = "completado"; guardar_estado(estado)
        except Exception as e:
            log(f"[X] Inferencia FT {clave} falló: {e}", "ERROR")
            import traceback; log(traceback.format_exc(), "ERROR")
            continue

        try:
            ok = metricas_ft(clave, args)
            if ok: estado[clave]["metricas"] = "completado"; guardar_estado(estado)
        except Exception as e:
            log(f"[X] Métricas FT {clave} falló: {e}", "ERROR")
            import traceback; log(traceback.format_exc(), "ERROR")

    log("#" * 60)
    log("PIPELINE FINE-TUNING - FIN")
    log("#" * 60)


if __name__ == "__main__":
    main()
