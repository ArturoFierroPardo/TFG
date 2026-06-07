"""
Pipeline GAN - versión corregida.

CAMBIOS RESPECTO A LA VERSIÓN ANTERIOR:
  - Pasa --splits-json al script gan_teleco_v3.py (cero data leakage)
  - Inferencia EFICIENTE: carga el modelo UNA vez (no por fila)
  - Evalúa solo sobre TEST SET (4.280 pares) - comparable con Mini-SLMs FT
  - Resume robusto: usa pd.read_csv y descarta última fila por seguridad
  - Usa el resume del entrenamiento (gan_train_state.pt)

Pasos:
  1. Detecta GPU
  2. Verifica/crea splits por subtema
  3. Entrena GAN modo final (con resume desde checkpoint completo)
  4. Genera curvas de aprendizaje
  5. Inferencia sobre TEST SET (cargando modelo UNA vez)
  6. Calcula métricas por fila + costes + CO2

Uso:
    python pipeline_gan.py
    python pipeline_gan.py --epochs 100
"""

import sys
# Forzar UTF-8 en stdout/stderr (necesario en Windows con cp1252)
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


import os, sys, json, csv, time, shutil, argparse, subprocess, datetime as dt

OUTDIR_BASE     = "resultados_arturo"
OUTDIR_GAN      = os.path.join(OUTDIR_BASE, "gan")
OUTDIR_FIGURAS  = os.path.join(OUTDIR_GAN,  "figuras")
SPLITS_PATH     = os.path.join(OUTDIR_BASE, "splits", "splits_por_subtema.json")
ESTADO_PATH     = os.path.join(OUTDIR_GAN, "estado.json")
LOG_PATH        = os.path.join(OUTDIR_GAN, "pipeline_gan.log")

# Helpers para nombres de archivos por split (val/test/valtest)
def results_csv_path(split):  return os.path.join(OUTDIR_GAN, f"results_teleco_GAN_{split}.csv")
def metricas_csv_path(split): return os.path.join(OUTDIR_GAN, f"metricas_por_fila_GAN_{split}.csv")


def log(msg, level="INFO"):
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def cargar_estado():
    """Carga estado anterior. Si faltan claves, las añade con default."""
    estado_default = {
        "splits_listos":      False,
        "entrenamiento":      "pendiente",
        "curvas_generadas":   False,
        "inferencia":         "pendiente",
        "metricas":           "pendiente",
    }
    if not os.path.exists(ESTADO_PATH):
        return estado_default
    try:
        with open(ESTADO_PATH, encoding="utf-8") as f:
            estado = json.load(f)
    except Exception:
        return estado_default
    # Añadir claves nuevas si faltan
    for k, v in estado_default.items():
        if k not in estado:
            estado[k] = v
    return estado


def guardar_estado(estado):
    os.makedirs(os.path.dirname(ESTADO_PATH), exist_ok=True)
    with open(ESTADO_PATH, "w", encoding="utf-8") as f:
        json.dump(estado, f, indent=2)


def detectar_gpu():
    try:
        import torch
        if not torch.cuda.is_available():
            log("[!] NO hay GPU. La GAN tardará MUCHO en CPU.", "WARN")
            return {"vram_gb": 0, "device": "cpu"}
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        nombre = torch.cuda.get_device_name(0)
        log(f"GPU detectada: {nombre} con {vram:.1f} GB VRAM")
        return {"vram_gb": vram, "device": "cuda"}
    except Exception as e:
        log(f"[!] Error detectando GPU: {e}", "WARN")
        return {"vram_gb": 0, "device": "cpu"}


def filas_completadas(csv_path, n_total=None):
    """
    Cuenta filas completadas robustamente con pd.read_csv.
    - Si len_actual >= n_total: devuelve n_total (caso ya completado)
    - Si len_actual > 0: devuelve len_actual - 1 (descarta última por seguridad)
    - Si len_actual == 0: devuelve 0
    """
    if not os.path.exists(csv_path):
        return 0
    try:
        import pandas as pd
        df = pd.read_csv(csv_path, encoding='utf-8')
        n = len(df)
        if n_total is not None and n >= n_total:
            return n_total
        return max(0, n - 1) if n > 0 else 0
    except Exception as e:
        log(f"[!] Error leyendo CSV existente {csv_path}: {e}", "WARN")
        return 0


def truncar_csv_a(csv_path, n_filas):
    if not os.path.exists(csv_path) or n_filas == 0:
        return
    try:
        import pandas as pd
        df = pd.read_csv(csv_path, encoding='utf-8')
        if len(df) > n_filas:
            df.head(n_filas).to_csv(csv_path, index=False)
    except Exception as e:
        log(f"[!] No pude truncar {csv_path}: {e}", "WARN")


# =========================================================
# FASE 1: Splits
# =========================================================
def fase_splits(args, estado):
    log("=" * 60)
    log("FASE 1: Splits por subtema")
    log("=" * 60)

    if not os.path.exists(SPLITS_PATH):
        log("Splits no existen. Ejecutando hacer_splits.py...")
        ret = subprocess.run([sys.executable, "hacer_splits.py",
                              "--csv", args.csv,
                              "--outdir", os.path.join(OUTDIR_BASE, "splits")])
        if ret.returncode != 0:
            log("[X] Error generando splits", "ERROR")
            return False
    else:
        log(f"[OK] Splits ya existen: {SPLITS_PATH}")

    estado["splits_listos"] = True
    guardar_estado(estado)
    return True


# =========================================================
# FASE 2: Entrenamiento (con resume completo)
# =========================================================
def fase_entrenamiento(args, estado):
    log("=" * 60)
    log("FASE 2: Entrenamiento de la GAN")
    log("=" * 60)

    if estado["entrenamiento"] == "completado":
        log("[OK] Entrenamiento ya completado, saltando")
        return True

    estado["entrenamiento"] = "en_curso"
    guardar_estado(estado)

    # Si existe el checkpoint de entrenamiento en gan/, copiarlo al cwd para que el script lo use
    ckpt_train_dst = "gan_train_state.pt"
    ckpt_train_src = os.path.join(OUTDIR_GAN, "gan_train_state.pt")
    if os.path.exists(ckpt_train_src) and not os.path.exists(ckpt_train_dst):
        shutil.copy(ckpt_train_src, ckpt_train_dst)
        log(f"  Restaurando checkpoint de entrenamiento desde {ckpt_train_src}")

    # Lo mismo con tokenizador (necesario si se reanuda)
    for fname in ["gan_tokenizer.model", "gan_tokenizer.vocab"]:
        src = os.path.join(OUTDIR_GAN, fname)
        if os.path.exists(src) and not os.path.exists(fname):
            shutil.copy(src, fname)

    # Intentar con batch_size decreciente (OOM fallback automático)
    batch_sizes = [16, 8, 4]
    ret = None
    for bs in batch_sizes:
        cmd = [sys.executable, "gan_teleco_v3.py",
               "--modo",          args.modo,
               "--epochs",        str(args.epochs),
               "--csv",           args.csv,
               "--splits-json",   SPLITS_PATH,
               "--asignatura",    "all",
               "--batch-size",    str(bs)]
        if args.vocab_size:
            cmd.extend(["--vocab-size", str(args.vocab_size)])
        log(f"Intentando con batch_size={bs}: {' '.join(cmd)}")

        ret = subprocess.run(cmd)

        if ret.returncode == 0:
            log(f"[OK] Entrenamiento completado con batch_size={bs}")
            break
        else:
            log(f"[!] Falló con batch_size={bs}, probando siguiente...", "WARN")

    if ret.returncode != 0:
        log("[X] Entrenamiento falló con todos los batch sizes", "ERROR")
        log("  Al relanzar, reanudará desde gan_train_state.pt", "INFO")
        # Mover lo que haya conseguido al directorio gan/ para conservarlo
        for fname in ["gan_train_state.pt", "gan_training_log.json",
                      "gan_tokenizer.model", "gan_tokenizer.vocab"]:
            if os.path.exists(fname):
                dst = os.path.join(OUTDIR_GAN, fname)
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(fname, dst)
        return False

    # Mover archivos generados al directorio gan/
    archivos_a_mover = [
        "gan_generador.pt", "gan_discriminador.pt",
        "gan_tokenizer.model", "gan_tokenizer.vocab",
        "gan_training_log.json", "gan_splits.json",
        "gan_train_state.pt",
    ]
    os.makedirs(OUTDIR_GAN, exist_ok=True)
    for fname in archivos_a_mover:
        if os.path.exists(fname):
            dst = os.path.join(OUTDIR_GAN, fname)
            if os.path.exists(dst):
                os.remove(dst)
            shutil.move(fname, dst)
            log(f"  -> Movido: {fname}")

    # Verificar que realmente se entrenó: debe existir el generador
    gen_pt = os.path.join(OUTDIR_GAN, "gan_generador.pt")
    if not os.path.exists(gen_pt):
        log(f"[X] El subproceso terminó pero no se generó el modelo.", "ERROR")
        log(f"    Esperado: {gen_pt}", "ERROR")
        log(f"    Revisa la salida del subproceso para ver el error real.", "ERROR")
        return False

    estado["entrenamiento"] = "completado"
    guardar_estado(estado)
    log("[OK] Entrenamiento completado")
    return True


# =========================================================
# FASE 3: Curvas de aprendizaje
# =========================================================
def fase_curvas(args, estado):
    log("=" * 60)
    log("FASE 3: Curvas de aprendizaje")
    log("=" * 60)

    if estado["curvas_generadas"]:
        log("[OK] Curvas ya generadas, saltando")
        return True

    log_json = os.path.join(OUTDIR_GAN, "gan_training_log.json")
    if not os.path.exists(log_json):
        log(f"[!] No existe {log_json}, saltando curvas", "WARN")
        return True

    os.makedirs(OUTDIR_FIGURAS, exist_ok=True)
    cmd = [sys.executable, "plot_gan_curves.py",
           "--log",    log_json,
           "--outdir", OUTDIR_FIGURAS]
    log(f"Ejecutando: {' '.join(cmd)}")
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        log("[!] Curvas falló (no crítico)", "WARN")
        return True

    estado["curvas_generadas"] = True
    guardar_estado(estado)
    log("[OK] Curvas generadas")
    return True


def _inferencia_un_split(split_name, split_ids, df, cfg_modelo, gen, sp, gan_mod, device):
    """Genera inferencia de la GAN sobre un split. Devuelve True si OK."""
    import torch
    from tqdm import tqdm

    df_split = df[df['id'].isin(split_ids)].reset_index(drop=True)
    n = len(df_split)
    log(f"  [{split_name}] {n} pares a procesar")

    csv_path = results_csv_path(split_name)
    n_done = filas_completadas(csv_path, n_total=n)
    if n_done > 0 and n_done < n:
        truncar_csv_a(csv_path, n_done)
        log(f"  [{split_name}] Reanudando desde fila {n_done}")
    elif n_done >= n:
        log(f"  [{split_name}] Ya estaba todo procesado: {n_done}/{n}")
        return True
    else:
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Structured_Data', 'Human_Reference',
                             'LLM_Generated', 'Time_per_row_seconds'])
        log(f"  [{split_name}] Empezando desde 0")

    if n_done >= n:
        return True

    def generar_rapido(pregunta):
        preg_ids = sp.encode(pregunta, out_type=int)[:cfg_modelo["max_seq_len"]-2]
        src = torch.tensor(
            [[gan_mod.BOS_ID] + preg_ids + [gan_mod.SEP_ID]], device=device)
        with torch.no_grad():
            return gen.generar(src, sp, max_new_tokens=150,
                              temperature=0.8, top_k=50)

    is_gpu = (device == "cuda")
    with open(csv_path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for idx in tqdm(range(n_done, n), desc=f"GAN inferencia {split_name}"):
            row = df_split.iloc[idx]
            pregunta  = str(row['pregunta'])
            respuesta = str(row['respuesta'])

            if is_gpu:
                torch.cuda.synchronize()
            t0 = time.time()
            try:
                generado = generar_rapido(pregunta)
                if not generado.strip():
                    generado = "<vacio>"
                # Limpiar saltos de línea para no romper el CSV
                generado = generado.replace('\n', ' ').replace('\r', ' ').strip()
            except Exception as e:
                generado = f"Error: {str(e)[:80]}"
            if is_gpu:
                torch.cuda.synchronize()
            elapsed = time.time() - t0

            # También limpiar la pregunta y respuesta de saltos de línea
            preg_limpia = pregunta.replace('\n', ' ').replace('\r', ' ').strip()
            resp_limpia = respuesta.replace('\n', ' ').replace('\r', ' ').strip()
            writer.writerow([preg_limpia, resp_limpia, generado, round(elapsed, 4)])
            f.flush()
    return True


# =========================================================
# FASE 4: Inferencia EFICIENTE sobre VAL + TEST (3.420 + 4.280)
# =========================================================
def fase_inferencia(args, estado):
    log("=" * 60)
    log("FASE 4: Inferencia GAN sobre VAL + TEST")
    log("=" * 60)

    if not os.path.exists(SPLITS_PATH):
        log(f"[X] No existe {SPLITS_PATH}", "ERROR")
        return False
    with open(SPLITS_PATH, encoding='utf-8') as f:
        splits = json.load(f)
    val_ids  = set(splits["val_ids"])
    test_ids = set(splits["test_ids"])
    log(f"  Val: {len(val_ids)} | Test: {len(test_ids)}")

    estado["inferencia"] = "en_curso"
    guardar_estado(estado)

    try:
        import torch, pandas as pd
        from tqdm import tqdm
        sys.path.insert(0, ".")
        import gan_teleco_v3 as gan_mod
    except Exception as e:
        log(f"[X] No pude importar gan_teleco_v3.py: {e}", "ERROR")
        return False

    # Copiar archivos al cwd
    archivos_modelo = ["gan_generador.pt", "gan_discriminador.pt",
                       "gan_tokenizer.model", "gan_tokenizer.vocab"]
    for fname in archivos_modelo:
        src = os.path.join(OUTDIR_GAN, fname)
        if os.path.exists(src) and not os.path.exists(fname):
            shutil.copy(src, fname)

    if not os.path.exists("gan_generador.pt"):
        log(f"[X] No existe el modelo entrenado", "ERROR")
        return False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"  Device: {device}")
    log(f"  Cargando modelo entrenado (UNA SOLA VEZ)...")

    try:
        sp = gan_mod.cargar_tokenizador(gan_mod.SPM_PREFIX)
        ckpt = torch.load("gan_generador.pt", map_location=device, weights_only=False)
        cfg_modelo = ckpt["cfg"]
        gen = gan_mod.Generador(cfg_modelo).to(device)
        gen.load_state_dict(ckpt["state_dict"])
        gen.eval()
        log(f"  [OK] Modelo cargado (época {ckpt.get('epoch','?')}, val_lm={ckpt.get('val_lm','?')})")
    except Exception as e:
        log(f"[X] Error cargando modelo: {e}", "ERROR")
        return False

    df = pd.read_csv(args.csv)

    try:
        # 1) Inferencia VAL
        ok_val = _inferencia_un_split("val", val_ids, df, cfg_modelo, gen, sp, gan_mod, device)
        # 2) Inferencia TEST
        ok_test = _inferencia_un_split("test", test_ids, df, cfg_modelo, gen, sp, gan_mod, device)
    finally:
        for fname in archivos_modelo:
            if os.path.exists(fname):
                try: os.remove(fname)
                except: pass

    # 3) Combinar val+test en CSV único
    if ok_val and ok_test:
        try:
            df_v  = pd.read_csv(results_csv_path("val"))
            df_t  = pd.read_csv(results_csv_path("test"))
            df_vt = pd.concat([df_v, df_t], ignore_index=True)
            df_vt.to_csv(results_csv_path("valtest"), index=False)
            log(f"  [OK] Combinado val+test: {len(df_vt)} filas -> {results_csv_path('valtest')}")
        except Exception as e:
            log(f"  [!] Error combinando val+test: {e}", "WARN")

    estado["inferencia"] = "completado"
    guardar_estado(estado)
    log(f"[OK] Inferencia completada")
    return True


# =========================================================
# FASE 5: Métricas + costes + CO2
# =========================================================
def _metricas_un_split_gan(split_name, results_csv, metricas_csv):
    """Calcula métricas para un CSV de la GAN (split val/test/valtest)."""
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
    n_total = len(df)
    n_ok    = len(df_ok)
    log(f"  [{split_name}] Filas totales: {n_total}, sin errores: {n_ok}")
    if n_ok == 0:
        log(f"  [{split_name}] [X] Cero filas válidas", "ERROR")
        return False

    preds = df_ok['LLM_Generated'].astype(str).tolist()
    refs  = df_ok['Human_Reference'].astype(str).tolist()
    structured = df_ok['Structured_Data'].astype(str).tolist()
    tiempos = df_ok['Time_per_row_seconds'].astype(float).tolist()

    log(f"  [{split_name}] Calculando ROUGE-L...")
    rouge = evaluate.load('rouge')
    rouges = []
    for i in tqdm(range(n_ok), desc=f"ROUGE-L {split_name}"):
        try:
            r = rouge.compute(predictions=[preds[i]], references=[refs[i]])
            rouges.append(round(r['rougeL'], 4))
        except: rouges.append(0.0)

    log(f"  [{split_name}] Calculando METEOR...")
    meteor = evaluate.load('meteor')
    meteors = []
    for i in tqdm(range(n_ok), desc=f"METEOR {split_name}"):
        try:
            r = meteor.compute(predictions=[preds[i]], references=[refs[i]])
            meteors.append(round(r['meteor'], 4))
        except: meteors.append(0.0)

    log(f"  [{split_name}] Calculando BLEU...")
    bleu = evaluate.load('bleu')
    bleus = []
    for i in tqdm(range(n_ok), desc=f"BLEU {split_name}"):
        try:
            r = bleu.compute(predictions=[preds[i]], references=[[refs[i]]])
            bleus.append(round(r['bleu'], 4))
        except: bleus.append(0.0)

    log(f"  [{split_name}] Calculando BERTScore (xlm-roberta-base, español)...")
    bert = evaluate.load('bertscore')
    BATCH = 64
    bertscores = []
    for i in tqdm(range(0, n_ok, BATCH), desc=f"BERTScore {split_name}"):
        try:
            r = bert.compute(predictions=preds[i:i+BATCH],
                             references=refs[i:i+BATCH],
                             lang="es", model_type="xlm-roberta-base")
            bertscores.extend([round(x, 4) for x in r['f1']])
        except Exception as e:
            log(f"  [!] BERTScore batch falló: {e}", "WARN")
            bertscores.extend([0.0] * len(preds[i:i+BATCH]))

    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        def n_tokens(t): return len(enc.encode(str(t)))
    except:
        def n_tokens(t): return len(str(t)) // 4

    GAN_KWH_POR_TOKEN = 1e-8
    CO2_POR_KWH = 475
    tok_in  = [n_tokens(s) for s in structured]
    tok_out = [n_tokens(p) for p in preds]
    coste   = [0.0] * n_ok
    co2     = [round((ti+to) * GAN_KWH_POR_TOKEN * CO2_POR_KWH, 6)
               for ti, to in zip(tok_in, tok_out)]

    df_out = pd.DataFrame({
        'ROUGE_L':       rouges,
        'METEOR':        meteors,
        'BLEU':          bleus,
        'BERTScore':     bertscores,
        'Time_seconds':  tiempos,
        'Tokens_Input':  tok_in,
        'Tokens_Output': tok_out,
        'Coste_USD':     coste,
        'CO2_gramos':    co2,
    })
    df_out.to_csv(metricas_csv, index=False)
    log(f"  [{split_name}] [OK] Guardado: {metricas_csv}")

    with open(metricas_csv, 'a', encoding='utf-8') as f:
        f.write(f"\n--- RESUMEN GAN ({split_name}) ---\n")
        f.write(f"Filas procesadas,{n_ok}\n")
        f.write(f"ROUGE-L medio,{round(sum(rouges)/n_ok, 4)}\n")
        f.write(f"METEOR medio,{round(sum(meteors)/n_ok, 4)}\n")
        f.write(f"BLEU medio,{round(sum(bleus)/n_ok, 4)}\n")
        f.write(f"BERTScore medio,{round(sum(bertscores)/n_ok, 4)}\n")
        f.write(f"Tiempo medio (s),{round(sum(tiempos)/n_ok, 4)}\n")
        f.write(f"Coste total USD,{round(sum(coste), 4)}\n")
        f.write(f"CO2 total (g),{round(sum(co2), 4)}\n")

    return True


def fase_metricas(args, estado):
    log("=" * 60)
    log("FASE 5: Métricas por fila + costes + CO2 (val + test + valtest)")
    log("=" * 60)

    estado["metricas"] = "en_curso"
    guardar_estado(estado)

    ok_total = True
    for split in ["val", "test", "valtest"]:
        rcsv = results_csv_path(split)
        mcsv = metricas_csv_path(split)
        ok = _metricas_un_split_gan(split, rcsv, mcsv)
        ok_total = ok_total and ok

    if ok_total:
        estado["metricas"] = "completado"
        guardar_estado(estado)
        log("[OK] Métricas completadas (val + test + valtest)")
    return ok_total


# =========================================================
# MAIN
# =========================================================
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
    parser.add_argument("--csv",    default="dataset_teleco.csv")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--modo",   default="final",
                        choices=["prueba", "final"],
                        help="Modo de entrenamiento (default: final)")
    parser.add_argument("--vocab-size", type=int, default=None,
                        help="Override vocab_size (útil para smoke tests)")
    args = parser.parse_args()

    os.makedirs(OUTDIR_GAN, exist_ok=True)
    log("#" * 60)
    log("PIPELINE GAN - INICIO")
    log("#" * 60)

    detectar_gpu()
    log("Preparando recursos NLTK para METEOR...")
    preparar_nltk()
    estado = cargar_estado()
    log(f"Estado actual: {json.dumps(estado, indent=2)}")

    fases = [
        ("Splits",            lambda: fase_splits(args, estado)),
        ("Entrenamiento",     lambda: fase_entrenamiento(args, estado)),
        ("Curvas",            lambda: fase_curvas(args, estado)),
        ("Inferencia",        lambda: fase_inferencia(args, estado)),
        ("Métricas",          lambda: fase_metricas(args, estado)),
    ]

    for nombre, func in fases:
        try:
            ok = func()
            if not ok:
                log(f"[X] Fase '{nombre}' falló. Continuando con la siguiente...", "WARN")
        except Exception as e:
            log(f"[X] Excepción en '{nombre}': {e}", "ERROR")
            import traceback
            log(traceback.format_exc(), "ERROR")

    log("#" * 60)
    log("PIPELINE GAN - FIN")
    log("#" * 60)


if __name__ == "__main__":
    main()
