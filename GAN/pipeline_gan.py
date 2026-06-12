"""
Pipeline de la GAN de generacion de texto para Q&A de Telecomunicaciones.

Orquesta el flujo completo por fases reanudables: genera los splits por subtema,
entrena la GAN (gan_teleco_v3.py) con reanudacion y fallback de batch size,
dibuja las curvas de aprendizaje, ejecuta inferencia sobre val y test cargando el
modelo una sola vez, y calcula metricas (ROUGE-L, METEOR, BLEU, BERTScore), coste
y CO2 por fila. Repite inferencia y metricas sobre KELM (BERTScore en ingles).

El estado se guarda en estado.json para poder reanudar cualquier fase.

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
KELM_CSV        = "kelm_stem_60k.csv"

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
        "inferencia_kelm":    "pendiente",
        "metricas_kelm":      "pendiente",
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
            log("NO hay GPU. La GAN tardará MUCHO en CPU.", "WARN")
            return {"vram_gb": 0, "device": "cpu"}
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        nombre = torch.cuda.get_device_name(0)
        log(f"GPU detectada: {nombre} con {vram:.1f} GB VRAM")
        return {"vram_gb": vram, "device": "cuda"}
    except Exception as e:
        log(f"Error detectando GPU: {e}", "WARN")
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
        log(f"Error leyendo CSV existente {csv_path}: {e}", "WARN")
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
        log(f"No pude truncar {csv_path}: {e}", "WARN")


# FASE 1: Splits
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
            log("Error generando splits", "ERROR")
            return False
    else:
        log(f"Splits ya existen: {SPLITS_PATH}")

    estado["splits_listos"] = True
    guardar_estado(estado)
    return True


# FASE 2: Entrenamiento (con resume completo)
def fase_entrenamiento(args, estado):
    log("=" * 60)
    log("FASE 2: Entrenamiento de la GAN")
    log("=" * 60)

    if estado["entrenamiento"] == "completado":
        log("Entrenamiento ya completado, saltando")
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
            log(f"Entrenamiento completado con batch_size={bs}")
            break
        else:
            log(f"Falló con batch_size={bs}, probando siguiente...", "WARN")

    if ret.returncode != 0:
        log("Entrenamiento falló con todos los batch sizes", "ERROR")
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
        log(f"El subproceso terminó pero no se generó el modelo.", "ERROR")
        log(f"    Esperado: {gen_pt}", "ERROR")
        log(f"    Revisa la salida del subproceso para ver el error real.", "ERROR")
        return False

    estado["entrenamiento"] = "completado"
    guardar_estado(estado)
    log("Entrenamiento completado")
    return True


# FASE 3: Curvas de aprendizaje
def fase_curvas(args, estado):
    log("=" * 60)
    log("FASE 3: Curvas de aprendizaje")
    log("=" * 60)

    if estado["curvas_generadas"]:
        log("Curvas ya generadas, saltando")
        return True

    log_json = os.path.join(OUTDIR_GAN, "gan_training_log.json")
    if not os.path.exists(log_json):
        log(f"No existe {log_json}, saltando curvas", "WARN")
        return True

    os.makedirs(OUTDIR_FIGURAS, exist_ok=True)

    try:
        _generar_curvas_gan(log_json, OUTDIR_FIGURAS)
    except Exception as e:
        log(f"Curvas falló (no crítico): {e}", "WARN")
        return True

    estado["curvas_generadas"] = True
    guardar_estado(estado)
    log("Curvas generadas")
    return True


# Generación de curvas GAN (integrado de plot_gan_curves.py)

def _cargar_log_gan(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return {
        "epochs":   [e["epoch"]          for e in data],
        "lm_train": [e["loss_lm_train"]  for e in data],
        "lm_val":   [e["loss_lm_val"]    for e in data],
        "adv":      [e["loss_adv"]       for e in data],
        "dis":      [e["loss_dis"]       for e in data],
    }


def _generar_curvas_gan(log_path, outdir):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25,
        "grid.linestyle": "--", "grid.linewidth": 0.5,
        "lines.linewidth": 1.8, "legend.frameon": False,
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    C = {"lm_train": "#1f77b4", "lm_val": "#ff7f0e",
         "adv": "#d62728", "dis": "#2ca02c"}

    d = _cargar_log_gan(log_path)
    log(f"  {len(d['epochs'])} épocas registradas")

    # 1. LM loss train vs val
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(d["epochs"], d["lm_train"], color=C["lm_train"], label="Train", marker='o', markersize=4)
    ax.plot(d["epochs"], d["lm_val"],   color=C["lm_val"],   label="Validación", marker='s', markersize=4)
    best_idx = d["lm_val"].index(min(d["lm_val"]))
    ax.axvline(d["epochs"][best_idx], color="grey", linestyle=":", alpha=0.5)
    ax.annotate(f"Mejor val = {d['lm_val'][best_idx]:.3f}\n(época {d['epochs'][best_idx]})",
                xy=(d["epochs"][best_idx], d["lm_val"][best_idx]),
                xytext=(10, 10), textcoords='offset points', fontsize=9, color="grey")
    ax.set_xlabel("Época"); ax.set_ylabel("Pérdida (cross-entropy)")
    ax.set_title("Pérdida del modelo de lenguaje del generador"); ax.legend()
    fig.savefig(os.path.join(outdir, "lm_loss.png")); plt.close(fig)
    log("  lm_loss.png")

    # 2. Dinámica adversarial
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(d["epochs"], d["adv"], color=C["adv"], label="Adversarial (generador)", marker='o', markersize=4)
    ax.plot(d["epochs"], d["dis"], color=C["dis"], label="Discriminador", marker='s', markersize=4)
    ax.axhline(0.693, color="grey", linestyle="--", alpha=0.5, label="Equilibrio teórico (ln 2)")
    ax.set_xlabel("Época"); ax.set_ylabel("Pérdida (BCE)")
    ax.set_title("Dinámica adversarial: generador vs. discriminador"); ax.legend()
    fig.savefig(os.path.join(outdir, "gan_dynamics.png")); plt.close(fig)
    log("  gan_dynamics.png")

    # 3. Todas las pérdidas
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(d["epochs"], d["lm_train"], color=C["lm_train"], label="LM train", marker='o', markersize=3)
    ax.plot(d["epochs"], d["lm_val"],   color=C["lm_val"],   label="LM validación", marker='s', markersize=3)
    ax.plot(d["epochs"], d["adv"],      color=C["adv"],      label="Adversarial", marker='^', markersize=3)
    ax.plot(d["epochs"], d["dis"],      color=C["dis"],      label="Discriminador", marker='v', markersize=3)
    ax.set_xlabel("Época"); ax.set_ylabel("Pérdida")
    ax.set_title("Curvas de aprendizaje completas"); ax.legend(ncol=2)
    fig.savefig(os.path.join(outdir, "all_losses.png")); plt.close(fig)
    log("  all_losses.png")

    # 4. Resumen para la memoria (subplots)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(d["epochs"], d["lm_train"], color=C["lm_train"], label="Train", marker='o', markersize=3)
    ax.plot(d["epochs"], d["lm_val"],   color=C["lm_val"],   label="Validación", marker='s', markersize=3)
    ax.axvline(d["epochs"][best_idx], color="grey", linestyle=":", alpha=0.5)
    ax.set_xlabel("Época"); ax.set_ylabel("Pérdida LM")
    ax.set_title("(a) Pérdida del modelo de lenguaje"); ax.legend()
    ax = axes[1]
    ax.plot(d["epochs"], d["adv"], color=C["adv"], label="Adversarial (G)", marker='o', markersize=3)
    ax.plot(d["epochs"], d["dis"], color=C["dis"], label="Discriminador (D)", marker='s', markersize=3)
    ax.axhline(0.693, color="grey", linestyle="--", alpha=0.5, label="Equilibrio (ln 2)")
    ax.set_xlabel("Época"); ax.set_ylabel("Pérdida BCE")
    ax.set_title("(b) Dinámica adversarial"); ax.legend()
    fig.suptitle("Curvas de aprendizaje - GAN desde cero", fontsize=13, y=1.02)
    fig.savefig(os.path.join(outdir, "gan_summary.png")); plt.close(fig)
    log("  gan_summary.png")

    # Diagnóstico
    n = len(d["epochs"])
    gap = d["lm_val"][-1] - d["lm_train"][-1]
    log(f"  Diagnóstico: {n} épocas, LM train={d['lm_train'][-1]:.4f}, "
        f"LM val={d['lm_val'][-1]:.4f}, gap={gap:+.4f}, "
        f"mejor val={min(d['lm_val']):.4f} (época {d['epochs'][best_idx]})")


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


# FASE 4: Inferencia EFICIENTE sobre VAL + TEST (3.420 + 4.280)
def fase_inferencia(args, estado):
    log("=" * 60)
    log("FASE 4: Inferencia GAN sobre VAL + TEST")
    log("=" * 60)

    if not os.path.exists(SPLITS_PATH):
        log(f"No existe {SPLITS_PATH}", "ERROR")
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
        log(f"No pude importar gan_teleco_v3.py: {e}", "ERROR")
        return False

    # Copiar archivos al cwd
    archivos_modelo = ["gan_generador.pt", "gan_discriminador.pt",
                       "gan_tokenizer.model", "gan_tokenizer.vocab"]
    for fname in archivos_modelo:
        src = os.path.join(OUTDIR_GAN, fname)
        if os.path.exists(src) and not os.path.exists(fname):
            shutil.copy(src, fname)

    if not os.path.exists("gan_generador.pt"):
        log(f"No existe el modelo entrenado", "ERROR")
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
        log(f"  Modelo cargado (época {ckpt.get('epoch','?')}, val_lm={ckpt.get('val_lm','?')})")
    except Exception as e:
        log(f"Error cargando modelo: {e}", "ERROR")
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
            log(f"  Combinado val+test: {len(df_vt)} filas -> {results_csv_path('valtest')}")
        except Exception as e:
            log(f"  Error combinando val+test: {e}", "WARN")

    estado["inferencia"] = "completado"
    guardar_estado(estado)
    log(f"Inferencia completada")
    return True


# FASE 5: Métricas + costes + CO2
def _metricas_un_split_gan(split_name, results_csv, metricas_csv):
    """Calcula métricas para un CSV de la GAN (split val/test/valtest)."""
    if not os.path.exists(results_csv):
        log(f"  [{split_name}] No existe {results_csv}", "WARN")
        return False
    if os.path.exists(metricas_csv):
        log(f"  [{split_name}] Métricas ya existen, saltando")
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
        log(f"  [{split_name}] Cero filas válidas", "ERROR")
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
            log(f"  BERTScore batch falló: {e}", "WARN")
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
    log(f"  [{split_name}] Guardado: {metricas_csv}")

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
        log("Métricas completadas (val + test + valtest)")
    return ok_total


# FASE 6: Inferencia GAN sobre KELM

def _kelm_results_path():
    return os.path.join(OUTDIR_GAN, "results_kelm_GAN.csv")

def _kelm_metricas_path():
    return os.path.join(OUTDIR_GAN, "metricas_por_fila_GAN_kelm.csv")


def fase_inferencia_kelm(args, estado):
    log("=" * 60)
    log("FASE 6: Inferencia GAN sobre KELM")
    log("=" * 60)

    if estado.get("inferencia_kelm") == "completado":
        log("Inferencia KELM ya completada, saltando")
        return True

    kelm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), KELM_CSV)
    if not os.path.exists(kelm_path):
        log(f"No existe {kelm_path}", "ERROR")
        return False

    estado["inferencia_kelm"] = "en_curso"
    guardar_estado(estado)

    try:
        import torch, pandas as pd
        from tqdm import tqdm
        sys.path.insert(0, ".")
        import gan_teleco_v3 as gan_mod
    except Exception as e:
        log(f"No pude importar gan_teleco_v3.py: {e}", "ERROR")
        return False

    # Copiar archivos del modelo al cwd
    archivos_modelo = ["gan_generador.pt", "gan_discriminador.pt",
                       "gan_tokenizer.model", "gan_tokenizer.vocab"]
    for fname in archivos_modelo:
        src = os.path.join(OUTDIR_GAN, fname)
        if os.path.exists(src) and not os.path.exists(fname):
            shutil.copy(src, fname)

    if not os.path.exists("gan_generador.pt"):
        log(f"No existe el modelo entrenado", "ERROR")
        return False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"  Device: {device}")
    log(f"  Cargando modelo entrenado...")

    try:
        sp = gan_mod.cargar_tokenizador(gan_mod.SPM_PREFIX)
        ckpt = torch.load("gan_generador.pt", map_location=device, weights_only=False)
        cfg_modelo = ckpt["cfg"]
        gen = gan_mod.Generador(cfg_modelo).to(device)
        gen.load_state_dict(ckpt["state_dict"])
        gen.eval()
        log(f"  Modelo cargado")
    except Exception as e:
        log(f"Error cargando modelo: {e}", "ERROR")
        return False

    df = pd.read_csv(kelm_path)
    n_total = len(df)
    log(f"  KELM: {n_total} filas")

    csv_path = _kelm_results_path()

    # Resume robusto
    n_done = filas_completadas(csv_path, n_total=n_total)
    if n_done > 0 and n_done < n_total:
        truncar_csv_a(csv_path, n_done)
        log(f"  Reanudando desde fila {n_done}")
    elif n_done >= n_total:
        log(f"  Ya procesado por completo")
        estado["inferencia_kelm"] = "completado"
        guardar_estado(estado)
        return True
    else:
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Structured_Data', 'Human_Reference',
                             'LLM_Generated', 'Time_per_row_seconds'])

    def generar_rapido(texto_input):
        input_ids = sp.encode(texto_input, out_type=int)[:cfg_modelo["max_seq_len"]-2]
        src = torch.tensor(
            [[gan_mod.BOS_ID] + input_ids + [gan_mod.SEP_ID]], device=device)
        with torch.no_grad():
            return gen.generar(src, sp, max_new_tokens=150,
                              temperature=0.8, top_k=50)

    is_gpu = (device == "cuda")
    try:
        with open(csv_path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            for idx in tqdm(range(n_done, n_total), desc="GAN inferencia KELM"):
                row = df.iloc[idx]
                structured = str(row['Structured_Data']).replace('\n', ' ').replace('\r', ' ').strip()
                reference  = str(row['Human_Reference']).replace('\n', ' ').replace('\r', ' ').strip()

                if is_gpu:
                    torch.cuda.synchronize()
                t0 = time.time()
                try:
                    generado = generar_rapido(structured)
                    if not generado.strip():
                        generado = "<vacio>"
                    generado = generado.replace('\n', ' ').replace('\r', ' ').strip()
                except Exception as e:
                    generado = f"Error: {str(e)[:80]}"
                if is_gpu:
                    torch.cuda.synchronize()
                elapsed = time.time() - t0

                writer.writerow([structured, reference, generado, round(elapsed, 4)])
                f.flush()
    finally:
        for fname in archivos_modelo:
            if os.path.exists(fname):
                try: os.remove(fname)
                except: pass

    estado["inferencia_kelm"] = "completado"
    guardar_estado(estado)
    log(f"Inferencia KELM completada: {csv_path}")
    return True


# FASE 7: Métricas KELM (BERTScore en inglés)

def fase_metricas_kelm(args, estado):
    log("=" * 60)
    log("FASE 7: Métricas KELM")
    log("=" * 60)

    if estado.get("metricas_kelm") == "completado":
        log("Métricas KELM ya completadas, saltando")
        return True

    results_csv  = _kelm_results_path()
    metricas_csv = _kelm_metricas_path()

    if not os.path.exists(results_csv):
        log(f"No existe {results_csv}", "WARN")
        return False
    if os.path.exists(metricas_csv):
        log("Métricas KELM ya existen, saltando")
        estado["metricas_kelm"] = "completado"
        guardar_estado(estado)
        return True

    import pandas as pd
    from tqdm import tqdm
    import evaluate

    df = pd.read_csv(results_csv)
    df_ok = df[~df['LLM_Generated'].astype(str).str.startswith('Error:')].reset_index(drop=True)
    n_ok = len(df_ok)
    log(f"  Filas válidas: {n_ok}")
    if n_ok == 0:
        return False

    preds = df_ok['LLM_Generated'].astype(str).tolist()
    refs  = df_ok['Human_Reference'].astype(str).tolist()
    structured = df_ok['Structured_Data'].astype(str).tolist()
    tiempos = df_ok['Time_per_row_seconds'].astype(float).tolist()

    log("  Calculando ROUGE-L...")
    rouge = evaluate.load('rouge')
    rouges = []
    for i in tqdm(range(n_ok), desc="ROUGE-L KELM"):
        try:
            r = rouge.compute(predictions=[preds[i]], references=[refs[i]])
            rouges.append(round(r['rougeL'], 4))
        except: rouges.append(0.0)

    log("  Calculando METEOR...")
    meteor = evaluate.load('meteor')
    meteors = []
    for i in tqdm(range(n_ok), desc="METEOR KELM"):
        try:
            r = meteor.compute(predictions=[preds[i]], references=[refs[i]])
            meteors.append(round(r['meteor'], 4))
        except: meteors.append(0.0)

    log("  Calculando BLEU...")
    bleu = evaluate.load('bleu')
    bleus = []
    for i in tqdm(range(n_ok), desc="BLEU KELM"):
        try:
            r = bleu.compute(predictions=[preds[i]], references=[[refs[i]]])
            bleus.append(round(r['bleu'], 4))
        except: bleus.append(0.0)

    log("  Calculando BERTScore (distilbert-base-uncased, inglés)...")
    bert = evaluate.load('bertscore')
    BATCH = 64
    bertscores = []
    for i in tqdm(range(0, n_ok, BATCH), desc="BERTScore KELM"):
        try:
            r = bert.compute(predictions=preds[i:i+BATCH],
                             references=refs[i:i+BATCH],
                             lang="en", model_type="distilbert-base-uncased")
            bertscores.extend([round(x, 4) for x in r['f1']])
        except:
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
        'ROUGE_L': rouges, 'METEOR': meteors, 'BLEU': bleus,
        'BERTScore': bertscores, 'Time_seconds': tiempos,
        'Tokens_Input': tok_in, 'Tokens_Output': tok_out,
        'Coste_USD': coste, 'CO2_gramos': co2,
    })
    df_out.to_csv(metricas_csv, index=False)

    with open(metricas_csv, 'a', encoding='utf-8') as f:
        f.write(f"\n--- RESUMEN GAN (KELM) ---\n")
        f.write(f"Filas,{n_ok}\n")
        f.write(f"ROUGE-L medio,{round(sum(rouges)/n_ok, 4)}\n")
        f.write(f"METEOR medio,{round(sum(meteors)/n_ok, 4)}\n")
        f.write(f"BLEU medio,{round(sum(bleus)/n_ok, 4)}\n")
        f.write(f"BERTScore medio,{round(sum(bertscores)/n_ok, 4)}\n")
        f.write(f"Tiempo medio (s),{round(sum(tiempos)/n_ok, 4)}\n")
        f.write(f"CO2 total (g),{round(sum(co2), 4)}\n")

    estado["metricas_kelm"] = "completado"
    guardar_estado(estado)
    log(f"Métricas KELM guardadas: {metricas_csv}")
    return True


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
                    log(f"  NLTK: no pude descargar '{pkg}': {e}", "WARN")
    except Exception as e:
        log(f"  NLTK no disponible: {e}", "WARN")


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
        ("Inferencia KELM",   lambda: fase_inferencia_kelm(args, estado)),
        ("Métricas KELM",     lambda: fase_metricas_kelm(args, estado)),
    ]

    for nombre, func in fases:
        try:
            ok = func()
            if not ok:
                log(f"Fase '{nombre}' falló. Continuando con la siguiente...", "WARN")
        except Exception as e:
            log(f"Excepción en '{nombre}': {e}", "ERROR")
            import traceback
            log(traceback.format_exc(), "ERROR")

    log("#" * 60)
    log("PIPELINE GAN - FIN")
    log("#" * 60)


if __name__ == "__main__":
    main()