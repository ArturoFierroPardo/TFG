"""
GAN de generacion de texto para Q&A de Telecomunicaciones.
Arquitectura: SeqGAN condicional (encoder-decoder Transformer) con Gumbel-Softmax.

Tamanos aproximados:
    prueba (CPU):  generador ~23M params,  discriminador ~3M params
    final  (GPU):  generador ~430M params, discriminador ~100M params

Lee el dataset desde dataset_teleco.csv. El split por subtema se pasa con
--splits-json (recomendado, evita data leakage).

Requisitos:
    pip install sentencepiece tqdm torch nltk rouge-score bert-score tiktoken

Uso:
    python gan_teleco_v3.py --modo final --epochs 100 --splits-json splits_por_subtema.json
    python gan_teleco_v3.py --generar --pregunta "Que es una funcion recursiva?"
    python gan_teleco_v3.py --evaluar --splits-json splits_por_subtema.json
"""

import sys
# Forzar UTF-8 en stdout/stderr (necesario en Windows con cp1252)
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


import os, json, math, argparse, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import csv
import sentencepiece as spm
from tqdm import tqdm


DATASET_CSV = "dataset_teleco.csv"
SPM_PREFIX  = "gan_tokenizer"
GEN_CKPT    = "gan_generador.pt"
DIS_CKPT    = "gan_discriminador.pt"
LOG_FILE    = "gan_training_log.json"
EVAL_FILE   = "gan_eval_results.json"
SPLITS_FILE = "gan_splits.json"

# Tokens especiales
PAD_ID = 0   # <pad>
BOS_ID = 1   # <s>
EOS_ID = 2   # </s>
UNK_ID = 3   # <unk>
SEP_ID = 4   # <SEP>  (control_symbol, id asignado tras los specials)

# Reproducibilidad
SEED = 42

# Configuración por modo
CONFIGS = {
    "prueba": {
        # Generador ~21M params
        "vocab_size":     4000,
        "d_model":         512,
        "nhead":             8,
        "num_layers":        4,
        "dim_feedforward":  1024,
        "dropout":          0.1,
        "max_seq_len":       128,
        # Discriminador ~5M params
        "d_model_dis":     256,
        "nhead_dis":         4,
        "num_layers_dis":    3,
        "dim_feedforward_dis": 512,
        # Entrenamiento
        "batch_size":        8,
        "lr_gen":         1e-4,
        "lr_dis":         1e-4,
        "gumbel_tau":       1.0,
        "device":         "cpu",
    },
    "final": {
        # Generador ~430M params  (d_model=1024, L=14, ff=4096)
        "vocab_size":    16000,
        "d_model":        1024,
        "nhead":            16,
        "num_layers":       14,
        "dim_feedforward": 4096,
        "dropout":          0.1,
        "max_seq_len":       256,
        # Discriminador ~100M params  (d_model=768, L=12, ff=3072)
        "d_model_dis":     768,
        "nhead_dis":        12,
        "num_layers_dis":   12,
        "dim_feedforward_dis": 3072,
        # Entrenamiento
        "batch_size":       16,
        "lr_gen":         5e-5,
        "lr_dis":         5e-5,
        "gumbel_tau":       0.8,
        "device":        "cuda",
    },
}


def leer_dataset(csv_path, asignatura_filtro=None):
    """Lee dataset_teleco.csv. Devuelve lista de (id, pregunta, respuesta)."""
    print(f"Cargando dataset: {csv_path}...")

    pares = []
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if asignatura_filtro and row['asignatura'] != asignatura_filtro:
                continue
            pregunta  = (row['pregunta']  or "").strip()
            respuesta = (row['respuesta'] or "").strip()
            if pregunta and respuesta:
                pares.append((int(row['id']), pregunta, respuesta))

    print(f"{len(pares)} pares cargados"
          + (f" de '{asignatura_filtro}'" if asignatura_filtro else ""))
    if not pares and asignatura_filtro:
        print(f"  No se encontraron pares para '{asignatura_filtro}'.")
        print(f"     Comprueba el nombre exacto (con tildes y mayúsculas).")
    return pares

def split_dataset_externo(pares, splits_json_path):
    """
    Split usando el JSON externo generado por hacer_splits.py (split por subtema).
    SIN data leakage: cada subtema solo aparece en un split.

    Args:
        pares: lista de (id, pregunta, respuesta) de leer_dataset.
        splits_json_path: ruta al JSON con {train_ids, val_ids, test_ids}.

    Returns:
        train, val, test: listas de (pregunta, respuesta) [sin id, para QADataset].
    """
    print(f"Cargando split por subtema: {splits_json_path}")
    with open(splits_json_path, encoding='utf-8') as f:
        splits = json.load(f)

    train_ids = set(splits["train_ids"])
    val_ids   = set(splits["val_ids"])
    test_ids  = set(splits["test_ids"])

    train, val, test = [], [], []
    for id_, preg, resp in pares:
        if id_ in train_ids:
            train.append((preg, resp))
        elif id_ in val_ids:
            val.append((preg, resp))
        elif id_ in test_ids:
            test.append((preg, resp))

    print(f"Split por subtema -> train={len(train)} | val={len(val)} | test={len(test)}")
    return train, val, test


def split_dataset(pares, train_ratio=0.7, val_ratio=0.15, seed=SEED):
    """
    Split aleatorio 70/15/15 LEGACY. Solo se usa si NO hay JSON de split por subtema.
    NO recomendado para experimento final (puede haber data leakage).

    Acepta pares en dos formatos:
      - (id, pregunta, respuesta)  -> nuevo formato
      - (pregunta, respuesta)      -> formato antiguo
    """
    # Detectar formato y normalizar
    if pares and len(pares[0]) == 3:
        pares_norm = [(p, r) for _id, p, r in pares]
    else:
        pares_norm = pares

    n = len(pares_norm)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    train = [pares_norm[i] for i in train_idx]
    val   = [pares_norm[i] for i in val_idx]
    test  = [pares_norm[i] for i in test_idx]

    # Guardar splits para reproducibilidad
    with open(SPLITS_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            "seed": seed,
            "n_total": n,
            "train_idx": train_idx,
            "val_idx":   val_idx,
            "test_idx":  test_idx,
        }, f, indent=2)

    print(f"Split aleatorio 70/15/15 (LEGACY) -> train={len(train)} | val={len(val)} | test={len(test)}")
    print(f"   ATENCIÓN: posible data leakage. Usa --splits-json para split por subtema.")
    return train, val, test


def entrenar_tokenizador(pares, vocab_size, prefix):
    corpus_file = f"{prefix}_corpus.txt"
    print(f"\nEntrenando tokenizador BPE (vocab={vocab_size})...")

    with open(corpus_file, 'w', encoding='utf-8') as f:
        for pregunta, respuesta in pares:
            f.write(pregunta + "\n")
            f.write(respuesta + "\n")

    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=prefix,
        vocab_size=vocab_size,
        pad_id=PAD_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        unk_id=UNK_ID,
        control_symbols=["<SEP>"],
        character_coverage=0.9995,
        model_type="bpe",
    )
    os.remove(corpus_file)
    print(f"Tokenizador guardado: {prefix}.model")

def cargar_tokenizador(prefix):
    sp = spm.SentencePieceProcessor()
    sp.load(f"{prefix}.model")
    sep_id = sp.piece_to_id("<SEP>")
    if sep_id != SEP_ID:
        raise ValueError(f"SEP_ID esperado={SEP_ID}, asignado={sep_id}. "
                         f"Borra {prefix}.model y reentrena.")
    return sp


class QADataset(Dataset):
    def __init__(self, pares, sp, max_seq_len):
        self.data = []
        for pregunta, respuesta in pares:
            p_ids = sp.encode(pregunta,  out_type=int)[:max_seq_len - 2]
            r_ids = sp.encode(respuesta, out_type=int)[:max_seq_len - 2]

            enc_ids = [BOS_ID] + p_ids + [SEP_ID]    # encoder input
            dec_in  = [BOS_ID] + r_ids                # decoder input
            dec_out = r_ids + [EOS_ID]                # decoder target

            self.data.append((enc_ids, dec_in, dec_out))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, max_seq_len):
    enc_b, din_b, dout_b = zip(*batch)

    enc_max  = min(max(len(x) for x in enc_b),  max_seq_len)
    din_max  = min(max(len(x) for x in din_b),  max_seq_len)
    dout_max = min(max(len(x) for x in dout_b), max_seq_len)

    B = len(batch)
    enc_pad  = torch.zeros(B, enc_max,  dtype=torch.long)
    din_pad  = torch.zeros(B, din_max,  dtype=torch.long)
    dout_pad = torch.zeros(B, dout_max, dtype=torch.long)

    for i, (enc, din, dout) in enumerate(zip(enc_b, din_b, dout_b)):
        enc, din, dout = enc[:enc_max], din[:din_max], dout[:dout_max]
        enc_pad[i,  :len(enc)]  = torch.tensor(enc)
        din_pad[i,  :len(din)]  = torch.tensor(din)
        dout_pad[i, :len(dout)] = torch.tensor(dout)

    return enc_pad, din_pad, dout_pad


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class Generador(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        V, D, H, L, FF, DR = (cfg["vocab_size"], cfg["d_model"], cfg["nhead"],
                              cfg["num_layers"], cfg["dim_feedforward"],
                              cfg["dropout"])

        self.embedding = nn.Embedding(V, D, padding_idx=PAD_ID)
        self.pos_enc   = PositionalEncoding(D, max_len=cfg["max_seq_len"]+10,
                                            dropout=DR)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=H, dim_feedforward=FF,
            dropout=DR, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=L)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=D, nhead=H, dim_feedforward=FF,
            dropout=DR, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=L)

        self.proj = nn.Linear(D, V, bias=False)
        self.proj.weight = self.embedding.weight   # weight tying

        self._init_weights(D)

    def _init_weights(self, D):
        # Embedding con N(0, 1/sqrt(d_model)), resto Xavier
        nn.init.normal_(self.embedding.weight, mean=0.0, std=D ** -0.5)
        with torch.no_grad():
            self.embedding.weight[PAD_ID].zero_()
        for name, p in self.named_parameters():
            if "embedding" in name or "pe" in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_key_padding_mask=None):
        x = self.pos_enc(self.embedding(src))
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)

    def decode_step(self, tgt, memory, tgt_mask=None,
                    tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = self.pos_enc(self.embedding(tgt))
        out = self.decoder(
            x, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.proj(out)

    def forward(self, src, dec_in, tau=1.0,
                src_pad_mask=None, dec_in_pad_mask=None):
        memory   = self.encode(src, src_key_padding_mask=src_pad_mask)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            dec_in.size(1), device=src.device)
        logits   = self.decode_step(
            dec_in, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=dec_in_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        gumbel = F.gumbel_softmax(logits, tau=tau, hard=False)
        return logits, gumbel

    @torch.no_grad()
    def generar(self, src, sp, max_new_tokens=100,
                temperature=0.8, top_k=50):
        self.eval()
        device  = src.device
        memory  = self.encode(src)
        dec_ids = torch.tensor([[BOS_ID]], device=device)

        for _ in range(max_new_tokens):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                dec_ids.size(1), device=device)
            logits   = self.decode_step(dec_ids, memory, tgt_mask=tgt_mask)
            next_logits = logits[:, -1, :].clone() / temperature

            if top_k > 0:
                k = min(top_k, next_logits.size(-1))
                vals, _ = torch.topk(next_logits, k)
                next_logits[next_logits < vals[:, -1:]] = float('-inf')

            probs    = F.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)

            if next_tok.item() == EOS_ID:
                break
            dec_ids = torch.cat([dec_ids, next_tok], dim=1)

        tokens = dec_ids[0, 1:].tolist()
        return sp.decode(tokens)


class Discriminador(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        V, D, H, L, FF, DR = (cfg["vocab_size"], cfg["d_model_dis"],
                              cfg["nhead_dis"], cfg["num_layers_dis"],
                              cfg["dim_feedforward_dis"], cfg["dropout"])

        self.embedding = nn.Embedding(V, D, padding_idx=PAD_ID)
        self.pos_enc   = PositionalEncoding(D, max_len=cfg["max_seq_len"]*2+10,
                                            dropout=DR)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=H, dim_feedforward=FF,
            dropout=DR, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=L)

        self.classifier = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Dropout(DR),
            nn.Linear(D // 2, 1),
        )
        self._init_weights(D)

    def _init_weights(self, D):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=D ** -0.5)
        with torch.no_grad():
            self.embedding.weight[PAD_ID].zero_()
        for name, p in self.named_parameters():
            if "embedding" in name or "pe" in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def mean_pool(self, enc_out, pad_mask):
        content_mask = (~pad_mask).float().unsqueeze(-1)
        sum_emb = (enc_out * content_mask).sum(dim=1)
        count   = content_mask.sum(dim=1).clamp(min=1e-9)
        return sum_emb / count

    def forward(self, src_ids, src_pad_mask,
                resp_soft=None, resp_ids=None, resp_pad_mask=None):
        src_emb = self.pos_enc(self.embedding(src_ids))

        if resp_soft is not None:
            resp_emb = self.pos_enc(resp_soft @ self.embedding.weight)
        else:
            resp_emb = self.pos_enc(self.embedding(resp_ids))

        x = torch.cat([src_emb, resp_emb], dim=1)

        if resp_pad_mask is not None:
            pad_concat = torch.cat([src_pad_mask, resp_pad_mask], dim=1)
        else:
            fake_mask = torch.zeros(
                resp_emb.size(0), resp_emb.size(1),
                dtype=torch.bool, device=src_ids.device)
            pad_concat = torch.cat([src_pad_mask, fake_mask], dim=1)

        enc_out = self.encoder(x, src_key_padding_mask=pad_concat)
        pooled  = self.mean_pool(enc_out, pad_concat)
        return self.classifier(pooled)


def hacer_pad_mask(ids):
    return ids == PAD_ID

def contar_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

@torch.no_grad()
def evaluar_lm_loss(gen, loader, device, ce, vocab_size):
    """LM loss sobre el conjunto de validación (proxy de calidad)."""
    gen.eval()
    total, n = 0.0, 0
    for src_ids, dec_in, dec_out in loader:
        src_ids = src_ids.to(device)
        dec_in  = dec_in.to(device)
        dec_out = dec_out.to(device)
        src_pad = hacer_pad_mask(src_ids)
        dec_pad = hacer_pad_mask(dec_in)

        logits, _ = gen(src_ids, dec_in, tau=1.0,
                        src_pad_mask=src_pad, dec_in_pad_mask=dec_pad)
        loss = ce(logits.reshape(-1, vocab_size), dec_out.reshape(-1))
        total += loss.item()
        n += 1
    return total / max(1, n)

def entrenar(cfg, train_pares, val_pares, sp, epochs):
    # Verificación REAL de GPU - is_available() puede mentir en Windows sin GPU física
    requested_device = cfg["device"]
    if requested_device == "cuda":
        cuda_funciona = False
        try:
            # Probar realmente si CUDA funciona
            if torch.cuda.is_available():
                test_tensor = torch.zeros(1).cuda()
                _ = test_tensor + 1   # forzar uso real
                del test_tensor
                cuda_funciona = True
        except Exception as e:
            print(f"CUDA no funciona realmente: {str(e)[:120]}")
            cuda_funciona = False

        if cuda_funciona:
            device = torch.device("cuda")
        else:
            print(f"Cambiando a CPU (entrenamiento será MUCHO más lento)")
            device = torch.device("cpu")
            cfg = dict(cfg)
            cfg["device"] = "cpu"
    else:
        device = torch.device(requested_device)
    print(f"\nDevice: {device}")

    train_ds = QADataset(train_pares, sp, cfg["max_seq_len"])
    val_ds   = QADataset(val_pares,   sp, cfg["max_seq_len"])

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        collate_fn=lambda b: collate_fn(b, cfg["max_seq_len"]))
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        collate_fn=lambda b: collate_fn(b, cfg["max_seq_len"]))

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | "
          f"{len(train_loader)} batches/época")

    gen = Generador(cfg).to(device)
    dis = Discriminador(cfg).to(device)

    print(f"Generador:     {contar_params(gen):.1f}M params")
    print(f"Discriminador: {contar_params(dis):.1f}M params")
    print(f"Total:         {contar_params(gen)+contar_params(dis):.1f}M params")

    opt_gen = torch.optim.AdamW(gen.parameters(), lr=cfg["lr_gen"],
                                 betas=(0.9, 0.98), weight_decay=0.01)
    opt_dis = torch.optim.AdamW(dis.parameters(), lr=cfg["lr_dis"],
                                 betas=(0.9, 0.98), weight_decay=0.01)

    warmup = max(1, len(train_loader) * 2)
    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        return max(0.1, 1.0 - (step - warmup) / max(1, epochs * len(train_loader)))

    sched_gen = torch.optim.lr_scheduler.LambdaLR(opt_gen, lr_lambda)
    sched_dis = torch.optim.lr_scheduler.LambdaLR(opt_dis, lr_lambda)

    bce = nn.BCEWithLogitsLoss()
    ce  = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    # RESUME - cargar checkpoint si existe
    RESUME_CKPT = "gan_train_state.pt"   # checkpoint completo del entrenamiento
    log = []
    best_val = float('inf')
    epoch_inicial = 1

    if os.path.exists(RESUME_CKPT):
        try:
            print(f"\nEncontrado checkpoint de entrenamiento: {RESUME_CKPT}")
            ckpt = torch.load(RESUME_CKPT, map_location=device, weights_only=False)
            gen.load_state_dict(ckpt["gen_state"])
            dis.load_state_dict(ckpt["dis_state"])
            opt_gen.load_state_dict(ckpt["opt_gen_state"])
            opt_dis.load_state_dict(ckpt["opt_dis_state"])
            sched_gen.load_state_dict(ckpt["sched_gen_state"])
            sched_dis.load_state_dict(ckpt["sched_dis_state"])
            epoch_inicial = ckpt["epoch"] + 1
            best_val = ckpt["best_val"]
            log = ckpt.get("log", [])
            print(f"Resumiendo desde época {epoch_inicial} (best_val={best_val:.4f})")
        except Exception as e:
            print(f"Error cargando checkpoint: {e}. Empezando desde cero.")
            epoch_inicial = 1
            best_val = float('inf')
            log = []

    if epoch_inicial > epochs:
        print(f"Entrenamiento ya completado ({epoch_inicial-1}/{epochs} épocas)")
        return

    print(f"\nEntrenando {epochs} épocas (desde {epoch_inicial})\n")

    for epoch in range(epoch_inicial, epochs + 1):
        gen.train()
        dis.train()

        sum_lm = sum_adv = sum_dis = 0.0
        n = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{epochs}", leave=False)

        for src_ids, dec_in, dec_out in pbar:
            src_ids = src_ids.to(device)
            dec_in  = dec_in.to(device)
            dec_out = dec_out.to(device)
            B = src_ids.size(0)

            src_pad = hacer_pad_mask(src_ids)
            dec_pad = hacer_pad_mask(dec_in)

            real_resp     = dec_out
            real_resp_pad = hacer_pad_mask(real_resp)

            real_lbl = torch.ones(B,  1, device=device)
            fake_lbl = torch.zeros(B, 1, device=device)

            opt_dis.zero_grad()

            pred_real = dis(src_ids, src_pad,
                            resp_ids=real_resp, resp_pad_mask=real_resp_pad)
            loss_real = bce(pred_real, real_lbl)

            with torch.no_grad():
                _, gumbel = gen(src_ids, dec_in, tau=cfg["gumbel_tau"],
                                src_pad_mask=src_pad, dec_in_pad_mask=dec_pad)

            pred_fake = dis(src_ids, src_pad, resp_soft=gumbel.detach())
            loss_fake = bce(pred_fake, fake_lbl)

            loss_dis = (loss_real + loss_fake) / 2
            loss_dis.backward()
            nn.utils.clip_grad_norm_(dis.parameters(), 1.0)
            opt_dis.step()
            sched_dis.step()

            opt_gen.zero_grad()

            logits, gumbel = gen(src_ids, dec_in, tau=cfg["gumbel_tau"],
                                 src_pad_mask=src_pad, dec_in_pad_mask=dec_pad)

            lm_loss  = ce(logits.reshape(-1, cfg["vocab_size"]),
                          dec_out.reshape(-1))
            pred_gen = dis(src_ids, src_pad, resp_soft=gumbel)
            adv_loss = bce(pred_gen, real_lbl)

            lm_w     = max(0.3, 1.0 - epoch / epochs)
            adv_w    = 1.0 - lm_w
            loss_gen = lm_w * lm_loss + adv_w * adv_loss

            loss_gen.backward()
            nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
            opt_gen.step()
            sched_gen.step()

            sum_lm  += lm_loss.item()
            sum_adv += adv_loss.item()
            sum_dis += loss_dis.item()
            n += 1

            pbar.set_postfix({
                "LM":  f"{lm_loss.item():.3f}",
                "Adv": f"{adv_loss.item():.3f}",
                "Dis": f"{loss_dis.item():.3f}",
            })

        avg_lm  = sum_lm  / n
        avg_adv = sum_adv / n
        avg_dis = sum_dis / n

        # Validación
        val_lm = evaluar_lm_loss(gen, val_loader, device, ce, cfg["vocab_size"])

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"LM_train: {avg_lm:.4f} | LM_val: {val_lm:.4f} | "
              f"Adv: {avg_adv:.4f} | Dis: {avg_dis:.4f}")

        log.append({"epoch": epoch,
                    "loss_lm_train": round(avg_lm,  4),
                    "loss_lm_val":   round(val_lm,  4),
                    "loss_adv":      round(avg_adv, 4),
                    "loss_dis":      round(avg_dis, 4)})
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2)

        # Guardar mejor modelo según val_lm + checkpoint regular
        is_best = val_lm < best_val
        if is_best:
            best_val = val_lm
            torch.save({"epoch": epoch, "cfg": cfg, "val_lm": val_lm,
                        "state_dict": gen.state_dict()}, GEN_CKPT)
            torch.save({"epoch": epoch, "cfg": cfg,
                        "state_dict": dis.state_dict()}, DIS_CKPT)
            print(f"  Mejor val_lm={val_lm:.4f} -> checkpoint guardado")

        # Checkpoint completo del entrenamiento (para resume si se cae)
        torch.save({
            "epoch":            epoch,
            "best_val":         best_val,
            "cfg":              cfg,
            "gen_state":        gen.state_dict(),
            "dis_state":        dis.state_dict(),
            "opt_gen_state":    opt_gen.state_dict(),
            "opt_dis_state":    opt_dis.state_dict(),
            "sched_gen_state":  sched_gen.state_dict(),
            "sched_dis_state":  sched_dis.state_dict(),
            "log":              log,
        }, RESUME_CKPT)

        if epoch % 10 == 0:
            gen.eval()
            muestra = train_pares[0][0]
            src_t   = torch.tensor(
                [[BOS_ID] + sp.encode(muestra, out_type=int)[:60] + [SEP_ID]],
                device=device)
            with torch.no_grad():
                resp = gen.generar(src_t, sp, max_new_tokens=80)
            print(f"\n  Muestra época {epoch}:")
            print(f"     P: {muestra[:80]}")
            print(f"     R: {resp[:150]}\n")

    print(f"\nEntrenamiento completado. Mejor val_lm={best_val:.4f}")
    print(f"  Log: {LOG_FILE}")

# Esquema alineado con costes.py del TFG:
# Tokens contados con tiktoken cl100k_base
# Coste = tok_in x precio_in + tok_out x precio_out
# CO2   = (tok_in + tok_out) x kwh_token x CO2_POR_KWH
# Para la GAN (modelo local entrenado por nosotros):
# precio_in = precio_out = 0  (no hay tarifa de API)
# kwh_token estimado proporcional al tamaño (~430M params)
# TOKENS_PROMPT_SISTEMA = 0  (la GAN no usa prompt de sistema)

GAN_PRECIO_INPUT      = 0.0          # $/token entrada (modelo local)
GAN_PRECIO_OUTPUT     = 0.0          # $/token salida  (modelo local)
GAN_KWH_POR_TOKEN     = 0.00000001   # kWh/token estimado para ~430M params
CO2_POR_KWH           = 475          # gCO2/kWh (mismo valor que costes.py)
TOKENS_PROMPT_SISTEMA = 0            # GAN no añade prompt de sistema

def contar_tokens_tiktoken(texto):
    """Mismo método que costes.py para que los conteos sean comparables."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(str(texto)))
    except ImportError:
        return len(str(texto)) // 4

def evaluar_metricas(test_pares, sp, device="cpu"):
    """
    Genera respuestas para todo el test set y guarda por fila:
      - tiempo de generación (s)
      - Tokens_Input, Tokens_Output  (tiktoken, igual que costes.py)
      - métricas: ROUGE-L, BLEU, METEOR, BERTScore-F1
      - Coste_USD   (= 0 para GAN local)
      - CO2_gramos  (= (tok_in + tok_out) x kwh_token x 475)
    Esquema idéntico al de costes.py para comparabilidad directa.
    """
    import time

    try:
        from rouge_score import rouge_scorer
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.translate.meteor_score import meteor_score
        import nltk
        for pkg in ["wordnet", "omw-1.4", "punkt"]:
            try:
                nltk.data.find(f"corpora/{pkg}")
            except LookupError:
                nltk.download(pkg, quiet=True)
    except ImportError as e:
        print(f"Faltan dependencias: {e}")
        print("  pip install nltk rouge-score bert-score tiktoken")
        return

    if not os.path.exists(GEN_CKPT):
        print(f"No existe {GEN_CKPT}. Entrena primero.")
        return

    ckpt = torch.load(GEN_CKPT, map_location=device, weights_only=False)
    cfg  = ckpt["cfg"]
    gen  = Generador(cfg).to(device)
    gen.load_state_dict(ckpt["state_dict"])
    gen.eval()

    is_gpu = (device == "cuda" or
              (hasattr(device, 'type') and device.type == "cuda"))

    print(f"Modelo cargado (época {ckpt.get('epoch','?')}, "
          f"val_lm={ckpt.get('val_lm','?')})")
    print(f"\nEvaluando sobre {len(test_pares)} ejemplos del test set...\n")

    rouge  = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smooth = SmoothingFunction().method1

    refs, hyps, rows = [], [], []
    t_total_start = time.time()

    for i, (preg, ref) in enumerate(tqdm(test_pares, desc="Generando")):
        # Tokenizar pregunta para el modelo (SentencePiece)
        preg_ids = sp.encode(preg, out_type=int)[:cfg["max_seq_len"]-2]
        src = torch.tensor([[BOS_ID] + preg_ids + [SEP_ID]], device=device)

        # Cronometrar la generación
        if is_gpu:
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            hyp = gen.generar(src, sp, max_new_tokens=150,
                              temperature=0.8, top_k=50)
        if is_gpu:
            torch.cuda.synchronize()
        elapsed_s = time.time() - t0

        if not hyp.strip():
            hyp = "<vacio>"

        # Conteo de tokens estilo costes.py (tiktoken sobre el texto)
        tok_in  = contar_tokens_tiktoken(preg) + TOKENS_PROMPT_SISTEMA
        tok_out = contar_tokens_tiktoken(hyp)

        # Métricas de calidad
        rouge_l = rouge.score(ref, hyp)['rougeL'].fmeasure
        ref_tok = ref.split()
        hyp_tok = hyp.split()
        bleu = sentence_bleu([ref_tok], hyp_tok,
                             weights=(0.25, 0.25, 0.25, 0.25),
                             smoothing_function=smooth) if hyp_tok else 0.0
        try:
            meteor = meteor_score([ref_tok], hyp_tok) if hyp_tok else 0.0
        except Exception:
            meteor = 0.0

        # Coste y CO2 (esquema costes.py)
        coste_usd  = tok_in * GAN_PRECIO_INPUT + tok_out * GAN_PRECIO_OUTPUT
        co2_gramos = (tok_in + tok_out) * GAN_KWH_POR_TOKEN * CO2_POR_KWH

        rows.append({
            "i":             i,
            "pregunta":      preg,
            "referencia":    ref,
            "generado":      hyp,
            "tiempo_s":      round(elapsed_s, 4),
            "Tokens_Input":  tok_in,
            "Tokens_Output": tok_out,
            "rougeL":        round(rouge_l, 4),
            "bleu":          round(bleu,    4),
            "meteor":        round(meteor,  4),
            "Coste_USD":     round(coste_usd,  8),
            "CO2_gramos":    round(co2_gramos, 6),
        })

        refs.append(ref)
        hyps.append(hyp)

    t_total = time.time() - t_total_start

    # BERTScore en batch
    print("\nCalculando BERTScore...")
    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(hyps, refs, lang="es", verbose=False,
                                 device=device)
        for i, f1 in enumerate(F1.tolist()):
            rows[i]["bertscore_f1"] = round(f1, 4)
        bert_avg = float(F1.mean())
    except Exception as e:
        print(f"BERTScore falló: {e}")
        bert_avg = None
        for r in rows:
            r["bertscore_f1"] = None

    # Promedios y totales
    n = len(rows)
    avg = {
        "rougeL":         round(sum(r["rougeL"]        for r in rows) / n, 4),
        "bleu":           round(sum(r["bleu"]          for r in rows) / n, 4),
        "meteor":         round(sum(r["meteor"]        for r in rows) / n, 4),
        "bertscore_f1":   round(bert_avg, 4) if bert_avg is not None else None,
        "tiempo_s":       round(sum(r["tiempo_s"]      for r in rows) / n, 4),
        "Tokens_Output":  round(sum(r["Tokens_Output"] for r in rows) / n, 1),
    }
    totals = {
        "tiempo_total_s":    round(t_total, 2),
        "Tokens_Input_sum":  sum(r["Tokens_Input"]  for r in rows),
        "Tokens_Output_sum": sum(r["Tokens_Output"] for r in rows),
        "Coste_USD_total":   round(sum(r["Coste_USD"]  for r in rows), 6),
        "CO2_gramos_total":  round(sum(r["CO2_gramos"] for r in rows), 4),
    }

    print("\n" + "="*55)
    print("  RESULTADOS GAN - TEST SET")
    print("="*55)
    print(f"  Ejemplos:           {n}")
    print(f"  Tiempo total:       {totals['tiempo_total_s']:.1f} s "
          f"({totals['tiempo_total_s']/60:.1f} min)")
    print(f"  Tiempo medio/fila:  {avg['tiempo_s']:.3f} s")
    print(f"  Tokens medios out:  {avg['Tokens_Output']:.1f}")
    print()
    print(f"  ROUGE-L:            {avg['rougeL']}")
    print(f"  BLEU:               {avg['bleu']}")
    print(f"  METEOR:             {avg['meteor']}")
    print(f"  BERTScore F1:       {avg['bertscore_f1']}")
    print()
    print(f"  Coste total:        ${totals['Coste_USD_total']:.6f}  "
          f"(modelo local, tarifa = 0)")
    print(f"  CO2 total:          {totals['CO2_gramos_total']:.4f} g")
    print("="*55)

    # JSON detallado
    with open(EVAL_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            "n_test":           n,
            "checkpoint_epoch": ckpt.get("epoch"),
            "esquema_costes": {
                "fuente":                "costes.py del TFG",
                "tokens":                "tiktoken cl100k_base",
                "precio_input":          GAN_PRECIO_INPUT,
                "precio_output":         GAN_PRECIO_OUTPUT,
                "kwh_por_token":         GAN_KWH_POR_TOKEN,
                "co2_por_kwh":           CO2_POR_KWH,
                "tokens_prompt_sistema": TOKENS_PROMPT_SISTEMA,
            },
            "avg":              avg,
            "totals":           totals,
            "rows":             rows,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados en {EVAL_FILE}")

    # CSV con el mismo formato que costes.py añade a tus análisis
    csv_file = EVAL_FILE.replace('.json', '.csv')
    import csv as csv_mod
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv_mod.writer(f)
        writer.writerow([
            "i", "pregunta", "referencia", "generado",
            "tiempo_s", "Tokens_Input", "Tokens_Output",
            "rougeL", "bleu", "meteor", "bertscore_f1",
            "Coste_USD", "CO2_gramos",
        ])
        for r in rows:
            writer.writerow([
                r["i"], r["pregunta"], r["referencia"], r["generado"],
                r["tiempo_s"], r["Tokens_Input"], r["Tokens_Output"],
                r["rougeL"], r["bleu"], r["meteor"], r.get("bertscore_f1", ""),
                r["Coste_USD"], r["CO2_gramos"],
            ])
    print(f"CSV compatible con costes.py: {csv_file}")


def generar_respuesta(pregunta, sp, device="cpu"):
    ckpt = torch.load(GEN_CKPT, map_location=device, weights_only=False)
    cfg  = ckpt["cfg"]
    gen  = Generador(cfg).to(device)
    gen.load_state_dict(ckpt["state_dict"])
    gen.eval()

    src = torch.tensor(
        [[BOS_ID] + sp.encode(pregunta, out_type=int)[:cfg["max_seq_len"]-2]
          + [SEP_ID]], device=device)

    with torch.no_grad():
        return gen.generar(src, sp, max_new_tokens=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modo",       default="prueba",
                        choices=["prueba", "final"])
    parser.add_argument("--asignatura", default="Programación I")
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--generar",    action="store_true")
    parser.add_argument("--evaluar",    action="store_true",
                        help="Evaluar sobre test set con métricas")
    parser.add_argument("--pregunta",   default=None)
    parser.add_argument("--csv",        default=DATASET_CSV,
                        help="Ruta al dataset_teleco.csv")
    parser.add_argument("--splits-json", default=None,
                        help="Ruta al JSON de split por subtema (recomendado)")
    parser.add_argument("--vocab-size", type=int, default=None,
                        help="Tamaño del vocabulario BPE (override del modo)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch_size (fallback para OOM)")
    args = parser.parse_args()

    cfg = CONFIGS[args.modo]

    # Override vocab_size si se especifica (útil para smoke tests con pocos datos)
    if args.vocab_size is not None:
        cfg = dict(cfg)  # copia para no modificar el dict global
        cfg["vocab_size"] = args.vocab_size
        print(f"vocab_size override: {args.vocab_size}")

    if args.batch_size is not None:
        if not isinstance(cfg, dict):
            cfg = dict(cfg)
        cfg["batch_size"] = args.batch_size
        print(f"batch_size override: {args.batch_size}")

    # Reproducibilidad
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Modo: generar respuesta individual
    if args.generar:
        if not os.path.exists(f"{SPM_PREFIX}.model"):
            print("No hay tokenizador. Entrena primero.")
            return
        sp = cargar_tokenizador(SPM_PREFIX)
        pregunta = args.pregunta or input("Pregunta: ")
        print(f"\nGenerando respuesta...")
        print(f"\n{generar_respuesta(pregunta, sp, cfg['device'])}\n")
        return

    # Modo: evaluar con métricas
    if args.evaluar:
        if not os.path.exists(f"{SPM_PREFIX}.model"):
            print("No hay tokenizador. Entrena primero.")
            return
        if not os.path.exists(args.csv):
            print(f"No existe {args.csv}")
            print(f"  Ejecuta primero: python reestructurar_dataset.py")
            return
        sp = cargar_tokenizador(SPM_PREFIX)
        asig  = args.asignatura if args.modo == "prueba" else None
        pares = leer_dataset(args.csv, asignatura_filtro=asig)

        # Usar split externo si existe
        if args.splits_json and os.path.exists(args.splits_json):
            _, _, test_pares = split_dataset_externo(pares, args.splits_json)
        else:
            print("No hay --splits-json, usando split aleatorio LEGACY")
            _, _, test_pares = split_dataset(pares)

        evaluar_metricas(test_pares, sp, device=cfg["device"])
        return

    # Modo: entrenar
    if not os.path.exists(args.csv):
        print(f"No existe {args.csv}")
        print(f"  Ejecuta primero: python reestructurar_dataset.py")
        sys.exit(1)

    asig  = args.asignatura if (args.modo == "prueba" and args.asignatura and args.asignatura != "all") else None
    pares = leer_dataset(args.csv, asignatura_filtro=asig)

    if not pares:
        print("Sin pares. Verifica el nombre de la asignatura.")
        sys.exit(1)

    # Usar split externo si existe (cero data leakage)
    if args.splits_json and os.path.exists(args.splits_json):
        train_pares, val_pares, test_pares = split_dataset_externo(pares, args.splits_json)
    else:
        if args.modo == "final":
            print("ATENCIÓN: --splits-json no proporcionado en modo final.")
            print("   Usando split aleatorio (puede haber data leakage).")
            print("   Recomendación: ejecuta hacer_splits.py primero.")
        train_pares, val_pares, test_pares = split_dataset(pares)

    # Tokenizador entrenado SOLO con train (evita filtrado del test)
    if not os.path.exists(f"{SPM_PREFIX}.model"):
        entrenar_tokenizador(train_pares, cfg["vocab_size"], SPM_PREFIX)
    else:
        print(f"Tokenizador ya existe: {SPM_PREFIX}.model")

    sp = cargar_tokenizador(SPM_PREFIX)

    print(f"\n{'='*55}")
    print(f"  MODO:        {args.modo.upper()}")
    print(f"  ASIGNATURA:  {asig or 'todas'}")
    print(f"  TRAIN/VAL/TEST: {len(train_pares)}/{len(val_pares)}/{len(test_pares)}")
    print(f"  ÉPOCAS:      {args.epochs}")
    print(f"  DEVICE:      {cfg['device']}")
    print(f"{'='*55}\n")

    entrenar(cfg, train_pares, val_pares, sp, epochs=args.epochs)

    print(f"\nPara evaluar con métricas sobre el test set:")
    print(f"   python {os.path.basename(__file__)} --evaluar "
          f"--modo {args.modo}"
          + (f" --splits-json {args.splits_json}" if args.splits_json else ""))


if __name__ == "__main__":
    main()