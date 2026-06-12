"""
Genera el split del dataset por subtema para evitar data leakage.

Lee dataset_teleco.csv y produce splits/splits_por_subtema.json con las listas
train_ids, val_ids y test_ids. Cada subtema se asigna por completo a un unico
split (70/15/15), estratificado por asignatura, de modo que ningun subtema
aparece en mas de un split.

Uso:
    python hacer_splits.py
    python hacer_splits.py --csv dataset_teleco.csv --outdir resultados_arturo/splits
"""

import sys
# Forzar UTF-8 en stdout/stderr (necesario en Windows con cp1252)
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


import os
import csv
import json
import random
import argparse
from collections import defaultdict

SEED = 42

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    default="dataset_teleco.csv",
                        help="Ruta al dataset_teleco.csv")
    parser.add_argument("--outdir", default="resultados_arturo/splits",
                        help="Directorio de salida")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio",   type=float, default=0.15)
    args = parser.parse_args()

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio <= 0:
        print("Las ratios train+val deben sumar menos de 1.0")
        return

    if not os.path.exists(args.csv):
        print(f"No existe {args.csv}")
        return

    os.makedirs(args.outdir, exist_ok=True)

    # Leer dataset y agrupar por subtema
    print(f"Leyendo {args.csv}...")
    with open(args.csv, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"{len(rows)} filas cargadas")

    # Agrupar IDs por (asignatura, tema, subtema)
    # Esto identifica de forma única cada subtema
    grupos = defaultdict(list)
    for row in rows:
        key = (row['asignatura'], row['tema'], row['subtema'])
        grupos[key].append(int(row['id']))

    print(f"{len(grupos)} subtemas únicos identificados")

    # Estratificar por asignatura
    # Para que cada asignatura tenga proporcionalmente split 70/15/15
    asig_to_subtemas = defaultdict(list)
    for key in grupos:
        asignatura = key[0]
        asig_to_subtemas[asignatura].append(key)

    print(f"{len(asig_to_subtemas)} asignaturas")

    # Asignar subtemas a splits
    rng = random.Random(SEED)
    train_subtemas = []
    val_subtemas   = []
    test_subtemas  = []

    for asig, subtemas in sorted(asig_to_subtemas.items()):
        # Mezclar los subtemas de esta asignatura
        subtemas_shuffled = subtemas.copy()
        rng.shuffle(subtemas_shuffled)

        n = len(subtemas_shuffled)
        n_train = int(n * args.train_ratio)
        n_val   = int(n * args.val_ratio)

        # Garantizar al menos 1 subtema en val y test si hay >= 3 subtemas
        if n >= 3:
            n_train = max(1, n_train)
            n_val   = max(1, n_val)
            n_test  = n - n_train - n_val
            if n_test < 1:
                n_train -= 1
                n_test = 1
        elif n == 2:
            n_train, n_val, n_test = 1, 1, 0
        else:
            n_train, n_val, n_test = 1, 0, 0

        train_subtemas.extend(subtemas_shuffled[:n_train])
        val_subtemas.extend(  subtemas_shuffled[n_train:n_train+n_val])
        test_subtemas.extend( subtemas_shuffled[n_train+n_val:])

    # Convertir subtemas a IDs
    train_ids = sorted([id_ for st in train_subtemas for id_ in grupos[st]])
    val_ids   = sorted([id_ for st in val_subtemas   for id_ in grupos[st]])
    test_ids  = sorted([id_ for st in test_subtemas  for id_ in grupos[st]])

    # Verificación de no-solapamiento
    train_set = set(train_ids)
    val_set   = set(val_ids)
    test_set  = set(test_ids)

    overlap_tv = train_set & val_set
    overlap_tt = train_set & test_set
    overlap_vt = val_set & test_set

    if overlap_tv or overlap_tt or overlap_vt:
        print(f"ERROR: Hay solapamiento entre splits!")
        print(f"  train AND val:  {len(overlap_tv)}")
        print(f"  train AND test: {len(overlap_tt)}")
        print(f"  val AND test:   {len(overlap_vt)}")
        return
    else:
        print(f"Cero solapamiento entre splits (verificado)")

    # Verificar también que los subtemas no solapan
    train_st = set(train_subtemas)
    val_st   = set(val_subtemas)
    test_st  = set(test_subtemas)
    if train_st & val_st or train_st & test_st or val_st & test_st:
        print(f"ERROR: Hay subtemas en múltiples splits!")
        return
    else:
        print(f"Cero solapamiento de subtemas entre splits")

    # Resumen
    n_total = len(rows)
    print(f"\n{'='*55}")
    print(f"  RESULTADO DEL SPLIT POR SUBTEMA")
    print(f"{'='*55}")
    print(f"  Subtemas:")
    print(f"    Train: {len(train_subtemas):>5}  ({len(train_subtemas)/len(grupos)*100:.1f}%)")
    print(f"    Val:   {len(val_subtemas):>5}  ({len(val_subtemas)/len(grupos)*100:.1f}%)")
    print(f"    Test:  {len(test_subtemas):>5}  ({len(test_subtemas)/len(grupos)*100:.1f}%)")
    print(f"    Total: {len(grupos):>5}")
    print(f"  Filas:")
    print(f"    Train: {len(train_ids):>6}  ({len(train_ids)/n_total*100:.1f}%)")
    print(f"    Val:   {len(val_ids):>6}  ({len(val_ids)/n_total*100:.1f}%)")
    print(f"    Test:  {len(test_ids):>6}  ({len(test_ids)/n_total*100:.1f}%)")
    print(f"    Total: {n_total:>6}")
    print(f"{'='*55}")

    # Guardar JSON
    out_path = os.path.join(args.outdir, "splits_por_subtema.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            "seed":          SEED,
            "metodo":        "split por subtema (estratificado por asignatura)",
            "train_ratio":   args.train_ratio,
            "val_ratio":     args.val_ratio,
            "test_ratio":    test_ratio,
            "n_subtemas_total": len(grupos),
            "n_subtemas_train": len(train_subtemas),
            "n_subtemas_val":   len(val_subtemas),
            "n_subtemas_test":  len(test_subtemas),
            "n_filas_total":    n_total,
            "n_filas_train":    len(train_ids),
            "n_filas_val":      len(val_ids),
            "n_filas_test":     len(test_ids),
            "train_ids":     train_ids,
            "val_ids":       val_ids,
            "test_ids":      test_ids,
        }, f, indent=2)

    print(f"\nGuardado: {out_path}")
    print(f"  Tamaño: {os.path.getsize(out_path)/1024:.0f} KB")


if __name__ == "__main__":
    main()