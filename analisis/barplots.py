# pip install pandas matplotlib numpy
"""
Barplots globales estilo paper — 9 modelos.
6 originales (LLM+SLM) + 3 mini-SLMs.
Genera Coste, CO2, Tiempo a partir de los CSVs por fila (no necesita resumen_global).

USO: python barplots_9modelos.py --input-dir analisis
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob
import os
import argparse
import numpy as np

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})

MODELO_COLORES = {
    'DeepSeek':    '#2E75B6',
    'Llama 70B':   '#D6604D',
    'Qwen 72B':    '#E08214',
    'Gemma 9B':    '#4393C3',
    'Llama 3B':    '#7570B3',
    'Qwen 7B':     '#1B7837',
    'Gemma 3.1B':  '#E7298A',
    'Llama 1B':    '#66A61E',
    'Qwen3 1.7B':  '#D95F02',
}

ORDEN_MODELOS = [
    'DeepSeek', 'Llama 70B', 'Qwen 72B',
    'Gemma 9B', 'Llama 3B', 'Qwen 7B',
    'Gemma 3.1B', 'Llama 1B', 'Qwen3 1.7B',
]

DATASETS = ['WebNLG', 'ToTTo', 'KELM']

ARCHIVO_MAPA = [
    ('webNLG_Gemma_3_1B',       'Gemma 3.1B',  'WebNLG'),
    ('webNLG_Llama_1B',         'Llama 1B',    'WebNLG'),
    ('webNLG_Qwen3_1.7B',      'Qwen3 1.7B',  'WebNLG'),
    ('totto_Gemma_3_1B',        'Gemma 3.1B',  'ToTTo'),
    ('totto_Llama_1B',          'Llama 1B',    'ToTTo'),
    ('totto_Qwen3_1.7B',       'Qwen3 1.7B',  'ToTTo'),
    ('kelm_stem_Gemma_3_1B',   'Gemma 3.1B',  'KELM'),
    ('kelm_stem_Llama_1B',     'Llama 1B',    'KELM'),
    ('kelm_stem_Qwen3_1.7B',  'Qwen3 1.7B',  'KELM'),
    ('DeepSeek_WebNLG',   'DeepSeek',  'WebNLG'),
    ('DeepSeek_ToTTo',    'DeepSeek',  'ToTTo'),
    ('DeepSeek_KELM',     'DeepSeek',  'KELM'),
    ('Llama_70B_WebNLG',  'Llama 70B', 'WebNLG'),
    ('Llama_70B_ToTTo',   'Llama 70B', 'ToTTo'),
    ('Llama_70B_KELM',    'Llama 70B', 'KELM'),
    ('Qwen_72B_WebNLG',   'Qwen 72B',  'WebNLG'),
    ('Qwen_72B_ToTTo',    'Qwen 72B',  'ToTTo'),
    ('Qwen_72B_KELM',     'Qwen 72B',  'KELM'),
    ('Gemma_9B_WebNLG',   'Gemma 9B',  'WebNLG'),
    ('Gemma_9B_ToTTo',    'Gemma 9B',  'ToTTo'),
    ('Gemma_9B_KELM',     'Gemma 9B',  'KELM'),
    ('Llama_3B_WebNLG',   'Llama 3B',  'WebNLG'),
    ('Llama_3B_ToTTo',    'Llama 3B',  'ToTTo'),
    ('Llama_3B_KELM',     'Llama 3B',  'KELM'),
    ('Qwen_7B_WebNLG',    'Qwen 7B',   'WebNLG'),
    ('Qwen_7B_ToTTo',     'Qwen 7B',   'ToTTo'),
    ('Qwen_7B_KELM',      'Qwen 7B',   'KELM'),
]


def cargar_todos(input_dir):
    df_total = pd.DataFrame()
    archivos = glob.glob(os.path.join(input_dir, "metricas_por_fila_*.csv"))
    for archivo in archivos:
        nombre_base = os.path.basename(archivo).replace("metricas_por_fila_", "").replace(".csv", "")
        modelo, dataset = None, None
        for patron, m, d in ARCHIVO_MAPA:
            if nombre_base == patron:
                modelo, dataset = m, d
                break
        if modelo is None:
            continue
        try:
            df_temp = pd.read_csv(archivo)
            df_temp['Modelo'] = modelo
            df_temp['Dataset'] = dataset
            df_total = pd.concat([df_total, df_temp], ignore_index=True)
        except Exception as e:
            print(f"  [ERROR] {archivo}: {e}")
    return df_total


def hex_to_rgba(hex_color, alpha=0.25):
    h = hex_color.lstrip('#')
    r, g, b = tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))
    return (r, g, b, alpha)


def generar_barplots_globales(df):
    """Genera barplots de totales (suma) por modelo y dataset desde los CSVs por fila."""

    # Construir resumen sumando por (modelo, dataset)
    col_map = {
        'Coste_USD':     ('coste_usd',     'Coste Total (USD)',                 'USD',           '${:.4f}'),
        'CO2_gramos':    ('co2_gramos',     'CO$_2$ Total (gramos)',            'gramos CO$_2$', '{:.1f}g'),
        'Time_seconds':  ('tiempo_total_s', 'Tiempo Total',                     'Segundos',      '{:.0f}s'),
    }

    datasets_presentes = [d for d in DATASETS if d in df['Dataset'].unique()]
    modelos_presentes = [m for m in ORDEN_MODELOS if m in df['Modelo'].unique()]

    if not datasets_presentes or not modelos_presentes:
        print("No hay suficientes datos.")
        return

    # Conteos de filas por dataset (primer modelo que tenga datos)
    conteos_filas = {}
    for ds in datasets_presentes:
        sub = df[df['Dataset'] == ds]
        if not sub.empty:
            primer_modelo = sub['Modelo'].iloc[0]
            conteos_filas[ds] = len(sub[sub['Modelo'] == primer_modelo])
        else:
            conteos_filas[ds] = '?'

    n_ds = len(datasets_presentes)
    n_mod = len(modelos_presentes)
    bar_w = 0.8 / n_mod
    x = np.arange(n_ds)

    for col_fila, (_, titulo, ylabel, fmt) in col_map.items():
        if col_fila not in df.columns:
            print(f"  Columna {col_fila} no encontrada, saltando {titulo}")
            continue

        # Resumen: suma por (modelo, dataset)
        resumen = df.groupby(['Modelo', 'Dataset'])[col_fila].sum().reset_index()
        resumen.rename(columns={col_fila: 'valor'}, inplace=True)

        fig, ax = plt.subplots(figsize=(13, 6))
        max_val = resumen['valor'].max()

        for k, m in enumerate(modelos_presentes):
            offset = (k - n_mod / 2 + 0.5) * bar_w
            vals = []
            for ds in datasets_presentes:
                v = resumen[(resumen['Modelo'] == m) & (resumen['Dataset'] == ds)]['valor']
                vals.append(v.values[0] if not v.empty else np.nan)

            c = MODELO_COLORES.get(m, '#888')
            bars = ax.bar(x + offset, vals, width=bar_w * 0.9,
                          color=hex_to_rgba(c, 0.4), edgecolor=c, linewidth=1.2, label=m)

            for bar, val in zip(bars, vals):
                if pd.notna(val) and val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + (max_val * 0.015),
                            fmt.format(val), ha='center', va='bottom', fontsize=6, rotation=0)

        # Etiquetas X
        labels_x = []
        for ds in datasets_presentes:
            if col_fila == 'CO2_gramos':
                labels_x.append(f"{ds}\n({conteos_filas.get(ds, '?')} datos)")
            else:
                labels_x.append(ds)

        ax.set_xticks(x)
        ax.set_xticklabels(labels_x, fontsize=12)
        ax.tick_params(axis='x', rotation=0, direction='in', length=7)
        ax.tick_params(axis='y', direction='in', length=7)

        ax.set_title(titulo, fontsize=13, fontweight='normal', pad=10)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_ylim(0, max_val * 1.25)

        ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='#888888', linewidth=0.5)
        ax.set_axisbelow(True)

        legend_patches = [
            mpatches.Patch(facecolor=hex_to_rgba(MODELO_COLORES[m], 0.4),
                           edgecolor=MODELO_COLORES[m], linewidth=1.2, label=m)
            for m in modelos_presentes
        ]
        ax.legend(handles=legend_patches, loc='upper right', fontsize=8,
                  framealpha=0.95, facecolor='white', edgecolor='black', fancybox=False)

        plt.tight_layout()
        nombre_png = f"Global_9m_{col_fila}.png"
        plt.savefig(nombre_png, dpi=300, facecolor='white')
        plt.close()
        print(f"  {nombre_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="analisis")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: No se encuentra '{args.input_dir}'")
        exit(1)

    print("Cargando datos (9 modelos)...")
    df = cargar_todos(args.input_dir)
    if df.empty:
        print("Error: No se han cargado datos.")
        exit(1)

    print(f"  {len(df)} filas | {df['Modelo'].nunique()} modelos | {df['Dataset'].nunique()} datasets\n")

    print("Generando barplots globales (9 modelos)...")
    generar_barplots_globales(df)
