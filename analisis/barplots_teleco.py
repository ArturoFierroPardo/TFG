# pip install pandas matplotlib numpy
"""
Barplots totales BBDD Teleco — 7 modelos.
GAN vs Mini-SLM Base vs Mini-SLM Fine-Tuned.
Genera Coste, CO2, Tiempo sumando desde los CSVs por fila.

USO: python barplots_teleco.py --input-dir analisis
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
    'GAN':              '#E41A1C',
    'Gemma 3.1B':       '#E7298A',
    'Llama 1B':         '#66A61E',
    'Qwen3 1.7B':       '#D95F02',
    'Gemma 3.1B FT':    '#984EA3',
    'Llama 1B FT':      '#377EB8',
    'Qwen3 1.7B FT':    '#FF7F00',
}

ORDEN_MODELOS = [
    'GAN',
    'Gemma 3.1B', 'Llama 1B', 'Qwen3 1.7B',
    'Gemma 3.1B FT', 'Llama 1B FT', 'Qwen3 1.7B FT',
]

ARCHIVO_MAPA = [
    ('GAN_valtest',              'GAN'),
    ('teleco_Gemma_3_1B',        'Gemma 3.1B'),
    ('teleco_Llama_1B',          'Llama 1B'),
    ('teleco_Qwen3_1.7B',       'Qwen3 1.7B'),
    ('teleco_Qwen_7B',          None),
    ('teleco_Gemma_9B',          None),
    ('teleco_Llama_3B',          None),
    ('Gemma_3_1B_FT_valtest',   'Gemma 3.1B FT'),
    ('Llama_1B_FT_valtest',     'Llama 1B FT'),
    ('Qwen3_1.7B_FT_valtest',  'Qwen3 1.7B FT'),
]


def cargar_todos(input_dir):
    df_total = pd.DataFrame()
    archivos = glob.glob(os.path.join(input_dir, "metricas_por_fila_*.csv"))
    for archivo in archivos:
        nombre_base = os.path.basename(archivo).replace("metricas_por_fila_", "").replace(".csv", "")
        modelo = None
        for patron, m in ARCHIVO_MAPA:
            if nombre_base == patron:
                modelo = m
                break
        if modelo is None:
            continue
        try:
            df_temp = pd.read_csv(archivo)
            df_temp['Modelo'] = modelo
            df_total = pd.concat([df_total, df_temp], ignore_index=True)
        except Exception as e:
            print(f"  [ERROR] {archivo}: {e}")
    # Forzar columnas numéricas (algunos CSVs tienen strings)
    cols_num = ['ROUGE_L', 'METEOR', 'BERTScore', 'BLEU', 'Time_seconds',
                'Coste_USD', 'CO2_gramos', 'Tokens_Input', 'Tokens_Output']
    for c in cols_num:
        if c in df_total.columns:
            df_total[c] = pd.to_numeric(df_total[c], errors='coerce')

    return df_total


def hex_to_rgba(hex_color, alpha=0.25):
    h = hex_color.lstrip('#')
    r, g, b = tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))
    return (r, g, b, alpha)


def generar_barplots(df):
    col_configs = [
        ('Coste_USD',    'Coste Total (USD) — BBDD Teleco',      'USD',           '${:.4f}'),
        ('CO2_gramos',   'CO$_2$ Total (gramos) — BBDD Teleco', 'gramos CO$_2$', '{:.1f}g'),
        ('Time_seconds', 'Tiempo Total — BBDD Teleco',           'Segundos',      '{:.0f}s'),
    ]

    modelos_presentes = [m for m in ORDEN_MODELOS if m in df['Modelo'].unique()]
    if not modelos_presentes:
        print("No hay modelos.")
        return

    n_mod = len(modelos_presentes)
    bar_w = 0.8 / max(n_mod, 1)
    x = np.arange(1)  # Un solo "dataset" (Teleco)

    for col, titulo, ylabel, fmt in col_configs:
        if col not in df.columns:
            print(f"  Columna {col} no encontrada, saltando")
            continue

        resumen = df.groupby('Modelo')[col].sum().reset_index()
        resumen.rename(columns={col: 'valor'}, inplace=True)

        fig, ax = plt.subplots(figsize=(11, 6))
        max_val = resumen['valor'].max() if not resumen.empty else 1

        for k, m in enumerate(modelos_presentes):
            offset = (k - n_mod / 2 + 0.5) * bar_w
            v = resumen[resumen['Modelo'] == m]['valor']
            val = v.values[0] if not v.empty else np.nan

            c = MODELO_COLORES.get(m, '#888')
            bars = ax.bar(x + offset, [val], width=bar_w * 0.9,
                          color=hex_to_rgba(c, 0.4), edgecolor=c, linewidth=1.2, label=m)

            if pd.notna(val) and val > 0:
                ax.text(bars[0].get_x() + bars[0].get_width() / 2.,
                        bars[0].get_height() + max_val * 0.015,
                        fmt.format(val), ha='center', va='bottom', fontsize=7.5, rotation=0)

        n_filas = len(df[df['Modelo'] == modelos_presentes[0]]) if modelos_presentes else '?'
        ax.set_xticks(x)
        ax.set_xticklabels([f"Teleco\n({n_filas} datos)"], fontsize=12)

        ax.tick_params(axis='x', rotation=0, direction='in', length=7)
        ax.tick_params(axis='y', direction='in', length=7)

        ax.set_title(titulo, fontsize=13, fontweight='normal', pad=10)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_ylim(0, max_val * 1.30)

        ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='#888888', linewidth=0.5)
        ax.set_axisbelow(True)

        legend_patches = [
            mpatches.Patch(facecolor=hex_to_rgba(MODELO_COLORES[m], 0.4),
                           edgecolor=MODELO_COLORES[m], linewidth=1.2, label=m)
            for m in modelos_presentes
        ]
        ax.legend(handles=legend_patches, loc='upper right', fontsize=9,
                  framealpha=0.95, facecolor='white', edgecolor='black', fancybox=False)

        plt.tight_layout()
        nombre_png = f"teleco_Global_{col}.png"
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

    print("Cargando datos (BBDD Teleco — 7 modelos)...")
    df = cargar_todos(args.input_dir)
    if df.empty:
        print("Error: No se han cargado datos.")
        exit(1)

    print(f"  {len(df)} filas | {df['Modelo'].nunique()} modelos\n")
    print("Generando barplots globales teleco...")
    generar_barplots(df)
