# pip install pandas matplotlib numpy
"""
Boxplots agrupados estilo paper — 9 modelos.
6 originales (LLM+SLM) + 3 mini-SLMs (Gemma 3.1B, Llama 1B, Qwen3 1.7B).
- Misma estética que los boxplots originales.
- Parsing inteligente: detecta ambos formatos de nombre de archivo.

USO: python boxplots_9modelos.py --input-dir analisis
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
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

# --- 9 modelos: 3 LLMs + 3 SLMs + 3 mini-SLMs ---
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

# ── Mapeo explícito: fragmento de nombre de archivo → (modelo, dataset) ──
# Formato original: metricas_por_fila_{Modelo}_{Dataset}.csv
# Formato invertido: metricas_por_fila_{dataset}_{Modelo}.csv
# Ponemos primero los más específicos para evitar colisiones.
ARCHIVO_MAPA = [
    # --- Mini-SLMs (formato invertido: dataset_modelo) ---
    ('webNLG_Gemma_3_1B',       'Gemma 3.1B',  'WebNLG'),
    ('webNLG_Llama_1B',         'Llama 1B',    'WebNLG'),
    ('webNLG_Qwen3_1.7B',      'Qwen3 1.7B',  'WebNLG'),
    ('totto_Gemma_3_1B',        'Gemma 3.1B',  'ToTTo'),
    ('totto_Llama_1B',          'Llama 1B',    'ToTTo'),
    ('totto_Qwen3_1.7B',       'Qwen3 1.7B',  'ToTTo'),
    ('kelm_stem_Gemma_3_1B',   'Gemma 3.1B',  'KELM'),
    ('kelm_stem_Llama_1B',     'Llama 1B',    'KELM'),
    ('kelm_stem_Qwen3_1.7B',  'Qwen3 1.7B',  'KELM'),
    # --- Originales (formato: Modelo_Dataset) ---
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
    """Carga todos los CSVs de métricas por fila usando el mapeo explícito."""
    df_total = pd.DataFrame()
    archivos = glob.glob(os.path.join(input_dir, "metricas_por_fila_*.csv"))

    for archivo in archivos:
        nombre_base = os.path.basename(archivo).replace("metricas_por_fila_", "").replace(".csv", "")

        modelo = None
        dataset = None
        for patron, m, d in ARCHIVO_MAPA:
            if nombre_base == patron:
                modelo = m
                dataset = d
                break

        if modelo is None:
            print(f"  [WARN] Archivo no mapeado, saltando: {os.path.basename(archivo)}")
            continue

        try:
            df_temp = pd.read_csv(archivo)
            df_temp['Modelo'] = modelo
            df_temp['Dataset'] = dataset
            df_total = pd.concat([df_total, df_temp], ignore_index=True)
        except Exception as e:
            print(f"  [ERROR] {archivo}: {e}")

    return df_total


def hex_to_rgba(hex_color, alpha=0.3):
    h = hex_color.lstrip('#')
    r, g, b = tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))
    return (r, g, b, alpha)


def generar_boxplot(df, columna, titulo, ylabel, nombre_archivo):
    if columna not in df.columns:
        print(f"  Columna {columna} no existe, saltando")
        return

    modelos = [m for m in ORDEN_MODELOS if m in df['Modelo'].unique()]
    datasets = [d for d in DATASETS if d in df['Dataset'].unique()]
    if not modelos or not datasets:
        return

    box_w = 0.012
    gap = 0.020       # Un poco más apretado para que quepan 9
    extra_group_gap = 0.08

    fig, ax = plt.subplots(figsize=(13, 6))  # Más ancho para 9 modelos

    current_x = 0
    group_centers = []
    group_starts = []
    group_ends = []

    global_max_drawn = -np.inf
    global_min_drawn = np.inf

    for j, ds in enumerate(datasets):
        x_positions_in_ds = []

        for i, modelo in enumerate(modelos):
            subset = df[(df['Modelo'] == modelo) & (df['Dataset'] == ds)][columna].dropna()
            if len(subset) == 0:
                current_x += box_w + gap
                continue

            color = MODELO_COLORES.get(modelo, '#888888')
            face = hex_to_rgba(color, 0.25)
            subset_vals = subset.values

            bp = ax.boxplot(
                [subset_vals], positions=[current_x], widths=box_w,
                patch_artist=True, showfliers=False,
                medianprops=dict(color=color, linewidth=1.5),
                whiskerprops=dict(color='black', linewidth=0.7),
                capprops=dict(color='black', linewidth=0.7),
                boxprops=dict(facecolor=face, edgecolor=color, linewidth=1.1),
                zorder=3,
            )

            q1 = np.percentile(subset_vals, 25)
            q3 = np.percentile(subset_vals, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            actual_min_w = np.min(subset_vals[subset_vals >= lower_bound]) if np.any(subset_vals >= lower_bound) else q1
            actual_max_w = np.max(subset_vals[subset_vals <= upper_bound]) if np.any(subset_vals <= upper_bound) else q3

            box_max_drawn = actual_max_w
            box_min_drawn = actual_min_w

            outliers = subset_vals[(subset_vals < lower_bound) | (subset_vals > upper_bound)]
            if len(outliers) > 0:
                if iqr > 0:
                    zona_ideal = outliers[
                        ((outliers < lower_bound - 0.2 * iqr) & (outliers >= lower_bound - 2.5 * iqr)) |
                        ((outliers > upper_bound + 0.2 * iqr) & (outliers <= upper_bound + 2.5 * iqr))
                    ]
                    if len(zona_ideal) < 4:
                        zona_ideal = outliers[
                            ((outliers < lower_bound) & (outliers >= lower_bound - 4.0 * iqr)) |
                            ((outliers > upper_bound) & (outliers <= upper_bound + 4.0 * iqr))
                        ]
                    if len(zona_ideal) == 0:
                        zona_ideal = outliers
                else:
                    zona_ideal = outliers

                np.random.seed(42 + i * 10 + j)
                n_out = min(4, len(zona_ideal))
                selected = np.random.choice(zona_ideal, size=n_out, replace=False)
                ax.plot([current_x] * len(selected), selected,
                        marker='o', linestyle='none', markerfacecolor='none',
                        markeredgecolor='#555555', markeredgewidth=0.8,
                        markersize=4, alpha=0.7, zorder=2)
                box_max_drawn = max(box_max_drawn, max(selected))
                box_min_drawn = min(box_min_drawn, min(selected))

            global_max_drawn = max(global_max_drawn, box_max_drawn)
            global_min_drawn = min(global_min_drawn, box_min_drawn)

            x_positions_in_ds.append(current_x)
            current_x += box_w + gap

        if x_positions_in_ds:
            ds_start = x_positions_in_ds[0]
            ds_end = x_positions_in_ds[-1]
            group_centers.append((ds_start + ds_end) / 2.0)
            extension = (box_w / 2) + ((gap + extra_group_gap) / 2)
            group_starts.append(ds_start - extension)
            group_ends.append(ds_end + extension)
            current_x += extra_group_gap

    # Bandas grises
    for j in range(len(group_starts)):
        if j % 2 == 0:
            ax.axvspan(group_starts[j], group_ends[j], alpha=0.4, color='#E5E5E5', zorder=0, lw=0)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.set_xlabel('Dataset', fontsize=12)
    ax.tick_params(axis='x', which='major', direction='in', length=7)
    ax.tick_params(axis='x', which='minor', bottom=False, top=False)
    ax.tick_params(axis='y', which='major', direction='in', length=7)
    ax.tick_params(axis='y', which='minor', left=False, right=False)

    ax.set_title(titulo, fontsize=13, fontweight='normal', pad=10)
    ax.set_ylabel(ylabel, fontsize=12)

    if group_starts:
        ax.set_xlim(group_starts[0], group_ends[-1])

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.xaxis.grid(True, which='both', linestyle='-', alpha=0.3, color='#888888', linewidth=0.5)
    ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='#888888', linewidth=0.5)
    ax.set_axisbelow(True)

    rango = global_max_drawn - global_min_drawn
    if rango <= 0:
        rango = 1
    margin_y = rango * 0.05
    ax.set_ylim(global_min_drawn - margin_y, global_max_drawn + margin_y)

    # Leyenda
    legend_patches = [
        mpatches.Patch(facecolor=hex_to_rgba(MODELO_COLORES[m], 0.25),
                       edgecolor=MODELO_COLORES[m], linewidth=1.1, label=m)
        for m in modelos
    ]
    posicion = 'lower left' if columna == 'BERTScore' else 'upper right'
    ax.legend(handles=legend_patches, loc=posicion, ncol=1, fontsize=8,
              framealpha=0.95, facecolor='white', edgecolor='black',
              fancybox=False, handlelength=1.5, handleheight=1.0)

    plt.tight_layout()
    plt.savefig(f"{nombre_archivo}.png", dpi=300, facecolor='white')
    plt.close()
    print(f"  {nombre_archivo}.png")


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

    generar_boxplot(df, 'ROUGE_L',       'ROUGE-L',                       'Puntuación',    'agrupado_9m_ROUGE_L')
    generar_boxplot(df, 'METEOR',        'METEOR',                        'Puntuación',    'agrupado_9m_METEOR')
    generar_boxplot(df, 'BERTScore',     'BERTScore',                     'Puntuación',    'agrupado_9m_BERTScore')
    generar_boxplot(df, 'BLEU',          'BLEU Score',                    'Puntuación',    'agrupado_9m_BLEU')
    generar_boxplot(df, 'Time_seconds',  'Tiempo de Inferencia por Fila', 'Segundos',      'agrupado_9m_Tiempo')
    generar_boxplot(df, 'Coste_USD',     'Coste por Fila',                'USD',           'agrupado_9m_Coste_USD')
    generar_boxplot(df, 'CO2_gramos',    'CO$_2$ por Fila',              'gramos CO$_2$', 'agrupado_9m_CO2')
    generar_boxplot(df, 'Tokens_Input',  'Tokens de Entrada por Fila',    'Tokens',        'agrupado_9m_Tokens_Input')
    generar_boxplot(df, 'Tokens_Output', 'Tokens de Salida por Fila',     'Tokens',        'agrupado_9m_Tokens_Output')
