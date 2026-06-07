# pip install pandas matplotlib numpy
"""
Violinplots agrupados estilo paper — 9 modelos.
6 originales (LLM+SLM) + 3 mini-SLMs (Gemma 3.1B, Llama 1B, Qwen3 1.7B).
- Diseño idéntico a las boxplots (fondos, colores, separaciones compactas).
- Filtra outliers extremos (IQR×3) ANTES de dibujar para evitar ejes disparados.
- ylim basado en IQR global, nunca negativo.

USO: python violinplots_9modelos.py --input-dir analisis
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

    cols_num = ['ROUGE_L', 'METEOR', 'BERTScore', 'BLEU', 'Time_seconds',
                'Coste_USD', 'CO2_gramos', 'Tokens_Input', 'Tokens_Output']
    for c in cols_num:
        if c in df_total.columns:
            df_total[c] = pd.to_numeric(df_total[c], errors='coerce')

    return df_total


def hex_to_rgba(hex_color, alpha=0.3):
    h = hex_color.lstrip('#')
    r, g, b = tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))
    return (r, g, b, alpha)


def filtrar_iqr(vals, factor=3.0):
    """Filtra outliers extremos para que el violin no se deforme."""
    q1 = np.percentile(vals, 25)
    q3 = np.percentile(vals, 75)
    iqr = q3 - q1
    if iqr > 0:
        lb = q1 - factor * iqr
        ub = q3 + factor * iqr
        mask = (vals >= lb) & (vals <= ub)
        filtered = vals[mask]
        if len(filtered) >= 10:
            return filtered
    return vals


def generar_violinplot(df, columna, titulo, ylabel, nombre_archivo):
    if columna not in df.columns:
        print(f"  Columna {columna} no existe, saltando")
        return

    modelos = [m for m in ORDEN_MODELOS if m in df['Modelo'].unique()]
    datasets = [d for d in DATASETS if d in df['Dataset'].unique()]
    if not modelos or not datasets:
        return

    # Mismas proporciones que los boxplots originales
    box_w = 0.012
    gap = 0.025
    extra_group_gap = 0.08

    fig, ax = plt.subplots(figsize=(13, 6))

    current_x = 0
    group_centers = []
    group_starts = []
    group_ends = []

    all_whisker_min = []
    all_whisker_max = []

    for j, ds in enumerate(datasets):
        x_positions_in_ds = []

        for i, modelo in enumerate(modelos):
            subset = df[(df['Modelo'] == modelo) & (df['Dataset'] == ds)][columna].dropna()
            if len(subset) == 0:
                current_x += box_w + gap
                continue

            color = MODELO_COLORES.get(modelo, '#888888')
            # Filtrar outliers extremos ANTES de dibujar
            vals_plot = filtrar_iqr(subset.values)

            # Calcular whiskers como en boxplots para el ylim
            q1 = np.percentile(vals_plot, 25)
            q3 = np.percentile(vals_plot, 75)
            iqr = q3 - q1
            lb = q1 - 1.5 * iqr
            ub = q3 + 1.5 * iqr
            w_min = np.min(vals_plot[vals_plot >= lb]) if np.any(vals_plot >= lb) else q1
            w_max = np.max(vals_plot[vals_plot <= ub]) if np.any(vals_plot <= ub) else q3
            all_whisker_min.append(w_min)
            all_whisker_max.append(w_max)

            vp = ax.violinplot(
                vals_plot,
                positions=[current_x],
                widths=box_w * 1.2,
                showmeans=False,
                showmedians=True,
                showextrema=False,
            )

            for pc in vp['bodies']:
                pc.set_facecolor(hex_to_rgba(color, 0.4))
                pc.set_edgecolor(color)
                pc.set_linewidth(1.1)
                pc.set_alpha(0.6)

            vp['cmedians'].set_edgecolor(color)
            vp['cmedians'].set_linewidth(1.8)

            # Whiskers en min/max real de los datos filtrados
            v_min = vals_plot.min()
            v_max = vals_plot.max()
            half_w = box_w * 0.3
            ax.plot([current_x, current_x], [v_min, v_max],
                    color=color, linewidth=0.8, alpha=0.7, zorder=2)
            ax.plot([current_x - half_w, current_x + half_w], [v_min, v_min],
                    color=color, linewidth=0.8, alpha=0.7, zorder=2)
            ax.plot([current_x - half_w, current_x + half_w], [v_max, v_max],
                    color=color, linewidth=0.8, alpha=0.7, zorder=2)

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

    # ylim idéntico a boxplots: basado en whiskers globales con 5% margen
    # Métricas de puntuación siempre van de 0 a 1
    METRICAS_0_1 = {'ROUGE_L', 'METEOR', 'BERTScore', 'BLEU'}
    if columna in METRICAS_0_1:
        ax.set_ylim(0, 1.05)
    elif all_whisker_min and all_whisker_max:
        global_min = min(all_whisker_min)
        global_max = max(all_whisker_max)
        rango = global_max - global_min
        if rango <= 0:
            rango = max(abs(global_max), 1)
        margin_y = rango * 0.05
        y_low = max(0, global_min - margin_y)
        y_high = global_max + margin_y
        ax.set_ylim(y_low, y_high)

    # Leyenda
    legend_patches = [
        mpatches.Patch(facecolor=hex_to_rgba(MODELO_COLORES[m], 0.4),
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

    generar_violinplot(df, 'ROUGE_L',       'ROUGE-L',                       'Puntuación',    'violin_9m_ROUGE_L')
    generar_violinplot(df, 'METEOR',        'METEOR',                        'Puntuación',    'violin_9m_METEOR')
    generar_violinplot(df, 'BERTScore',     'BERTScore',                     'Puntuación',    'violin_9m_BERTScore')
    generar_violinplot(df, 'BLEU',          'BLEU Score',                    'Puntuación',    'violin_9m_BLEU')
    generar_violinplot(df, 'Time_seconds',  'Tiempo de Inferencia por Fila', 'Segundos',      'violin_9m_Tiempo')
    generar_violinplot(df, 'Coste_USD',     'Coste por Fila',                'USD',           'violin_9m_Coste_USD')
    generar_violinplot(df, 'CO2_gramos',    'CO$_2$ por Fila',              'gramos CO$_2$', 'violin_9m_CO2')
    generar_violinplot(df, 'Tokens_Input',  'Tokens de Entrada por Fila',    'Tokens',        'violin_9m_Tokens_Input')
    generar_violinplot(df, 'Tokens_Output', 'Tokens de Salida por Fila',     'Tokens',        'violin_9m_Tokens_Output')