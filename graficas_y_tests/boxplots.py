"""
Boxplots de las metricas de calidad a partir de los CSV por fila. Genera figuras
individuales por metrica y grids 2x2.

Grupo 1: 9 modelos sobre WebNLG, ToTTo y KELM.
Grupo 2: 7 modelos sobre Teleco (agrupados en GAN, base y fine-tuned).

Requisitos:
    pip install pandas matplotlib numpy

Uso:
    python boxplots.py --input-dir resultados --output-dir boxplots
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import glob, os, argparse

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'xtick.direction': 'in', 'ytick.direction': 'in',
})

# Paleta unificada
COLORES = {
    'DeepSeek':      '#2E75B6',
    'Llama 70B':     '#D6604D',
    'Qwen 72B':      '#E08214',
    'Gemma 9B':      '#4393C3',
    'Llama 3B':      '#7570B3',
    'Qwen 7B':       '#1B7837',
    'Gemma 3 1B':    '#E7298A',
    'Llama 1B':      '#66A61E',
    'Qwen3 1.7B':    '#D95F02',
    'GAN':           '#17BECF',
    'Gemma 3 1B FT': '#E67300',
    'Llama 1B FT':   '#8B0707',
    'Qwen3 1.7B FT': '#329262',
}

ORDEN_G1 = [
    'DeepSeek', 'Llama 70B', 'Qwen 72B',
    'Gemma 9B', 'Llama 3B', 'Qwen 7B',
    'Gemma 3 1B', 'Llama 1B', 'Qwen3 1.7B',
]
DATASETS_G1 = ['WebNLG', 'ToTTo', 'KELM']

ORDEN_G2 = [
    'GAN',
    'Gemma 3 1B', 'Llama 1B', 'Qwen3 1.7B',
    'Gemma 3 1B FT', 'Llama 1B FT', 'Qwen3 1.7B FT',
]
CATEGORIAS_G2 = ['GAN', 'Mini-SLM Base', 'Mini-SLM Fine-Tuned']
CATEGORIA_MODELOS = {
    'GAN':                 ['GAN'],
    'Mini-SLM Base':       ['Gemma 3 1B', 'Llama 1B', 'Qwen3 1.7B'],
    'Mini-SLM Fine-Tuned': ['Gemma 3 1B FT', 'Llama 1B FT', 'Qwen3 1.7B FT'],
}

METRICAS_CALIDAD = [
    ('ROUGE_L', 'ROUGE-L'), ('METEOR', 'METEOR'),
    ('BLEU', 'BLEU Score'), ('BERTScore', 'BERTScore'),
]

METRICAS_TODAS = [
    ('ROUGE_L',       'ROUGE-L',                       'Puntuación'),
    ('METEOR',        'METEOR',                        'Puntuación'),
    ('BERTScore',     'BERTScore',                     'Puntuación'),
    ('BLEU',          'BLEU Score',                    'Puntuación'),
    ('Time_seconds',  'Tiempo de Inferencia por Fila', 'Segundos'),
    ('Coste_USD',     'Coste por Fila',                'USD'),
    ('CO2_gramos',    'CO$_2$ por Fila',               'gramos CO$_2$'),
    ('Tokens_Input',  'Tokens de Entrada por Fila',    'Tokens'),
    ('Tokens_Output', 'Tokens de Salida por Fila',     'Tokens'),
]

ARCHIVO_MAPA = [
    ('webNLG_Gemma_3_1B',    'Gemma 3 1B', 'WebNLG'),
    ('webNLG_Llama_1B',      'Llama 1B',   'WebNLG'),
    ('webNLG_Qwen3_1.7B',   'Qwen3 1.7B', 'WebNLG'),
    ('totto_Gemma_3_1B',     'Gemma 3 1B', 'ToTTo'),
    ('totto_Llama_1B',       'Llama 1B',   'ToTTo'),
    ('totto_Qwen3_1.7B',    'Qwen3 1.7B', 'ToTTo'),
    ('kelm_stem_Gemma_3_1B', 'Gemma 3 1B', 'KELM'),
    ('kelm_stem_Llama_1B',   'Llama 1B',   'KELM'),
    ('kelm_stem_Qwen3_1.7B','Qwen3 1.7B', 'KELM'),
    ('teleco_Gemma_3_1B',    'Gemma 3 1B', 'Teleco'),
    ('teleco_Llama_1B',      'Llama 1B',   'Teleco'),
    ('teleco_Qwen3_1.7B',   'Qwen3 1.7B', 'Teleco'),
    ('GAN_valtest',          'GAN',            'Teleco'),
    ('Gemma_3_1B_FT_valtest','Gemma 3 1B FT', 'Teleco'),
    ('Llama_1B_FT_valtest',  'Llama 1B FT',   'Teleco'),
    ('Qwen3_1.7B_FT_valtest','Qwen3 1.7B FT', 'Teleco'),
    ('DeepSeek_WebNLG',  'DeepSeek', 'WebNLG'),
    ('DeepSeek_ToTTo',   'DeepSeek', 'ToTTo'),
    ('DeepSeek_KELM',    'DeepSeek', 'KELM'),
    ('Llama_70B_WebNLG', 'Llama 70B','WebNLG'),
    ('Llama_70B_ToTTo',  'Llama 70B','ToTTo'),
    ('Llama_70B_KELM',   'Llama 70B','KELM'),
    ('Qwen_72B_WebNLG',  'Qwen 72B', 'WebNLG'),
    ('Qwen_72B_ToTTo',   'Qwen 72B', 'ToTTo'),
    ('Qwen_72B_KELM',    'Qwen 72B', 'KELM'),
    ('Gemma_9B_WebNLG',  'Gemma 9B', 'WebNLG'),
    ('Gemma_9B_ToTTo',   'Gemma 9B', 'ToTTo'),
    ('Gemma_9B_KELM',    'Gemma 9B', 'KELM'),
    ('Llama_3B_WebNLG',  'Llama 3B', 'WebNLG'),
    ('Llama_3B_ToTTo',   'Llama 3B', 'ToTTo'),
    ('Llama_3B_KELM',    'Llama 3B', 'KELM'),
    ('Qwen_7B_WebNLG',   'Qwen 7B',  'WebNLG'),
    ('Qwen_7B_ToTTo',    'Qwen 7B',  'ToTTo'),
    ('Qwen_7B_KELM',     'Qwen 7B',  'KELM'),
]


# Utilidades
def hex_to_rgba(c, alpha=0.25):
    h = c.lstrip('#')
    r, g, b = (int(h[i:i+2], 16) / 255 for i in (0, 2, 4))
    return (r, g, b, alpha)


def cargar_todos(input_dir):
    df_total = pd.DataFrame()
    for archivo in glob.glob(os.path.join(input_dir, "metricas_por_fila_*.csv")):
        base = os.path.basename(archivo).replace("metricas_por_fila_", "").replace(".csv", "")
        modelo, dataset = None, None
        for pat, m, d in ARCHIVO_MAPA:
            if base == pat:
                modelo, dataset = m, d
                break
        if modelo is None:
            continue
        try:
            df = pd.read_csv(archivo)
            df['Modelo'] = modelo
            df['Dataset'] = dataset
            df_total = pd.concat([df_total, df], ignore_index=True)
        except Exception as e:
            print(f"  [ERROR] {archivo}: {e}")

    cols_num = ['ROUGE_L', 'METEOR', 'BLEU', 'BERTScore',
                'Time_seconds', 'Coste_USD', 'CO2_gramos',
                'Tokens_Input', 'Tokens_Output']
    for c in cols_num:
        if c in df_total.columns:
            df_total[c] = pd.to_numeric(df_total[c], errors='coerce')

    return df_total


# Dibujar boxplot en un Axes (reutilizable para individuales y grids)
def dibujar_boxplots_ax(ax, df, columna, modelos_por_grupo, group_labels,
                        box_w=0.012, gap=0.020, group_gap=0.08, hide_xlabel=False):
    """
    modelos_por_grupo: lista de listas de (modelo, dataset)
    group_labels: lista de strings para el eje X
    """
    current_x = 0
    group_centers, group_starts, group_ends = [], [], []
    gmin, gmax = np.inf, -np.inf

    for j, (mod_ds_list, label) in enumerate(zip(modelos_por_grupo, group_labels)):
        x_pos = []
        for i, (modelo, ds) in enumerate(mod_ds_list):
            subset = df[(df['Modelo'] == modelo) & (df['Dataset'] == ds)][columna].dropna()
            if len(subset) == 0:
                current_x += box_w + gap
                continue

            color = COLORES.get(modelo, '#888888')
            face = hex_to_rgba(color, 0.25)
            vals = subset.values

            ax.boxplot(
                [vals], positions=[current_x], widths=box_w,
                patch_artist=True, showfliers=False,
                medianprops=dict(color=color, linewidth=1.5),
                whiskerprops=dict(color='black', linewidth=0.7),
                capprops=dict(color='black', linewidth=0.7),
                boxprops=dict(facecolor=face, edgecolor=color, linewidth=1.1),
                zorder=3,
            )

            q1, q3 = np.percentile(vals, 25), np.percentile(vals, 75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            wmin = vals[vals >= lo].min() if np.any(vals >= lo) else q1
            wmax = vals[vals <= hi].max() if np.any(vals <= hi) else q3

            # Outliers seleccionados
            outliers = vals[(vals < lo) | (vals > hi)]
            drawn_max, drawn_min = wmax, wmin
            if len(outliers) > 0:
                if iqr > 0:
                    zona = outliers[
                        ((outliers < lo - 0.2 * iqr) & (outliers >= lo - 2.5 * iqr)) |
                        ((outliers > hi + 0.2 * iqr) & (outliers <= hi + 2.5 * iqr))
                    ]
                    if len(zona) < 4:
                        zona = outliers[
                            ((outliers < lo) & (outliers >= lo - 4.0 * iqr)) |
                            ((outliers > hi) & (outliers <= hi + 4.0 * iqr))
                        ]
                    if len(zona) == 0:
                        zona = outliers
                else:
                    zona = outliers
                np.random.seed(42 + i * 10 + j)
                sel = np.random.choice(zona, size=min(4, len(zona)), replace=False)
                ax.plot([current_x] * len(sel), sel,
                        marker='o', linestyle='none', markerfacecolor='none',
                        markeredgecolor='#555555', markeredgewidth=0.8,
                        markersize=4, alpha=0.7, zorder=2)
                drawn_max = max(drawn_max, max(sel))
                drawn_min = min(drawn_min, min(sel))

            gmax = max(gmax, drawn_max)
            gmin = min(gmin, drawn_min)
            x_pos.append(current_x)
            current_x += box_w + gap

        if x_pos:
            ext = (box_w / 2) + ((gap + group_gap) / 2)
            group_centers.append((x_pos[0] + x_pos[-1]) / 2)
            group_starts.append(x_pos[0] - ext)
            group_ends.append(x_pos[-1] + ext)
            current_x += group_gap

    # Bandas grises alternas
    for j in range(len(group_starts)):
        if j % 2 == 0:
            ax.axvspan(group_starts[j], group_ends[j], alpha=0.4, color='#E5E5E5', zorder=0, lw=0)

    ax.set_xticks(group_centers)
    if hide_xlabel:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', direction='in', length=0)
    else:
        ax.set_xticklabels(group_labels, fontsize=12)
        ax.tick_params(axis='x', direction='in', length=7)
    ax.tick_params(axis='y', direction='in', length=7)
    ax.xaxis.grid(True, which='both', linestyle='-', alpha=0.3, color='#888888', linewidth=0.5)
    ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='#888888', linewidth=0.5)
    ax.set_axisbelow(True)
    if group_starts:
        ax.set_xlim(group_starts[0], group_ends[-1])
    rango = (gmax - gmin) if gmax > gmin else 1
    ax.set_ylim(gmin - rango * 0.05, gmax + rango * 0.05)


# Helpers para construir modelos_por_grupo
def grupos_g1(modelos, datasets):
    """Un grupo por dataset, todos los modelos dentro."""
    return (
        [[(m, ds) for m in modelos] for ds in datasets],
        datasets,
    )


def grupos_g2(modelos_presentes):
    """Un grupo por categoría (GAN / Base / FT)."""
    labels, groups = [], []
    for cat in CATEGORIAS_G2:
        mods = [m for m in CATEGORIA_MODELOS[cat] if m in modelos_presentes]
        if mods:
            groups.append([(m, 'Teleco') for m in mods])
            labels.append(cat)
    return groups, labels


def boxplots_individuales(df, orden, datasets, colores_ref, prefijo, output_dir, es_teleco=False):
    modelos = [m for m in orden if m in df['Modelo'].unique()]
    ds_presentes = [d for d in datasets if d in df['Dataset'].unique()]
    if not modelos or not ds_presentes:
        return

    if es_teleco:
        mod_groups, labels = grupos_g2(modelos)
        xlabel = 'Tipo de Modelo'
    else:
        mod_groups, labels = grupos_g1(modelos, ds_presentes)
        xlabel = 'Dataset'

    for col, titulo, ylabel in METRICAS_TODAS:
        if col not in df.columns:
            continue

        sufijo = " — BBDD Teleco" if es_teleco else ""
        fig, ax = plt.subplots(figsize=(11 if es_teleco else 13, 6))
        dibujar_boxplots_ax(ax, df, col, mod_groups, labels)
        ax.set_title(titulo + sufijo, fontsize=13, fontweight='normal', pad=10)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel(xlabel, fontsize=12)

        legend_patches = [
            mpatches.Patch(facecolor=hex_to_rgba(colores_ref.get(m, '#888'), 0.25),
                           edgecolor=colores_ref.get(m, '#888'), linewidth=1.1, label=m)
            for m in modelos
        ]
        pos = 'lower left' if col == 'BERTScore' else 'upper right'
        ax.legend(handles=legend_patches, loc=pos, ncol=1,
                  fontsize=8 if len(modelos) > 5 else 9,
                  framealpha=0.95, facecolor='white', edgecolor='black', fancybox=False)

        plt.tight_layout()
        fname = os.path.join(output_dir, f"{prefijo}_{col}.png")
        plt.savefig(fname, dpi=300, facecolor='white')
        plt.close()
        print(f"  {fname}")


def grid_2x2(df, mod_groups, labels, modelos_leyenda, titulo_fig, fname):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11),
                             gridspec_kw={'hspace': 0.25, 'wspace': 0.28})
    fig.suptitle(titulo_fig, fontsize=17, fontweight='bold', y=0.98)
    axes_flat = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]

    for ax, (col, nombre) in zip(axes_flat, METRICAS_CALIDAD):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        dibujar_boxplots_ax(ax, df, col, mod_groups, labels,
                            group_gap=0.04 if len(labels) <= 3 else 0.08,
                            box_w=0.012, gap=0.018, hide_xlabel=True)
        ax.set_title(nombre, fontsize=14, pad=8)

    legend_patches = [
        mpatches.Patch(facecolor=hex_to_rgba(COLORES.get(m, '#888'), 0.25),
                       edgecolor=COLORES.get(m, '#888'), linewidth=1.1, label=m)
        for m in modelos_leyenda
    ]
    fig.legend(handles=legend_patches, loc='lower center',
               bbox_to_anchor=(0.5, 0.0), ncol=len(legend_patches),
               fontsize=12, framealpha=0.95, edgecolor='#CCCCCC',
               title='Modelo', title_fontsize=12,
               handlelength=1.5, handleheight=1.2, borderpad=0.5,
               labelspacing=0.4, columnspacing=0.9)

    plt.subplots_adjust(bottom=0.10)
    plt.savefig(fname, dpi=200, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"  {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="resultados")
    parser.add_argument("--output-dir", default="boxplots")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, args.input_dir) if not os.path.isabs(args.input_dir) else args.input_dir
    output_dir = os.path.join(script_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("Cargando datos...")
    df = cargar_todos(input_dir)
    if df.empty:
        print("Error: No se han cargado datos.")
        exit(1)
    print(f"  {len(df):,} filas | {df['Modelo'].nunique()} modelos | {df['Dataset'].nunique()} datasets\n")

    # G1: individuales
    df_g1 = df[df['Dataset'].isin(DATASETS_G1)]
    if not df_g1.empty:
        print("Boxplots individuales G1 (9 modelos × 3 datasets)...")
        boxplots_individuales(df_g1, ORDEN_G1, DATASETS_G1, COLORES, "G1", output_dir)

    # G2: individuales
    df_g2 = df[df['Dataset'] == 'Teleco']
    if not df_g2.empty:
        print("\nBoxplots individuales G2 (7 modelos × Teleco)...")
        boxplots_individuales(df_g2, ORDEN_G2, ['Teleco'], COLORES, "G2_Teleco", output_dir, es_teleco=True)

    # Grids 2×2 por dataset (G1)
    modelos_g1 = [m for m in ORDEN_G1 if m in df['Modelo'].unique()]
    if modelos_g1:
        print("\nGrids 2×2 por dataset (G1)...")
        for ds in DATASETS_G1:
            if ds not in df['Dataset'].unique():
                continue
            mg, lb = grupos_g1(modelos_g1, [ds])
            fname = os.path.join(output_dir, f"grid_{ds}.png")
            grid_2x2(df, mg, lb, modelos_g1, f"Dataset: {ds}", fname)

    # Grid 2×2 Teleco (G2)
    modelos_g2 = [m for m in ORDEN_G2 if m in df['Modelo'].unique()]
    if modelos_g2:
        print("\nGrid 2×2 Teleco (G2)...")
        mg, lb = grupos_g2(modelos_g2)
        fname = os.path.join(output_dir, "grid_Teleco.png")
        grid_2x2(df, mg, lb, modelos_g2, "Dataset: Teleco", fname)

    print("\nListo.")