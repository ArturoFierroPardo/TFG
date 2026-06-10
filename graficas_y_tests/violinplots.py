# pip install pandas matplotlib numpy
"""
Violinplots unificados — G1 (9 modelos × 3 datasets) + G2 (7 modelos × Teleco).
Filtra outliers extremos (IQR×3) antes de dibujar.

USO: python violinplots.py --input-dir resultados --output-dir violinplots
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import glob, os, argparse
import numpy as np

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'xtick.direction': 'in', 'ytick.direction': 'in',
})

# ── Paleta unificada ──────────────────────────────────────────────────────
COLORES = {
    'DeepSeek': '#2E75B6', 'Llama 70B': '#D6604D', 'Qwen 72B': '#E08214',
    'Gemma 9B': '#4393C3', 'Llama 3B': '#7570B3', 'Qwen 7B': '#1B7837',
    'Gemma 3 1B': '#E7298A', 'Llama 1B': '#66A61E', 'Qwen3 1.7B': '#D95F02',
    'GAN': '#E41A1C',
    'Gemma 3 1B FT': '#984EA3', 'Llama 1B FT': '#377EB8', 'Qwen3 1.7B FT': '#FF7F00',
}

ORDEN_G1 = ['DeepSeek', 'Llama 70B', 'Qwen 72B',
            'Gemma 9B', 'Llama 3B', 'Qwen 7B',
            'Gemma 3 1B', 'Llama 1B', 'Qwen3 1.7B']
DATASETS_G1 = ['WebNLG', 'ToTTo', 'KELM']

ORDEN_G2 = ['GAN', 'Gemma 3 1B', 'Llama 1B', 'Qwen3 1.7B',
            'Gemma 3 1B FT', 'Llama 1B FT', 'Qwen3 1.7B FT']
CATEGORIAS_G2 = ['GAN', 'Mini-SLM Base', 'Mini-SLM Fine-Tuned']
CATEGORIA_MODELOS = {
    'GAN': ['GAN'],
    'Mini-SLM Base': ['Gemma 3 1B', 'Llama 1B', 'Qwen3 1.7B'],
    'Mini-SLM Fine-Tuned': ['Gemma 3 1B FT', 'Llama 1B FT', 'Qwen3 1.7B FT'],
}

METRICAS_TODAS = [
    ('ROUGE_L', 'ROUGE-L', 'Puntuación'),
    ('METEOR', 'METEOR', 'Puntuación'),
    ('BERTScore', 'BERTScore', 'Puntuación'),
    ('BLEU', 'BLEU Score', 'Puntuación'),
    ('Time_seconds', 'Tiempo de Inferencia por Fila', 'Segundos'),
    ('Coste_USD', 'Coste por Fila', 'USD'),
    ('CO2_gramos', 'CO$_2$ por Fila', 'gramos CO$_2$'),
    ('Tokens_Input', 'Tokens de Entrada por Fila', 'Tokens'),
    ('Tokens_Output', 'Tokens de Salida por Fila', 'Tokens'),
]
METRICAS_0_1 = {'ROUGE_L', 'METEOR', 'BERTScore', 'BLEU'}

ARCHIVO_MAPA = [
    ('webNLG_Gemma_3_1B', 'Gemma 3 1B', 'WebNLG'),
    ('webNLG_Llama_1B', 'Llama 1B', 'WebNLG'),
    ('webNLG_Qwen3_1.7B', 'Qwen3 1.7B', 'WebNLG'),
    ('totto_Gemma_3_1B', 'Gemma 3 1B', 'ToTTo'),
    ('totto_Llama_1B', 'Llama 1B', 'ToTTo'),
    ('totto_Qwen3_1.7B', 'Qwen3 1.7B', 'ToTTo'),
    ('kelm_stem_Gemma_3_1B', 'Gemma 3 1B', 'KELM'),
    ('kelm_stem_Llama_1B', 'Llama 1B', 'KELM'),
    ('kelm_stem_Qwen3_1.7B', 'Qwen3 1.7B', 'KELM'),
    ('teleco_Gemma_3_1B', 'Gemma 3 1B', 'Teleco'),
    ('teleco_Llama_1B', 'Llama 1B', 'Teleco'),
    ('teleco_Qwen3_1.7B', 'Qwen3 1.7B', 'Teleco'),
    ('GAN_valtest', 'GAN', 'Teleco'),
    ('Gemma_3_1B_FT_valtest', 'Gemma 3 1B FT', 'Teleco'),
    ('Llama_1B_FT_valtest', 'Llama 1B FT', 'Teleco'),
    ('Qwen3_1.7B_FT_valtest', 'Qwen3 1.7B FT', 'Teleco'),
    ('DeepSeek_WebNLG', 'DeepSeek', 'WebNLG'), ('DeepSeek_ToTTo', 'DeepSeek', 'ToTTo'), ('DeepSeek_KELM', 'DeepSeek', 'KELM'),
    ('Llama_70B_WebNLG', 'Llama 70B', 'WebNLG'), ('Llama_70B_ToTTo', 'Llama 70B', 'ToTTo'), ('Llama_70B_KELM', 'Llama 70B', 'KELM'),
    ('Qwen_72B_WebNLG', 'Qwen 72B', 'WebNLG'), ('Qwen_72B_ToTTo', 'Qwen 72B', 'ToTTo'), ('Qwen_72B_KELM', 'Qwen 72B', 'KELM'),
    ('Gemma_9B_WebNLG', 'Gemma 9B', 'WebNLG'), ('Gemma_9B_ToTTo', 'Gemma 9B', 'ToTTo'), ('Gemma_9B_KELM', 'Gemma 9B', 'KELM'),
    ('Llama_3B_WebNLG', 'Llama 3B', 'WebNLG'), ('Llama_3B_ToTTo', 'Llama 3B', 'ToTTo'), ('Llama_3B_KELM', 'Llama 3B', 'KELM'),
    ('Qwen_7B_WebNLG', 'Qwen 7B', 'WebNLG'), ('Qwen_7B_ToTTo', 'Qwen 7B', 'ToTTo'), ('Qwen_7B_KELM', 'Qwen 7B', 'KELM'),
]


# ── Utilidades ────────────────────────────────────────────────────────────
def hex_to_rgba(c, alpha=0.3):
    h = c.lstrip('#')
    r, g, b = (int(h[i:i+2], 16) / 255 for i in (0, 2, 4))
    return (r, g, b, alpha)


def filtrar_iqr(vals, factor=3.0):
    q1, q3 = np.percentile(vals, 25), np.percentile(vals, 75)
    iqr = q3 - q1
    if iqr > 0:
        filtered = vals[(vals >= q1 - factor * iqr) & (vals <= q3 + factor * iqr)]
        if len(filtered) >= 10:
            return filtered
    return vals


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

    cols_num = ['ROUGE_L', 'METEOR', 'BERTScore', 'BLEU', 'Time_seconds',
                'Coste_USD', 'CO2_gramos', 'Tokens_Input', 'Tokens_Output']
    for c in cols_num:
        if c in df_total.columns:
            df_total[c] = pd.to_numeric(df_total[c], errors='coerce')
    return df_total


# ── Dibujar violins en un Axes ────────────────────────────────────────────
def dibujar_violins(ax, df, columna, modelos_por_grupo, group_labels,
                    box_w=0.012, gap=0.025, group_gap=0.08):
    current_x = 0
    group_centers, group_starts, group_ends = [], [], []
    all_wmin, all_wmax = [], []

    for j, (mod_ds_list, label) in enumerate(zip(modelos_por_grupo, group_labels)):
        x_pos = []
        for i, (modelo, ds) in enumerate(mod_ds_list):
            subset = df[(df['Modelo'] == modelo) & (df['Dataset'] == ds)][columna].dropna()
            if len(subset) == 0:
                current_x += box_w + gap
                continue

            color = COLORES.get(modelo, '#888')
            vals = filtrar_iqr(subset.values)

            q1, q3 = np.percentile(vals, 25), np.percentile(vals, 75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            wmin = vals[vals >= lo].min() if np.any(vals >= lo) else q1
            wmax = vals[vals <= hi].max() if np.any(vals <= hi) else q3
            all_wmin.append(wmin)
            all_wmax.append(wmax)

            vp = ax.violinplot(vals, positions=[current_x], widths=box_w * 1.2,
                               showmeans=False, showmedians=True, showextrema=False)

            for pc in vp['bodies']:
                pc.set_facecolor(hex_to_rgba(color, 0.4))
                pc.set_edgecolor(color)
                pc.set_linewidth(1.1)
                pc.set_alpha(0.6)
            vp['cmedians'].set_edgecolor(color)
            vp['cmedians'].set_linewidth(1.8)

            # Whiskers
            v_min, v_max = vals.min(), vals.max()
            hw = box_w * 0.3
            ax.plot([current_x, current_x], [v_min, v_max], color=color, lw=0.8, alpha=0.7, zorder=2)
            ax.plot([current_x - hw, current_x + hw], [v_min, v_min], color=color, lw=0.8, alpha=0.7, zorder=2)
            ax.plot([current_x - hw, current_x + hw], [v_max, v_max], color=color, lw=0.8, alpha=0.7, zorder=2)

            x_pos.append(current_x)
            current_x += box_w + gap

        if x_pos:
            ext = (box_w / 2) + ((gap + group_gap) / 2)
            group_centers.append((x_pos[0] + x_pos[-1]) / 2)
            group_starts.append(x_pos[0] - ext)
            group_ends.append(x_pos[-1] + ext)
            current_x += group_gap

    for j in range(len(group_starts)):
        if j % 2 == 0:
            ax.axvspan(group_starts[j], group_ends[j], alpha=0.4, color='#E5E5E5', zorder=0, lw=0)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(group_labels, fontsize=12)
    ax.tick_params(axis='x', direction='in', length=7)
    ax.tick_params(axis='y', direction='in', length=7)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.xaxis.grid(True, which='both', linestyle='-', alpha=0.3, color='#888888', linewidth=0.5)
    ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='#888888', linewidth=0.5)
    ax.set_axisbelow(True)
    if group_starts:
        ax.set_xlim(group_starts[0], group_ends[-1])

    if columna in METRICAS_0_1:
        ax.set_ylim(0, 1.05)
    elif all_wmin and all_wmax:
        gmin, gmax = min(all_wmin), max(all_wmax)
        rango = gmax - gmin if gmax > gmin else max(abs(gmax), 1)
        ax.set_ylim(max(0, gmin - rango * 0.05), gmax + rango * 0.05)


# ── Helpers ───────────────────────────────────────────────────────────────
def grupos_g1(modelos, datasets):
    return [[(m, ds) for m in modelos] for ds in datasets], datasets

def grupos_g2(modelos_presentes):
    labels, groups = [], []
    for cat in CATEGORIAS_G2:
        mods = [m for m in CATEGORIA_MODELOS[cat] if m in modelos_presentes]
        if mods:
            groups.append([(m, 'Teleco') for m in mods])
            labels.append(cat)
    return groups, labels


# ── Generador genérico ────────────────────────────────────────────────────
def generar_violins(df, orden, datasets, prefijo, output_dir, es_teleco=False):
    modelos = [m for m in orden if m in df['Modelo'].unique()]
    ds_pres = [d for d in datasets if d in df['Dataset'].unique()]
    if not modelos or not ds_pres:
        return

    if es_teleco:
        mod_groups, labels = grupos_g2(modelos)
        xlabel = 'Tipo de Modelo'
    else:
        mod_groups, labels = grupos_g1(modelos, ds_pres)
        xlabel = 'Dataset'

    for col, titulo, ylabel in METRICAS_TODAS:
        if col not in df.columns:
            continue

        sufijo = " — BBDD Teleco" if es_teleco else ""
        fig, ax = plt.subplots(figsize=(11 if es_teleco else 13, 6))
        dibujar_violins(ax, df, col, mod_groups, labels,
                        box_w=0.015 if es_teleco else 0.012,
                        gap=0.025, group_gap=0.10 if es_teleco else 0.08)
        ax.set_title(titulo + sufijo, fontsize=13, fontweight='normal', pad=10)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel(xlabel, fontsize=12)

        legend_patches = [
            mpatches.Patch(facecolor=hex_to_rgba(COLORES.get(m, '#888'), 0.4),
                           edgecolor=COLORES.get(m, '#888'), linewidth=1.1, label=m)
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


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="resultados")
    parser.add_argument("--output-dir", default="violinplots")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, args.input_dir) if not os.path.isabs(args.input_dir) else args.input_dir
    output_dir = os.path.join(script_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("Cargando datos...")
    df = cargar_todos(input_dir)
    if df.empty:
        print("Error: sin datos.")
        exit(1)
    print(f"  {len(df):,} filas | {df['Modelo'].nunique()} modelos\n")

    df_g1 = df[df['Dataset'].isin(DATASETS_G1)]
    if not df_g1.empty:
        print("Violinplots G1 (9 modelos × 3 datasets)...")
        generar_violins(df_g1, ORDEN_G1, DATASETS_G1, "G1", output_dir)

    df_g2 = df[df['Dataset'] == 'Teleco']
    if not df_g2.empty:
        print("\nViolinplots G2 (7 modelos × Teleco)...")
        generar_violins(df_g2, ORDEN_G2, ['Teleco'], "G2_Teleco", output_dir, es_teleco=True)

    print("\nListo.")