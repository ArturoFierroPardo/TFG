# pip install pandas matplotlib numpy
"""
Barplots globales — G1 (9 modelos × 3 datasets) + G2 (7 modelos × Teleco).
Genera Coste, CO2, Tiempo a partir de los CSVs por fila.

USO: python barplots.py --input-dir resultados --output-dir barplots
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob, os, argparse
import numpy as np

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'xtick.direction': 'in', 'ytick.direction': 'in',
})

# ── G1: 9 modelos ─────────────────────────────────────────────────────────
COLORES_G1 = {
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

ORDEN_G1 = [
    'DeepSeek', 'Llama 70B', 'Qwen 72B',
    'Gemma 9B', 'Llama 3B', 'Qwen 7B',
    'Gemma 3.1B', 'Llama 1B', 'Qwen3 1.7B',
]

DATASETS_G1 = ['WebNLG', 'ToTTo', 'KELM']

ARCHIVO_MAPA_G1 = [
    ('webNLG_Gemma_3_1B',      'Gemma 3.1B',  'WebNLG'),
    ('webNLG_Llama_1B',        'Llama 1B',    'WebNLG'),
    ('webNLG_Qwen3_1.7B',     'Qwen3 1.7B',  'WebNLG'),
    ('totto_Gemma_3_1B',       'Gemma 3.1B',  'ToTTo'),
    ('totto_Llama_1B',         'Llama 1B',    'ToTTo'),
    ('totto_Qwen3_1.7B',      'Qwen3 1.7B',  'ToTTo'),
    ('kelm_stem_Gemma_3_1B',  'Gemma 3.1B',  'KELM'),
    ('kelm_stem_Llama_1B',    'Llama 1B',    'KELM'),
    ('kelm_stem_Qwen3_1.7B', 'Qwen3 1.7B',  'KELM'),
    ('DeepSeek_WebNLG',  'DeepSeek',  'WebNLG'),
    ('DeepSeek_ToTTo',   'DeepSeek',  'ToTTo'),
    ('DeepSeek_KELM',    'DeepSeek',  'KELM'),
    ('Llama_70B_WebNLG', 'Llama 70B', 'WebNLG'),
    ('Llama_70B_ToTTo',  'Llama 70B', 'ToTTo'),
    ('Llama_70B_KELM',   'Llama 70B', 'KELM'),
    ('Qwen_72B_WebNLG',  'Qwen 72B',  'WebNLG'),
    ('Qwen_72B_ToTTo',   'Qwen 72B',  'ToTTo'),
    ('Qwen_72B_KELM',    'Qwen 72B',  'KELM'),
    ('Gemma_9B_WebNLG',  'Gemma 9B',  'WebNLG'),
    ('Gemma_9B_ToTTo',   'Gemma 9B',  'ToTTo'),
    ('Gemma_9B_KELM',    'Gemma 9B',  'KELM'),
    ('Llama_3B_WebNLG',  'Llama 3B',  'WebNLG'),
    ('Llama_3B_ToTTo',   'Llama 3B',  'ToTTo'),
    ('Llama_3B_KELM',    'Llama 3B',  'KELM'),
    ('Qwen_7B_WebNLG',   'Qwen 7B',   'WebNLG'),
    ('Qwen_7B_ToTTo',    'Qwen 7B',   'ToTTo'),
    ('Qwen_7B_KELM',     'Qwen 7B',   'KELM'),
]

# ── G2: 7 modelos Teleco ──────────────────────────────────────────────────
COLORES_G2 = {
    'GAN':              '#E41A1C',
    'Gemma 3.1B':       '#E7298A',
    'Llama 1B':         '#66A61E',
    'Qwen3 1.7B':       '#D95F02',
    'Gemma 3.1B FT':    '#984EA3',
    'Llama 1B FT':      '#377EB8',
    'Qwen3 1.7B FT':    '#FF7F00',
}

ORDEN_G2 = [
    'GAN',
    'Gemma 3.1B', 'Llama 1B', 'Qwen3 1.7B',
    'Gemma 3.1B FT', 'Llama 1B FT', 'Qwen3 1.7B FT',
]

ARCHIVO_MAPA_G2 = [
    ('GAN_valtest',              'GAN',            'Teleco'),
    ('teleco_Gemma_3_1B',        'Gemma 3.1B',     'Teleco'),
    ('teleco_Llama_1B',          'Llama 1B',       'Teleco'),
    ('teleco_Qwen3_1.7B',       'Qwen3 1.7B',     'Teleco'),
    ('Gemma_3_1B_FT_valtest',   'Gemma 3.1B FT',  'Teleco'),
    ('Llama_1B_FT_valtest',     'Llama 1B FT',    'Teleco'),
    ('Qwen3_1.7B_FT_valtest',  'Qwen3 1.7B FT',  'Teleco'),
]

# ── Métricas a graficar ───────────────────────────────────────────────────
METRICAS_BAR = [
    ('Coste_USD',    'Coste Total (USD)',           'USD',           '${:.4f}'),
    ('CO2_gramos',   'CO$_2$ Total (gramos)',       'gramos CO$_2$', '{:.1f}g'),
    ('Time_seconds', 'Tiempo Total',                'Segundos',      '{:.0f}s'),
]


# ── Utilidades ────────────────────────────────────────────────────────────
def hex_to_rgba(hex_color, alpha=0.25):
    h = hex_color.lstrip('#')
    r, g, b = tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))
    return (r, g, b, alpha)


def cargar(input_dir, archivo_mapa):
    df_total = pd.DataFrame()
    for archivo in glob.glob(os.path.join(input_dir, "metricas_por_fila_*.csv")):
        base = os.path.basename(archivo).replace("metricas_por_fila_", "").replace(".csv", "")
        modelo, dataset = None, None
        for entrada in archivo_mapa:
            patron, m = entrada[0], entrada[1]
            ds = entrada[2] if len(entrada) > 2 else 'Teleco'
            if base == patron and m is not None:
                modelo, dataset = m, ds
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

    cols_num = ['Coste_USD', 'CO2_gramos', 'Time_seconds']
    for c in cols_num:
        if c in df_total.columns:
            df_total[c] = pd.to_numeric(df_total[c], errors='coerce')

    return df_total


# ── Barplot genérico ──────────────────────────────────────────────────────
def generar_barplots(df, datasets, orden_modelos, colores, prefijo, output_dir):
    modelos_presentes = [m for m in orden_modelos if m in df['Modelo'].unique()]
    datasets_presentes = [d for d in datasets if d in df['Dataset'].unique()]

    if not modelos_presentes or not datasets_presentes:
        print(f"  [{prefijo}] Sin datos suficientes.")
        return

    # Conteo de filas por dataset
    conteos = {}
    for ds in datasets_presentes:
        sub = df[df['Dataset'] == ds]
        if not sub.empty:
            conteos[ds] = len(sub[sub['Modelo'] == sub['Modelo'].iloc[0]])

    n_ds = len(datasets_presentes)
    n_mod = len(modelos_presentes)
    bar_w = 0.8 / n_mod
    x = np.arange(n_ds)

    for col, titulo_base, ylabel, fmt in METRICAS_BAR:
        if col not in df.columns:
            print(f"  Columna {col} no encontrada, saltando")
            continue

        resumen = df.groupby(['Modelo', 'Dataset'])[col].sum().reset_index()
        resumen.rename(columns={col: 'valor'}, inplace=True)

        sufijo = f" — BBDD Teleco" if 'Teleco' in datasets_presentes else ""
        titulo = titulo_base + sufijo

        fig, ax = plt.subplots(figsize=(13 if n_ds > 1 else 11, 6))
        max_val = resumen['valor'].max() if not resumen.empty else 1

        for k, m in enumerate(modelos_presentes):
            offset = (k - n_mod / 2 + 0.5) * bar_w
            vals = []
            for ds in datasets_presentes:
                v = resumen[(resumen['Modelo'] == m) & (resumen['Dataset'] == ds)]['valor']
                vals.append(v.values[0] if not v.empty else np.nan)

            c = colores.get(m, '#888')
            bars = ax.bar(x + offset, vals, width=bar_w * 0.9,
                          color=hex_to_rgba(c, 0.4), edgecolor=c, linewidth=1.2, label=m)

            for bar, val in zip(bars, vals):
                if pd.notna(val) and val > 0:
                    fs = 7.5 if n_ds == 1 else 6
                    ax.text(bar.get_x() + bar.get_width() / 2.,
                            bar.get_height() + max_val * 0.015,
                            fmt.format(val), ha='center', va='bottom', fontsize=fs)

        labels_x = [f"{ds}\n({conteos.get(ds, '?')} datos)" for ds in datasets_presentes]
        ax.set_xticks(x)
        ax.set_xticklabels(labels_x, fontsize=12)
        ax.tick_params(axis='x', rotation=0, direction='in', length=7)
        ax.tick_params(axis='y', direction='in', length=7)

        ax.set_title(titulo, fontsize=13, fontweight='normal', pad=10)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_ylim(0, max_val * 1.30)

        ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='#888888', linewidth=0.5)
        ax.set_axisbelow(True)

        legend_patches = [
            mpatches.Patch(facecolor=hex_to_rgba(colores[m], 0.4),
                           edgecolor=colores[m], linewidth=1.2, label=m)
            for m in modelos_presentes
        ]
        ncol_leg = min(len(modelos_presentes), 4)
        ax.legend(handles=legend_patches, loc='upper right', fontsize=8 if n_mod > 5 else 9,
                  framealpha=0.95, facecolor='white', edgecolor='black', fancybox=False,
                  ncol=1 if n_mod <= 5 else 2)

        plt.tight_layout()
        fname = os.path.join(output_dir, f"{prefijo}_{col}.png")
        plt.savefig(fname, dpi=300, facecolor='white')
        plt.close()
        print(f"  {fname}")


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="resultados")
    parser.add_argument("--output-dir", default="barplots")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, args.input_dir) if not os.path.isabs(args.input_dir) else args.input_dir
    output_dir = os.path.join(script_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_dir):
        print(f"Error: No se encuentra '{input_dir}'")
        exit(1)

    # ── G1: 9 modelos × 3 datasets ──
    print("Cargando G1 (9 modelos × 3 datasets)...")
    df_g1 = cargar(input_dir, ARCHIVO_MAPA_G1)
    if not df_g1.empty:
        print(f"  {len(df_g1):,} filas | {df_g1['Modelo'].nunique()} modelos | {df_g1['Dataset'].nunique()} datasets")
        print("Generando barplots G1...")
        generar_barplots(df_g1, DATASETS_G1, ORDEN_G1, COLORES_G1, "Global_G1", output_dir)
    else:
        print("  Sin datos para G1.")

    # ── G2: 7 modelos × Teleco ──
    print("\nCargando G2 (7 modelos × Teleco)...")
    df_g2 = cargar(input_dir, ARCHIVO_MAPA_G2)
    if not df_g2.empty:
        print(f"  {len(df_g2):,} filas | {df_g2['Modelo'].nunique()} modelos")
        print("Generando barplots G2...")
        generar_barplots(df_g2, ['Teleco'], ORDEN_G2, COLORES_G2, "Global_G2_Teleco", output_dir)
    else:
        print("  Sin datos para G2.")

    print("\nListo.")