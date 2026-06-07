# pip install pandas matplotlib numpy
"""
Dos tipos de grid de boxplots:

1. grid_por_dataset: 4 métricas en 2×2 para un mismo dataset (estilo original)
   - Una imagen por dataset: WebNLG, ToTTo, KELM
   - Leyenda única abajo

2. grid_teleco: Mini-SLM vs FT vs GAN (estilo imagen de referencia)
   - Grupos: GAN | Mini-SLM Base | Mini-SLM Fine-Tuned
   - 4 métricas en 2×2
   - Leyenda única abajo

USO: python boxplot_grids.py --input-dir analisis
"""
import pandas as pd, matplotlib.pyplot as plt, matplotlib.patches as mpatches
import matplotlib.ticker as ticker, numpy as np, glob, os, argparse

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': 'black', 'xtick.direction': 'in', 'ytick.direction': 'in',
})

# ─── Paleta colores ────────────────────────────────────────────────────────
MODELO_COLORES = {
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

ORDEN_G1 = ['DeepSeek','Llama 70B','Qwen 72B','Gemma 9B','Llama 3B','Qwen 7B','Gemma 3 1B','Llama 1B','Qwen3 1.7B']
METRICAS = [('ROUGE_L','ROUGE-L'),('METEOR','METEOR'),('BLEU','BLEU Score'),('BERTScore','BERTScore')]

ARCHIVO_MAPA = [
    ('webNLG_Gemma_3_1B',    'Gemma 3 1B','WebNLG'),
    ('webNLG_Llama_1B',      'Llama 1B',  'WebNLG'),
    ('webNLG_Qwen3_1.7B',   'Qwen3 1.7B','WebNLG'),
    ('totto_Gemma_3_1B',     'Gemma 3 1B','ToTTo'),
    ('totto_Llama_1B',       'Llama 1B',  'ToTTo'),
    ('totto_Qwen3_1.7B',    'Qwen3 1.7B','ToTTo'),
    ('kelm_stem_Gemma_3_1B', 'Gemma 3 1B','KELM'),
    ('kelm_stem_Llama_1B',   'Llama 1B',  'KELM'),
    ('kelm_stem_Qwen3_1.7B','Qwen3 1.7B','KELM'),
    ('teleco_Gemma_3_1B',    'Gemma 3 1B','Teleco'),
    ('teleco_Llama_1B',      'Llama 1B',  'Teleco'),
    ('teleco_Qwen3_1.7B',   'Qwen3 1.7B','Teleco'),
    ('GAN_valtest',          'GAN',          'Teleco'),
    ('Gemma_3_1B_FT_valtest','Gemma 3 1B FT','Teleco'),
    ('Llama_1B_FT_valtest',  'Llama 1B FT',  'Teleco'),
    ('Qwen3_1.7B_FT_valtest','Qwen3 1.7B FT','Teleco'),
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

def cargar_todos(input_dir):
    df_total = pd.DataFrame()
    for archivo in glob.glob(os.path.join(input_dir,"metricas_por_fila_*.csv")):
        base = os.path.basename(archivo).replace("metricas_por_fila_","").replace(".csv","")
        modelo, dataset = None, None
        for pat, m, d in ARCHIVO_MAPA:
            if base == pat: modelo, dataset = m, d; break
        if modelo is None: continue
        try:
            df = pd.read_csv(archivo)
            df['Modelo'] = modelo; df['Dataset'] = dataset
            df_total = pd.concat([df_total, df], ignore_index=True)
        except: pass
    for c in ['ROUGE_L','METEOR','BLEU','BERTScore']:
        if c in df_total.columns:
            df_total[c] = pd.to_numeric(df_total[c], errors='coerce')
            df_total = df_total[df_total[c].isna()|(df_total[c]<=1.0)]
    return df_total

def hex_to_rgba(hex_color, alpha=0.25):
    h = hex_color.lstrip('#')
    r,g,b = tuple(int(h[i:i+2],16)/255 for i in (0,2,4))
    return (r,g,b,alpha)

# ─── Función boxplot individual (estilo original) ──────────────────────────
def dibujar_ax_original(ax, df, columna, modelos, datasets_labels,
                         group_gap=0.08, box_w=0.012, gap=0.020, hide_xlabel=False):
    """Replica exactamente el estilo de boxplots.py original."""
    current_x = 0
    group_centers = []
    group_starts  = []
    group_ends    = []
    gmin, gmax = np.inf, -np.inf

    for j, (ds, label) in enumerate(datasets_labels):
        x_pos_ds = []
        for i, modelo in enumerate(modelos):
            subset = df[(df['Modelo']==modelo)&(df['Dataset']==ds)][columna].dropna()
            if len(subset) == 0:
                current_x += box_w + gap; continue
            color = MODELO_COLORES.get(modelo,'#888888')
            face  = hex_to_rgba(color, 0.25)
            vals  = subset.values

            ax.boxplot([vals], positions=[current_x], widths=box_w,
                       patch_artist=True, showfliers=False,
                       medianprops=dict(color=color, linewidth=1.5),
                       whiskerprops=dict(color='black', linewidth=0.7),
                       capprops=dict(color='black', linewidth=0.7),
                       boxprops=dict(facecolor=face, edgecolor=color, linewidth=1.1),
                       zorder=3)

            q1,q3 = np.percentile(vals,25), np.percentile(vals,75)
            iqr = q3-q1
            lo,hi = q1-1.5*iqr, q3+1.5*iqr
            wmin = vals[vals>=lo].min() if np.any(vals>=lo) else q1
            wmax = vals[vals<=hi].max() if np.any(vals<=hi) else q3
            gmin = min(gmin,wmin); gmax = max(gmax,wmax)
            x_pos_ds.append(current_x)
            current_x += box_w + gap

        if x_pos_ds:
            center = (x_pos_ds[0]+x_pos_ds[-1])/2
            ext = (box_w/2) + ((gap+group_gap)/2)
            group_centers.append(center)
            group_starts.append(x_pos_ds[0]-ext)
            group_ends.append(x_pos_ds[-1]+ext)
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
        ax.set_xticklabels([l for _,l in datasets_labels], fontsize=13, fontweight='bold')
        ax.tick_params(axis='x', direction='in', length=7)
    ax.tick_params(axis='y', direction='in', length=7, labelsize=12)
    ax.xaxis.grid(True, linestyle='-', alpha=0.3, color='#888888', linewidth=0.5)
    ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='#888888', linewidth=0.5)
    ax.set_axisbelow(True)
    if group_starts:
        ax.set_xlim(group_starts[0], group_ends[-1])
    rango = gmax-gmin if gmax>gmin else 1
    ax.set_ylim(gmin-rango*0.05, gmax+rango*0.05)
    return gmin, gmax

# ════════════════════════════════════════════════════════
# GRID 1: 4 métricas (2×2) por dataset — estilo original
# ════════════════════════════════════════════════════════
def grid_por_dataset(df, dataset, modelos, output_dir):
    ds_label = [(dataset, dataset)]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11),
                              gridspec_kw={'hspace':0.25,'wspace':0.28})
    fig.suptitle(f'Dataset: {dataset}', fontsize=17, fontweight='bold', y=0.98)

    axes_flat = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]

    for ax, (col_met, nombre_met) in zip(axes_flat, METRICAS):
        if col_met not in df.columns:
            ax.set_visible(False); continue
        dibujar_ax_original(ax, df, col_met, modelos, ds_label,
                            group_gap=0.04, box_w=0.012, gap=0.018,
                            hide_xlabel=True)
        ax.set_title(nombre_met, fontsize=14, pad=8)

    # Leyenda única abajo
    legend_patches = [
        mpatches.Patch(facecolor=hex_to_rgba(MODELO_COLORES[m],0.25),
                       edgecolor=MODELO_COLORES[m], linewidth=1.1, label=m)
        for m in modelos if m in df['Modelo'].unique()
    ]
    fig.legend(handles=legend_patches, loc='lower center',
               bbox_to_anchor=(0.5, 0.0), ncol=len(legend_patches),
               fontsize=12, framealpha=0.95, edgecolor='#CCCCCC',
               title='Modelo', title_fontsize=12,
               handlelength=1.5, handleheight=1.2, borderpad=0.5,
               labelspacing=0.4, columnspacing=0.9)

    fname = os.path.join(output_dir, f'boxplot_grid_{dataset}.png')
    plt.subplots_adjust(bottom=0.10)
    plt.savefig(fname, dpi=200, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f'  OK: {fname}')

# ════════════════════════════════════════════════════════
# GRID 2: Mini-SLM vs FT vs GAN — Teleco (estilo imagen referencia)
# ════════════════════════════════════════════════════════
def grid_teleco(df, output_dir):
    # Grupos: GAN | Mini-SLM Base | Mini-SLM FT
    grupos = [
        ('GAN',          [('GAN','Teleco')],
                         [('GAN', 'GAN')]),
        ('Mini-SLM Base',[('Gemma 3 1B','Teleco'),('Llama 1B','Teleco'),('Qwen3 1.7B','Teleco')],
                         [('Gemma 3 1B','Gemma 3 1B'),('Llama 1B','Llama 1B'),('Qwen3 1.7B','Qwen3 1.7B')]),
        ('Mini-SLM Fine-Tuned',[('Gemma 3 1B FT','Teleco'),('Llama 1B FT','Teleco'),('Qwen3 1.7B FT','Teleco')],
                         [('Gemma 3 1B FT','Gemma 3 1B FT'),('Llama 1B FT','Llama 1B FT'),('Qwen3 1.7B FT','Qwen3 1.7B FT')]),
    ]

    # Construir lista (modelo, label_grupo) para dibujar_ax_original
    # Usamos dibujar_ax_original con un ds_label por grupo
    fig, axes = plt.subplots(2, 2, figsize=(16, 11),
                              gridspec_kw={'hspace':0.25,'wspace':0.28})
    fig.suptitle('Dataset: Teleco', fontsize=17, fontweight='bold', y=0.98)

    axes_flat = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]
    todos_modelos = []

    for ax, (col_met, nombre_met) in zip(axes_flat, METRICAS):
        if col_met not in df.columns:
            ax.set_visible(False); continue

        current_x = 0
        group_centers, group_starts, group_ends = [], [], []
        gmin, gmax = np.inf, -np.inf
        box_w, gap, group_gap = 0.012, 0.018, 0.08

        for g_label, mod_ds_list, _ in grupos:
            x_pos_g = []
            for modelo, ds in mod_ds_list:
                subset = df[(df['Modelo']==modelo)&(df['Dataset']==ds)][col_met].dropna()
                if len(subset)==0:
                    current_x += box_w+gap; continue
                color = MODELO_COLORES.get(modelo,'#888888')
                face  = hex_to_rgba(color, 0.25)
                vals  = subset.values
                ax.boxplot([vals], positions=[current_x], widths=box_w,
                           patch_artist=True, showfliers=False,
                           medianprops=dict(color=color, linewidth=1.5),
                           whiskerprops=dict(color='black', linewidth=0.7),
                           capprops=dict(color='black', linewidth=0.7),
                           boxprops=dict(facecolor=face, edgecolor=color, linewidth=1.1),
                           zorder=3)
                q1,q3 = np.percentile(vals,25), np.percentile(vals,75)
                iqr = q3-q1
                lo,hi = q1-1.5*iqr, q3+1.5*iqr
                wmin = vals[vals>=lo].min() if np.any(vals>=lo) else q1
                wmax = vals[vals<=hi].max() if np.any(vals<=hi) else q3
                gmin=min(gmin,wmin); gmax=max(gmax,wmax)
                x_pos_g.append(current_x)
                if modelo not in todos_modelos:
                    todos_modelos.append(modelo)
                current_x += box_w+gap

            if x_pos_g:
                center = (x_pos_g[0]+x_pos_g[-1])/2
                ext = (box_w/2)+((gap+group_gap)/2)
                group_centers.append(center)
                group_starts.append(x_pos_g[0]-ext)
                group_ends.append(x_pos_g[-1]+ext)
                current_x += group_gap

        # Bandas grises
        for j in range(len(group_starts)):
            if j%2==0:
                ax.axvspan(group_starts[j],group_ends[j],alpha=0.4,color='#E5E5E5',zorder=0,lw=0)

        ax.set_xticks(group_centers)
        ax.set_xticklabels([])
        ax.tick_params(axis='x', direction='in', length=0)
        ax.tick_params(axis='y', direction='in', length=7, labelsize=12)
        ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='#888888', linewidth=0.5)
        ax.set_axisbelow(True)
        if group_starts:
            ax.set_xlim(group_starts[0], group_ends[-1])
        rango = gmax-gmin if gmax>gmin else 1
        ax.set_ylim(gmin-rango*0.05, gmax+rango*0.05)
        ax.set_title(nombre_met, fontsize=14, pad=8)

    # Leyenda única abajo
    legend_patches = [
        mpatches.Patch(facecolor=hex_to_rgba(MODELO_COLORES[m],0.25),
                       edgecolor=MODELO_COLORES[m], linewidth=1.1, label=m)
        for m in todos_modelos
    ]
    fig.legend(handles=legend_patches, loc='lower center',
               bbox_to_anchor=(0.5, 0.0), ncol=len(legend_patches),
               fontsize=12, framealpha=0.95, edgecolor='#CCCCCC',
               title='Modelo', title_fontsize=12,
               handlelength=1.5, handleheight=1.2, borderpad=0.5,
               labelspacing=0.4, columnspacing=0.9)

    fname = os.path.join(output_dir, 'boxplot_grid_Teleco.png')
    plt.subplots_adjust(bottom=0.10)
    plt.savefig(fname, dpi=200, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f'  OK: {fname}')

# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="analisis")
    parser.add_argument("--output-dir", default="Boxplot_grids")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Cargando datos...")
    df = cargar_todos(args.input_dir)
    print(f"  {len(df):,} filas | {df['Modelo'].nunique()} modelos\n")

    modelos_g1 = [m for m in ORDEN_G1 if m in df['Modelo'].unique()]

    print("Generando grids por dataset (G1)...")
    for ds in ['WebNLG','ToTTo','KELM']:
        if ds in df['Dataset'].unique():
            grid_por_dataset(df, ds, modelos_g1, args.output_dir)

    print("Generando grid Teleco (GAN vs Mini vs FT)...")
    grid_teleco(df, args.output_dir)

    print("\nListo.")