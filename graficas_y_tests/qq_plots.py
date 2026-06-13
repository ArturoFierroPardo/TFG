"""
Q-Q plots de normalidad para cada metrica y modelo.

Genera dos tipos de figura: individuales (un subplot por modelo, en subcarpetas
por escenario y dataset) y grids (filas = metricas, columnas = modelos, una imagen
por familia y dataset).

Requisitos:
    pip install pandas matplotlib scipy

Uso:
    python qq_plots.py --input-dir resultados --output-dir QQ_plots
"""
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import glob, os, argparse

plt.rcParams.update({
    'font.size': 10, 'font.family': 'serif',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'xtick.direction': 'in', 'ytick.direction': 'in',
})

COLORES = {
    'DeepSeek':      '#3366CC',
    'Llama 70B':     '#DC3912',
    'Qwen 72B':      '#FF9900',
    'Gemma 9B':      '#109618',
    'Llama 3B':      '#990099',
    'Qwen 7B':       '#0099C6',
    'Gemma 3 1B':    '#DD4477',
    'Llama 1B':      '#66AA00',
    'Qwen3 1.7B':    '#B82E2E',
    'GAN':           '#17BECF',
    'Gemma 3 1B FT': '#E67300',
    'Llama 1B FT':   '#8B0707',
    'Qwen3 1.7B FT': '#329262',
}

METRICAS_CALIDAD = ['ROUGE_L', 'METEOR', 'BERTScore', 'BLEU']
METRICAS_EXTRA = ['Time_seconds', 'Coste_USD', 'CO2_gramos']

NOMBRES_MET = {
    'ROUGE_L': 'ROUGE-L', 'METEOR': 'METEOR', 'BERTScore': 'BERTScore',
    'BLEU': 'BLEU', 'Time_seconds': 'Tiempo (s)',
    'Coste_USD': 'Coste (USD)', 'CO2_gramos': 'CO₂ (g)',
}

ESCENARIOS = {
    'LLM': {
        'titulo': 'LLMs',
        'datasets': ['WebNLG', 'ToTTo', 'KELM'],
        'modelos': ['DeepSeek', 'Llama 70B', 'Qwen 72B'],
        'metricas_extra': True,
    },
    'SLM': {
        'titulo': 'SLMs',
        'datasets': ['WebNLG', 'ToTTo', 'KELM'],
        'modelos': ['Gemma 9B', 'Llama 3B', 'Qwen 7B'],
        'metricas_extra': True,
    },
    'Mini-SLM': {
        'titulo': 'Mini-SLMs',
        'datasets': ['WebNLG', 'ToTTo', 'KELM'],
        'modelos': ['Gemma 3 1B', 'Llama 1B', 'Qwen3 1.7B'],
        'metricas_extra': True,
    },
    'Mini-SLM_Teleco': {
        'titulo': 'Mini-SLMs (Teleco)',
        'datasets': ['Teleco'],
        'modelos': ['Gemma 3 1B', 'Llama 1B', 'Qwen3 1.7B'],
        'metricas_extra': True,
    },
    'GAN_y_Fine-Tuning': {
        'titulo': 'GAN y Fine-Tuning',
        'datasets': ['Teleco'],
        'modelos': ['GAN', 'Gemma 3 1B FT', 'Llama 1B FT', 'Qwen3 1.7B FT'],
        'metricas_extra': False,
    },
}

# Normalización de nombres
TABLA_MODELOS = {
    'deepseek': 'DeepSeek', 'llama_70b': 'Llama 70B', 'qwen_72b': 'Qwen 72B',
    'gemma_9b': 'Gemma 9B', 'llama_3b': 'Llama 3B', 'qwen_7b': 'Qwen 7B',
    'gemma_3_1b': 'Gemma 3 1B', 'llama_1b': 'Llama 1B',
    'qwen3_1.7b': 'Qwen3 1.7B', 'qwen3_1_7b': 'Qwen3 1.7B',
    'gan': 'GAN',
    'gemma_3_1b_ft': 'Gemma 3 1B FT', 'llama_1b_ft': 'Llama 1B FT',
    'qwen3_1.7b_ft': 'Qwen3 1.7B FT', 'qwen3_1_7b_ft': 'Qwen3 1.7B FT',
}


def normalizar(base):
    low = base.lower()

    # valtest → Teleco
    if 'valtest' in low:
        raw = base.replace('_valtest', '')
        m = TABLA_MODELOS.get(raw.lower(), raw.replace('_', ' '))
        return m, 'Teleco'

    # teleco_ prefijo
    if low.startswith('teleco_'):
        raw = base[7:]
        m = TABLA_MODELOS.get(raw.lower(), raw.replace('_', ' '))
        return m, 'Teleco'

    # dataset prefijo
    for pref, ds in [('webnlg_', 'WebNLG'), ('totto_', 'ToTTo'),
                     ('kelm_stem_', 'KELM'), ('kelm_', 'KELM')]:
        if low.startswith(pref):
            raw = base[len(pref):]
            m = TABLA_MODELOS.get(raw.lower(), raw.replace('_', ' '))
            return m, ds

    # dataset sufijo
    for suf, ds in [('_WebNLG', 'WebNLG'), ('_ToTTo', 'ToTTo'), ('_KELM', 'KELM')]:
        if base.endswith(suf):
            raw = base[:-len(suf)]
            m = TABLA_MODELOS.get(raw.lower(), raw.replace('_', ' '))
            return m, ds

    return base.replace('_', ' '), 'Desconocido'


# Carga
def cargar_todos(input_dir):
    df_total = pd.DataFrame()
    for archivo in glob.glob(os.path.join(input_dir, 'metricas_por_fila_*.csv')):
        try:
            df = pd.read_csv(archivo)
            base = os.path.basename(archivo).replace('metricas_por_fila_', '').replace('.csv', '')
            modelo, ds = normalizar(base)

            for col in METRICAS_CALIDAD + METRICAS_EXTRA:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Filtrar valores >1 en métricas de puntuación
            for col in METRICAS_CALIDAD:
                if col in df.columns:
                    df = df[df[col].isna() | (df[col] <= 1.0)]

            df['Modelo'] = modelo
            df['Dataset'] = ds
            df_total = pd.concat([df_total, df], ignore_index=True)
            print(f'  OK: {base:40s} → {modelo} / {ds}  ({len(df):,})')
        except Exception as e:
            print(f'  ERROR {archivo}: {e}')
    return df_total


# Dibujar QQ en un Axes
def dibujar_qq(ax, datos, color, titulo_modelo=None):
    if len(datos) < 10:
        ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center', transform=ax.transAxes)
        if titulo_modelo:
            ax.set_title(titulo_modelo, fontsize=12)
        return

    stats.probplot(datos, dist='norm', plot=ax)
    lineas = ax.get_lines()
    lineas[0].set_markerfacecolor('none')
    lineas[0].set_markeredgecolor(color)
    lineas[0].set_markersize(3)
    lineas[0].set_alpha(0.6)
    lineas[0].set_linestyle('none')
    lineas[1].set_color('#333333')
    lineas[1].set_linestyle('--')
    lineas[1].set_linewidth(1.5)

    if titulo_modelo:
        ax.set_title(titulo_modelo, fontsize=12, color=color, fontweight='bold', pad=8)

    ax.grid(True, linestyle=':', alpha=0.5, color='gray')
    ax.set_axisbelow(True)


# 1. INDIVIDUALES: 2×2 por métrica (un subplot por modelo)
def generar_individuales(df, output_dir):
    for familia, cfg in ESCENARIOS.items():
        metricas = METRICAS_CALIDAD + (METRICAS_EXTRA if cfg['metricas_extra'] else [])

        for ds in cfg['datasets']:
            carpeta = os.path.join(output_dir, 'individuales', familia, ds)
            os.makedirs(carpeta, exist_ok=True)

            for col in metricas:
                if col not in df.columns:
                    continue
                nombre = NOMBRES_MET.get(col, col)

                df_f = df[(df['Dataset'] == ds) & (df['Modelo'].isin(cfg['modelos'])) & df[col].notna()]
                if df_f.empty:
                    continue

                n_mod = len(cfg['modelos'])
                ncols = min(n_mod, 2)
                nrows = (n_mod + ncols - 1) // ncols
                fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
                fig.suptitle(f"Q-Q Plot: {nombre}\nDataset: {ds} ({cfg['titulo']})",
                             fontsize=16, y=0.98)

                axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

                for idx, modelo in enumerate(cfg['modelos']):
                    ax = axes_flat[idx]
                    datos = df_f[df_f['Modelo'] == modelo][col].values
                    color = COLORES.get(modelo, '#000')
                    dibujar_qq(ax, datos, color, modelo)
                    ax.set_xlabel('Normal Teórica')
                    ax.set_ylabel(f'Puntuación ({nombre})')

                # Ocultar ejes sobrantes
                for idx in range(n_mod, len(axes_flat)):
                    fig.delaxes(axes_flat[idx])

                plt.tight_layout()
                plt.subplots_adjust(top=0.88)
                fname = os.path.join(carpeta, f'QQ_{col}.png')
                plt.savefig(fname, dpi=300, bbox_inches='tight')
                plt.close()

    print("  ✓ Individuales generados")


# 2. GRIDS: filas=métricas, columnas=modelos
def generar_grids(df, output_dir):
    carpeta = os.path.join(output_dir, 'grids')
    os.makedirs(carpeta, exist_ok=True)

    for familia, cfg in ESCENARIOS.items():
        modelos = cfg['modelos']
        n_mod = len(modelos)
        n_met = len(METRICAS_CALIDAD)

        for ds in cfg['datasets']:
            df_ds = df[(df['Dataset'] == ds) & (df['Modelo'].isin(modelos))]
            if df_ds.empty:
                continue

            fig, axes = plt.subplots(n_met, n_mod, figsize=(5 * n_mod, 4.5 * n_met), squeeze=False)
            fig.suptitle(f'Q-Q Plots — {cfg["titulo"]} | Dataset: {ds}',
                         fontsize=16, y=1.005, fontweight='bold')

            for fila, col_met in enumerate(METRICAS_CALIDAD):
                nombre = NOMBRES_MET[col_met]
                for col_idx, modelo in enumerate(modelos):
                    ax = axes[fila][col_idx]
                    color = COLORES.get(modelo, '#000')
                    datos = df_ds[df_ds['Modelo'] == modelo][col_met].dropna().values

                    dibujar_qq(ax, datos, color,
                               titulo_modelo=modelo if fila == 0 else None)
                    if fila != 0:
                        ax.set_title('')

                    ax.set_ylabel(f'{nombre}\nPuntuación' if col_idx == 0 else '', fontsize=11,
                                  fontweight='bold' if col_idx == 0 else 'normal')
                    ax.set_xlabel('Cuantiles teóricos' if fila == n_met - 1 else '', fontsize=10)

                    if fila % 2 == 1:
                        ax.set_facecolor('#FAFAFA')

            plt.tight_layout(rect=[0, 0, 1, 1])
            sufijo_ds = '' if ds in familia else f'_{ds}'
            fname = os.path.join(carpeta, f'QQ_{familia.replace("-", "_")}{sufijo_ds}.png')
            plt.savefig(fname, dpi=200, bbox_inches='tight')
            plt.close()
            print(f'  {fname}')

    print("  ✓ Grids generados")


# Main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='resultados')
    parser.add_argument('--output-dir', default='QQ_plots')
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f'Error: no se encuentra "{args.input_dir}"')
        exit(1)

    print('Cargando datos...')
    df = cargar_todos(args.input_dir)
    if df.empty:
        print('Error: sin datos.')
        exit(1)
    print(f'{len(df):,} filas cargadas\n')

    print('Generando QQ individuales...')
    generar_individuales(df, args.output_dir)

    print('\nGenerando QQ grids...')
    generar_grids(df, args.output_dir)

    print(f'\nListo. Todo en: {args.output_dir}/')