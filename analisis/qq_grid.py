# pip install pandas matplotlib scipy
"""
Genera Q-Q plots organizados en una imagen por familia × dataset.
Layout: filas = métricas (ROUGE-L, METEOR, BERTScore, BLEU)
        columnas = modelos (3 por familia)

Resultado: una imagen por combinación, ej:
  QQ_LLMs_WebNLG.png, QQ_LLMs_ToTTo.png, QQ_SLMs_WebNLG.png ...

USO: python qq_grid.py --input-dir analisis
"""
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import glob, os, argparse

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
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

METRICAS = ['ROUGE_L', 'METEOR', 'BERTScore', 'BLEU']
NOMBRES_MET = {
    'ROUGE_L':   'ROUGE-L',
    'METEOR':    'METEOR',
    'BERTScore': 'BERTScore',
    'BLEU':      'BLEU',
}

ESCENARIOS = {
    'LLM': {
        'datasets': ['WebNLG', 'ToTTo', 'KELM'],
        'modelos':  ['DeepSeek', 'Llama 70B', 'Qwen 72B'],
    },
    'SLM': {
        'datasets': ['WebNLG', 'ToTTo', 'KELM'],
        'modelos':  ['Gemma 9B', 'Llama 3B', 'Qwen 7B'],
    },
    'Mini-SLM': {
        'datasets': ['WebNLG', 'ToTTo', 'KELM'],
        'modelos':  ['Gemma 3 1B', 'Llama 1B', 'Qwen3 1.7B'],
    },
    'Mini-SLM_Teleco': {
        'datasets': ['Teleco'],
        'modelos':  ['Gemma 3 1B', 'Llama 1B', 'Qwen3 1.7B'],
    },
    'GAN_y_Fine-Tuning': {
        'datasets': ['Teleco'],
        'modelos':  ['GAN', 'Gemma 3 1B FT', 'Llama 1B FT', 'Qwen3 1.7B FT'],
    },
}


def normalizar_modelo(base):
    """Extrae (modelo_canonico, dataset) del nombre de archivo."""
    TABLA = {
        'deepseek':       'DeepSeek',
        'llama 70b':      'Llama 70B',
        'llama_70b':      'Llama 70B',
        'qwen 72b':       'Qwen 72B',
        'qwen_72b':       'Qwen 72B',
        'gemma 9b':       'Gemma 9B',
        'gemma_9b':       'Gemma 9B',
        'llama 3b':       'Llama 3B',
        'llama_3b':       'Llama 3B',
        'qwen 7b':        'Qwen 7B',
        'qwen_7b':        'Qwen 7B',
        'gemma 3 1b':     'Gemma 3 1B',
        'gemma_3_1b':     'Gemma 3 1B',
        'llama 1b':       'Llama 1B',
        'llama_1b':       'Llama 1B',
        'qwen3 1.7b':     'Qwen3 1.7B',
        'qwen3_1.7b':     'Qwen3 1.7B',
        'qwen3_1_7b':     'Qwen3 1.7B',   # Windows reemplaza . por _
        'qwen3 1 7b':     'Qwen3 1.7B',
        'gan':            'GAN',
        'gemma 3 1b ft':  'Gemma 3 1B FT',
        'gemma_3_1b ft':  'Gemma 3 1B FT',
        'gemma_3_1b_ft':  'Gemma 3 1B FT',
        'llama 1b ft':    'Llama 1B FT',
        'llama_1b ft':    'Llama 1B FT',
        'llama_1b_ft':    'Llama 1B FT',
        'qwen3 1.7b ft':  'Qwen3 1.7B FT',
        'qwen3_1.7b ft':  'Qwen3 1.7B FT',
        'qwen3_1.7b_ft':  'Qwen3 1.7B FT',
        'qwen3_1_7b ft':  'Qwen3 1.7B FT',
        'qwen3_1_7b_ft':  'Qwen3 1.7B FT',
        'qwen3 1 7b ft':  'Qwen3 1.7B FT',
    }

    base_low = base.lower()

    # 1. valtest → Teleco
    if 'valtest' in base_low:
        raw = base.replace('_valtest', '').replace('_FT', '_FT')
        raw_low = raw.lower()
        modelo = TABLA.get(raw_low, TABLA.get(raw_low.replace('_', ' '), raw))
        return modelo, 'Teleco'

    # 2. teleco_ prefijo
    if base_low.startswith('teleco_'):
        raw = base[7:]
        modelo = TABLA.get(raw.lower(), TABLA.get(raw.lower().replace('_', ' '), raw))
        return modelo, 'Teleco'

    # 3. Prefijo dataset (webNLG_, totto_, kelm_stem_)
    for prefijo, ds in [('webnlg_', 'WebNLG'), ('totto_', 'ToTTo'),
                        ('kelm_stem_', 'KELM'), ('kelm_', 'KELM')]:
        if base_low.startswith(prefijo):
            raw = base[len(prefijo):]
            modelo = TABLA.get(raw.lower(), TABLA.get(raw.lower().replace('_', ' '), raw))
            return modelo, ds

    # 4. Sufijo dataset (Modelo_WebNLG, Modelo_ToTTo, Modelo_KELM)
    for sufijo, ds in [('_WebNLG', 'WebNLG'), ('_ToTTo', 'ToTTo'), ('_KELM', 'KELM')]:
        if base.endswith(sufijo):
            raw = base[:-len(sufijo)]
            modelo = TABLA.get(raw.lower(), TABLA.get(raw.lower().replace('_', ' '), raw))
            return modelo, ds

    return base, 'Desconocido'


def cargar_todos(input_dir):
    """Carga todos los CSVs del directorio y devuelve un DataFrame con cols Modelo y Dataset."""
    df_total = pd.DataFrame()
    archivos = glob.glob(os.path.join(input_dir, 'metricas_por_fila_*.csv'))

    for archivo in archivos:
        try:
            df = pd.read_csv(archivo)
            base = os.path.basename(archivo).replace('metricas_por_fila_', '').replace('.csv', '')
            modelo, ds = normalizar_modelo(base)

            # Forzar numérico en métricas
            for col in METRICAS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Filtrar filas de metadatos (valores > 1 en métricas de puntuación)
            mask = pd.Series([True] * len(df), index=df.index)
            for col in ['ROUGE_L', 'METEOR', 'BERTScore', 'BLEU']:
                if col in df.columns:
                    mask &= df[col].isna() | (df[col] <= 1.0)
            df = df[mask]

            df['Modelo']  = modelo
            df['Dataset'] = ds
            df_total = pd.concat([df_total, df], ignore_index=True)
            print(f'  OK: {base:40} → "{modelo}" / {ds}  ({len(df):,} filas)')

        except Exception as e:
            print(f'  ERROR {archivo}: {e}')

    return df_total


def generar_grid(df, output_dir='QQ_Grid'):
    """
    Por cada escenario × dataset genera una imagen con:
      filas    = métricas (ROUGE-L, METEOR, BERTScore, BLEU)
      columnas = modelos  (3 modelos de la familia)
    """
    os.makedirs(output_dir, exist_ok=True)

    for familia, cfg in ESCENARIOS.items():
        modelos = cfg['modelos']
        n_mod   = len(modelos)   # 3

        for ds in cfg['datasets']:
            df_ds = df[(df['Dataset'] == ds) & (df['Modelo'].isin(modelos))]
            if df_ds.empty:
                print(f'  Sin datos: {familia} / {ds}')
                continue

            # Grid: 4 filas (métricas) × 3 cols (modelos)
            fig, axes = plt.subplots(
                4, n_mod,
                figsize=(5 * n_mod, 4.5 * 4),
                squeeze=False
            )

            fig.suptitle(
                f'Q-Q Plots de Normalidad — {familia} | Dataset: {ds}',
                fontsize=16, y=1.005, fontweight='bold'
            )

            for fila, col_met in enumerate(METRICAS):
                nombre_met = NOMBRES_MET[col_met]

                for col_idx, modelo in enumerate(modelos):
                    ax = axes[fila][col_idx]
                    color = COLORES.get(modelo, '#000000')

                    datos = df_ds[df_ds['Modelo'] == modelo][col_met].dropna().values

                    if len(datos) < 10:
                        ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center',
                                transform=ax.transAxes)
                    else:
                        res = stats.probplot(datos, dist='norm', plot=ax)
                        lineas = ax.get_lines()

                        # Puntos: círculos vacíos con color del modelo
                        lineas[0].set_markerfacecolor('none')
                        lineas[0].set_markeredgecolor(color)
                        lineas[0].set_markersize(3)
                        lineas[0].set_alpha(0.6)
                        lineas[0].set_linestyle('none')

                        # Línea de referencia: gris discontinua
                        lineas[1].set_color('#333333')
                        lineas[1].set_linestyle('--')
                        lineas[1].set_linewidth(1.5)

                    # Título de columna solo en la primera fila
                    if fila == 0:
                        ax.set_title(modelo, fontsize=12, color=color,
                                     fontweight='bold', pad=8)
                    else:
                        ax.set_title('')

                    # Etiqueta de métrica solo en la primera columna
                    if col_idx == 0:
                        ax.set_ylabel(f'{nombre_met}\nPuntuación Real', fontsize=13, fontweight='bold', labelpad=8)
                    else:
                        ax.set_ylabel('')

                    # Etiqueta X solo en la última fila
                    if fila == len(METRICAS) - 1:
                        ax.set_xlabel('Cuantiles teóricos (Normal)', fontsize=10)
                    else:
                        ax.set_xlabel('')

                    ax.grid(True, linestyle=':', alpha=0.5, color='gray')
                    ax.set_axisbelow(True)

                    # Banda de métrica: fondo muy suave alternado por fila
                    if fila % 2 == 1:
                        ax.set_facecolor('#FAFAFA')

            plt.tight_layout(rect=[0, 0, 1, 1])

            # Evitar duplicar el dataset en el nombre si ya está en el nombre de familia
            sufijo_ds = '' if ds in familia else f'_{ds}'
            fname = os.path.join(output_dir, f'QQ_{familia.replace("-","_")}{sufijo_ds}.png')
            plt.savefig(fname, dpi=200, bbox_inches='tight')
            plt.close()
            print(f'  Generado: {fname}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='analisis')
    parser.add_argument('--output-dir', default='QQ_Grid')
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f'Error: no se encuentra la carpeta "{args.input_dir}"')
        exit(1)

    print('Cargando datos...')
    df = cargar_todos(args.input_dir)
    if df.empty:
        print('Error: no se han cargado datos.')
        exit(1)

    print(f'Datos cargados: {len(df):,} filas')
    print('Generando grids...')
    generar_grid(df, args.output_dir)
    print(f'\nListo. Imágenes en: {args.output_dir}/')
