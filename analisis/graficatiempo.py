# pip install pandas matplotlib numpy
"""
Gráficos de Líneas Acumulativos — 9 modelos.
Un gráfico por dataset. Escala Log-Log.
LLMs: línea continua. SLMs: línea discontinua. Mini-SLMs: línea punteada.

USO: python graficatiempo_9modelos.py --input-dir analisis
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
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

ESTILOS_LINEA = {
    'DeepSeek':    '-',
    'Llama 70B':   '-',
    'Qwen 72B':    '-',
    'Gemma 9B':    '--',
    'Llama 3B':    '--',
    'Qwen 7B':     '--',
    'Gemma 3.1B':  ':',
    'Llama 1B':    ':',
    'Qwen3 1.7B':  ':',
}

MARCADORES = {
    'DeepSeek':    'o',
    'Llama 70B':   's',
    'Qwen 72B':    '^',
    'Gemma 9B':    'D',
    'Llama 3B':    '*',
    'Qwen 7B':     'v',
    'Gemma 3.1B':  'P',   # Cruz gruesa
    'Llama 1B':    'X',   # X gruesa
    'Qwen3 1.7B':  'h',   # Hexágono
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


def generar_graficos_acumulativos(df):
    if 'Time_seconds' not in df.columns:
        print("Error: No se encuentra 'Time_seconds'.")
        return

    for ds in DATASETS:
        df_ds = df[df['Dataset'] == ds]
        if df_ds.empty:
            continue

        fig, ax = plt.subplots(figsize=(11, 6))

        for modelo in ORDEN_MODELOS:
            df_modelo = df_ds[df_ds['Modelo'] == modelo]
            if df_modelo.empty:
                continue

            tiempos = df_modelo['Time_seconds'].dropna().values
            tiempo_acum = pd.Series(tiempos).cumsum().values
            eje_x = range(1, len(tiempo_acum) + 1)

            color = MODELO_COLORES.get(modelo, '#000000')
            estilo = ESTILOS_LINEA.get(modelo, '-')
            marcador = MARCADORES.get(modelo, 'o')

            if len(eje_x) > 10:
                idx_marc = np.unique(np.geomspace(1, len(eje_x) - 1, num=12).astype(int))
            else:
                idx_marc = np.arange(len(eje_x))

            ax.plot(eje_x, tiempo_acum, label=modelo, color=color,
                    linestyle=estilo, linewidth=2.5, alpha=0.8,
                    marker=marcador, markersize=7, markevery=idx_marc.tolist())

        ax.set_title(f"Tiempo Total Acumulado - Dataset: {ds}", fontsize=14, pad=15)
        ax.set_xlabel("Cantidad de datos procesados (Nº de filas)", fontsize=12)
        ax.set_ylabel("Tiempo Total Acumulado (Segundos)", fontsize=12)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
        ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

        ax.grid(True, which='major', linestyle='-', alpha=0.4, color='#888888')
        ax.grid(True, which='minor', linestyle=':', alpha=0.2, color='#888888')
        ax.set_axisbelow(True)

        ax.tick_params(axis='both', which='major', direction='in', length=7)
        ax.tick_params(axis='both', which='minor', direction='in', length=4)

        legend_elements = [
            Line2D([0], [0], color=MODELO_COLORES[m], lw=2.5,
                   linestyle=ESTILOS_LINEA[m], marker=MARCADORES[m],
                   markersize=7, label=m)
            for m in ORDEN_MODELOS if m in df_ds['Modelo'].unique()
        ]

        ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
                  framealpha=0.95, facecolor='white', edgecolor='black',
                  fancybox=False, handlelength=3.0)

        plt.tight_layout()
        nombre = f"Acumulativo_Tiempo_Log_9m_{ds}.png"
        plt.savefig(nombre, dpi=300, facecolor='white')
        plt.close()
        print(f"  {nombre}")


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

    print(f"  Total: {len(df)} filas.\n")
    print("Generando curvas de tiempo (9 modelos)...")
    generar_graficos_acumulativos(df)
