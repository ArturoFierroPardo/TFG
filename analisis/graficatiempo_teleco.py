# pip install pandas matplotlib numpy
"""
Gráfico de Líneas Acumulativo — BBDD Teleco (7 modelos).
GAN vs Mini-SLM Base vs Mini-SLM Fine-Tuned.
Escala Log-Log. Un solo gráfico (un único dataset "Teleco").

USO: python graficatiempo_teleco.py --input-dir analisis
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
    'GAN':              '#E41A1C',
    'Gemma 3.1B':       '#E7298A',
    'Llama 1B':         '#66A61E',
    'Qwen3 1.7B':       '#D95F02',
    'Gemma 3.1B FT':    '#984EA3',
    'Llama 1B FT':      '#377EB8',
    'Qwen3 1.7B FT':    '#FF7F00',
}

# GAN: continua, Base: punteada, FT: discontinua
ESTILOS_LINEA = {
    'GAN':              '-',
    'Gemma 3.1B':       ':',
    'Llama 1B':         ':',
    'Qwen3 1.7B':       ':',
    'Gemma 3.1B FT':    '--',
    'Llama 1B FT':      '--',
    'Qwen3 1.7B FT':    '--',
}

MARCADORES = {
    'GAN':              'o',
    'Gemma 3.1B':       'P',
    'Llama 1B':         'X',
    'Qwen3 1.7B':       'h',
    'Gemma 3.1B FT':    's',
    'Llama 1B FT':      '^',
    'Qwen3 1.7B FT':    'D',
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


def generar_grafico_acumulativo(df):
    if 'Time_seconds' not in df.columns:
        print("Error: No se encuentra 'Time_seconds'.")
        return

    fig, ax = plt.subplots(figsize=(11, 6))

    modelos_presentes = [m for m in ORDEN_MODELOS if m in df['Modelo'].unique()]

    for modelo in modelos_presentes:
        df_modelo = df[df['Modelo'] == modelo]
        tiempos = df_modelo['Time_seconds'].dropna().values
        if len(tiempos) == 0:
            continue

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

    ax.set_title("Tiempo Total Acumulado — BBDD Teleco", fontsize=14, pad=15)
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
        for m in modelos_presentes
    ]

    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
              framealpha=0.95, facecolor='white', edgecolor='black',
              fancybox=False, handlelength=3.0)

    plt.tight_layout()
    nombre = "Acumulativo_Tiempo_Log_teleco.png"
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

    print("Cargando datos (BBDD Teleco — 7 modelos)...")
    df = cargar_todos(args.input_dir)
    if df.empty:
        print("Error: No se han cargado datos.")
        exit(1)

    print(f"  Total: {len(df)} filas.\n")
    print("Generando curva de tiempo acumulativo teleco...")
    generar_grafico_acumulativo(df)
