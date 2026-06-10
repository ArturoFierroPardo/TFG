# pip install pandas matplotlib numpy
"""
Gráficos de Líneas Acumulativos — G1 (9 modelos × 3 datasets) + G2 (7 modelos × Teleco).
Escala Log-Log. LLMs: continua. SLMs: discontinua. Mini-SLMs: punteada.
G2: GAN continua, Base punteada, FT discontinua.

USO: python graficatiempo.py --input-dir resultados --output-dir graficas_tiempo
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import glob, os, argparse
import numpy as np

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'xtick.direction': 'in', 'ytick.direction': 'in',
})

# ── G1 ────────────────────────────────────────────────────────────────────
COLORES_G1 = {
    'DeepSeek': '#2E75B6', 'Llama 70B': '#D6604D', 'Qwen 72B': '#E08214',
    'Gemma 9B': '#4393C3', 'Llama 3B': '#7570B3', 'Qwen 7B': '#1B7837',
    'Gemma 3.1B': '#E7298A', 'Llama 1B': '#66A61E', 'Qwen3 1.7B': '#D95F02',
}
ESTILOS_G1 = {
    'DeepSeek': '-', 'Llama 70B': '-', 'Qwen 72B': '-',
    'Gemma 9B': '--', 'Llama 3B': '--', 'Qwen 7B': '--',
    'Gemma 3.1B': ':', 'Llama 1B': ':', 'Qwen3 1.7B': ':',
}
MARCADORES_G1 = {
    'DeepSeek': 'o', 'Llama 70B': 's', 'Qwen 72B': '^',
    'Gemma 9B': 'D', 'Llama 3B': '*', 'Qwen 7B': 'v',
    'Gemma 3.1B': 'P', 'Llama 1B': 'X', 'Qwen3 1.7B': 'h',
}
ORDEN_G1 = ['DeepSeek', 'Llama 70B', 'Qwen 72B',
            'Gemma 9B', 'Llama 3B', 'Qwen 7B',
            'Gemma 3.1B', 'Llama 1B', 'Qwen3 1.7B']
DATASETS_G1 = ['WebNLG', 'ToTTo', 'KELM']

ARCHIVO_MAPA_G1 = [
    ('webNLG_Gemma_3_1B', 'Gemma 3.1B', 'WebNLG'),
    ('webNLG_Llama_1B', 'Llama 1B', 'WebNLG'),
    ('webNLG_Qwen3_1.7B', 'Qwen3 1.7B', 'WebNLG'),
    ('totto_Gemma_3_1B', 'Gemma 3.1B', 'ToTTo'),
    ('totto_Llama_1B', 'Llama 1B', 'ToTTo'),
    ('totto_Qwen3_1.7B', 'Qwen3 1.7B', 'ToTTo'),
    ('kelm_stem_Gemma_3_1B', 'Gemma 3.1B', 'KELM'),
    ('kelm_stem_Llama_1B', 'Llama 1B', 'KELM'),
    ('kelm_stem_Qwen3_1.7B', 'Qwen3 1.7B', 'KELM'),
    ('DeepSeek_WebNLG', 'DeepSeek', 'WebNLG'), ('DeepSeek_ToTTo', 'DeepSeek', 'ToTTo'), ('DeepSeek_KELM', 'DeepSeek', 'KELM'),
    ('Llama_70B_WebNLG', 'Llama 70B', 'WebNLG'), ('Llama_70B_ToTTo', 'Llama 70B', 'ToTTo'), ('Llama_70B_KELM', 'Llama 70B', 'KELM'),
    ('Qwen_72B_WebNLG', 'Qwen 72B', 'WebNLG'), ('Qwen_72B_ToTTo', 'Qwen 72B', 'ToTTo'), ('Qwen_72B_KELM', 'Qwen 72B', 'KELM'),
    ('Gemma_9B_WebNLG', 'Gemma 9B', 'WebNLG'), ('Gemma_9B_ToTTo', 'Gemma 9B', 'ToTTo'), ('Gemma_9B_KELM', 'Gemma 9B', 'KELM'),
    ('Llama_3B_WebNLG', 'Llama 3B', 'WebNLG'), ('Llama_3B_ToTTo', 'Llama 3B', 'ToTTo'), ('Llama_3B_KELM', 'Llama 3B', 'KELM'),
    ('Qwen_7B_WebNLG', 'Qwen 7B', 'WebNLG'), ('Qwen_7B_ToTTo', 'Qwen 7B', 'ToTTo'), ('Qwen_7B_KELM', 'Qwen 7B', 'KELM'),
]

# ── G2 ────────────────────────────────────────────────────────────────────
COLORES_G2 = {
    'GAN': '#E41A1C',
    'Gemma 3.1B': '#E7298A', 'Llama 1B': '#66A61E', 'Qwen3 1.7B': '#D95F02',
    'Gemma 3.1B FT': '#984EA3', 'Llama 1B FT': '#377EB8', 'Qwen3 1.7B FT': '#FF7F00',
}
ESTILOS_G2 = {
    'GAN': '-',
    'Gemma 3.1B': ':', 'Llama 1B': ':', 'Qwen3 1.7B': ':',
    'Gemma 3.1B FT': '--', 'Llama 1B FT': '--', 'Qwen3 1.7B FT': '--',
}
MARCADORES_G2 = {
    'GAN': 'o',
    'Gemma 3.1B': 'P', 'Llama 1B': 'X', 'Qwen3 1.7B': 'h',
    'Gemma 3.1B FT': 's', 'Llama 1B FT': '^', 'Qwen3 1.7B FT': 'D',
}
ORDEN_G2 = ['GAN', 'Gemma 3.1B', 'Llama 1B', 'Qwen3 1.7B',
            'Gemma 3.1B FT', 'Llama 1B FT', 'Qwen3 1.7B FT']

ARCHIVO_MAPA_G2 = [
    ('GAN_valtest', 'GAN', 'Teleco'),
    ('teleco_Gemma_3_1B', 'Gemma 3.1B', 'Teleco'),
    ('teleco_Llama_1B', 'Llama 1B', 'Teleco'),
    ('teleco_Qwen3_1.7B', 'Qwen3 1.7B', 'Teleco'),
    ('Gemma_3_1B_FT_valtest', 'Gemma 3.1B FT', 'Teleco'),
    ('Llama_1B_FT_valtest', 'Llama 1B FT', 'Teleco'),
    ('Qwen3_1.7B_FT_valtest', 'Qwen3 1.7B FT', 'Teleco'),
]


# ── Carga ─────────────────────────────────────────────────────────────────
def cargar(input_dir, archivo_mapa):
    df_total = pd.DataFrame()
    for archivo in glob.glob(os.path.join(input_dir, "metricas_por_fila_*.csv")):
        base = os.path.basename(archivo).replace("metricas_por_fila_", "").replace(".csv", "")
        modelo, dataset = None, None
        for pat, m, d in archivo_mapa:
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
    if 'Time_seconds' in df_total.columns:
        df_total['Time_seconds'] = pd.to_numeric(df_total['Time_seconds'], errors='coerce')
    return df_total


# ── Gráfico acumulativo genérico ──────────────────────────────────────────
def grafico_acumulativo(df, dataset, orden, colores, estilos, marcadores, titulo, fname):
    if 'Time_seconds' not in df.columns:
        print("  Sin columna Time_seconds")
        return

    df_ds = df[df['Dataset'] == dataset] if 'Dataset' in df.columns else df
    if df_ds.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    modelos_presentes = [m for m in orden if m in df_ds['Modelo'].unique()]

    for modelo in modelos_presentes:
        tiempos = df_ds[df_ds['Modelo'] == modelo]['Time_seconds'].dropna().values
        if len(tiempos) == 0:
            continue

        tiempo_acum = pd.Series(tiempos).cumsum().values
        eje_x = range(1, len(tiempo_acum) + 1)

        if len(eje_x) > 10:
            idx_marc = np.unique(np.geomspace(1, len(eje_x) - 1, num=12).astype(int))
        else:
            idx_marc = np.arange(len(eje_x))

        ax.plot(eje_x, tiempo_acum, label=modelo,
                color=colores.get(modelo, '#000'),
                linestyle=estilos.get(modelo, '-'),
                marker=marcadores.get(modelo, 'o'),
                linewidth=2.5, alpha=0.8, markersize=7,
                markevery=idx_marc.tolist())

    ax.set_title(titulo, fontsize=14, pad=15)
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
        Line2D([0], [0], color=colores[m], lw=2.5,
               linestyle=estilos[m], marker=marcadores[m],
               markersize=7, label=m)
        for m in modelos_presentes
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
              framealpha=0.95, facecolor='white', edgecolor='black',
              fancybox=False, handlelength=3.0)

    plt.tight_layout()
    plt.savefig(fname, dpi=300, facecolor='white')
    plt.close()
    print(f"  {fname}")


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="resultados")
    parser.add_argument("--output-dir", default="graficas_tiempo")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, args.input_dir) if not os.path.isabs(args.input_dir) else args.input_dir
    output_dir = os.path.join(script_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # G1
    print("Cargando G1 (9 modelos)...")
    df_g1 = cargar(input_dir, ARCHIVO_MAPA_G1)
    if not df_g1.empty:
        print(f"  {len(df_g1):,} filas")
        for ds in DATASETS_G1:
            if ds in df_g1['Dataset'].unique():
                fname = os.path.join(output_dir, f"Acumulativo_G1_{ds}.png")
                grafico_acumulativo(df_g1, ds, ORDEN_G1, COLORES_G1, ESTILOS_G1, MARCADORES_G1,
                                    f"Tiempo Total Acumulado - Dataset: {ds}", fname)

    # G2
    print("\nCargando G2 (7 modelos Teleco)...")
    df_g2 = cargar(input_dir, ARCHIVO_MAPA_G2)
    if not df_g2.empty:
        print(f"  {len(df_g2):,} filas")
        fname = os.path.join(output_dir, "Acumulativo_G2_Teleco.png")
        grafico_acumulativo(df_g2, 'Teleco', ORDEN_G2, COLORES_G2, ESTILOS_G2, MARCADORES_G2,
                            "Tiempo Total Acumulado — BBDD Teleco", fname)

    print("\nListo.")