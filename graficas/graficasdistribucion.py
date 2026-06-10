# pip install pandas matplotlib numpy seaborn
"""
Generador Maestro de Gráficas KDE (Montañas de Distribución).
- 13 Colores únicos.
- Filtra EXACTAMENTE las métricas y las BBDD que aplican a cada liga.
- Unifica 'teleco' y 'valtest' bajo el mismo nombre: 'Teleco'.
- Crea subcarpetas por modelo.

USO: python distribucion_fases.py --input-dir analisis
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import argparse

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})

# =====================================================================
# 1. PALETA DE COLORES (13 Colores Únicos)
# =====================================================================
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

# =====================================================================
# 2. SEPARACIÓN EXACTA DE MÉTRICAS
# =====================================================================
# Las 7 métricas para los modelos Comerciales y Minis
METRICAS_COMPLETAS = {
    'ROUGE_L':      'ROUGE-L',
    'METEOR':       'METEOR',
    'BERTScore':    'BERTScore',
    'BLEU':         'BLEU Score',
    'Time_seconds': 'Tiempo (Segundos)',
    'Coste_USD':    'Coste (USD)',
    'CO2_gramos':   'CO2 (Gramos)'
}

# Las 5 métricas para la GAN y el Finetuning
METRICAS_GAN_FT = {
    'ROUGE_L':      'ROUGE-L',
    'METEOR':       'METEOR',
    'BERTScore':    'BERTScore',
    'BLEU':         'BLEU Score',
    'Time_seconds': 'Tiempo (Segundos)'
}

# =====================================================================
# 3. LAS LIGAS (Configuración estricta de qué se cruza con qué)
# =====================================================================
ESCENARIOS = {
    "1_LLMs": {
        "datasets": ["WebNLG", "ToTTo", "KELM"],
        "modelos":  ['DeepSeek', 'Llama 70B', 'Qwen 72B'],
        "metricas": METRICAS_COMPLETAS
    },
    "2_SLMs": {
        "datasets": ["WebNLG", "ToTTo", "KELM", "Teleco"],
        "modelos":  ['Gemma 9B', 'Llama 3B', 'Qwen 7B'],
        "metricas": METRICAS_COMPLETAS
    },
    "3_MiniSLMs_Teleco": {
        "datasets": ["Teleco"], 
        "modelos":  ['Gemma 3 1B', 'Llama 1B', 'Qwen3 1.7B'],
        "metricas": METRICAS_COMPLETAS
    },
    "3b_MiniSLMs_Genericas": {
        "datasets": ["WebNLG", "ToTTo", "KELM"],
        "modelos":  ['Gemma 3 1B', 'Llama 1B', 'Qwen3 1.7B'],
        "metricas": METRICAS_COMPLETAS
    },
    "4_GAN_vs_Finetuning": {
        "datasets": ["Teleco"], 
        "modelos":  ['GAN', 'Gemma 3 1B FT', 'Llama 1B FT', 'Qwen3 1.7B FT'],
        "metricas": METRICAS_GAN_FT
    }
}

def cargar_todos(input_dir):
    df_total = pd.DataFrame()
    archivos = glob.glob(os.path.join(input_dir, "metricas_por_fila_*.csv"))
    
    for archivo in archivos:
        try:
            df_temp = pd.read_csv(archivo)
            base_name = os.path.basename(archivo).replace("metricas_por_fila_", "").replace(".csv", "")
            
            # --- Extractor inteligente de nombres y UNIFICADOR DE BBDD ---
            # Mini-SLMs en bases genéricas (formato invertido: dataset_modelo)
            if base_name.startswith("webNLG_"):
                ds = "WebNLG"
                modelo = base_name.replace("webNLG_", "")
            elif base_name.startswith("totto_"):
                ds = "ToTTo"
                modelo = base_name.replace("totto_", "")
            elif base_name.startswith("kelm_stem_"):
                ds = "KELM"
                modelo = base_name.replace("kelm_stem_", "")
            elif "valtest" in base_name:
                ds = "Teleco"
                modelo = base_name.replace("_valtest", "")
            elif "teleco" in base_name:
                ds = "Teleco"
                modelo = base_name.replace("teleco_", "")
            elif "WebNLG" in base_name:
                ds = "WebNLG"
                modelo = base_name.replace("_WebNLG", "")
            elif "ToTTo" in base_name:
                ds = "ToTTo"
                modelo = base_name.replace("_ToTTo", "")
            elif "KELM" in base_name:
                ds = "KELM"
                modelo = base_name.replace("_KELM", "")
            else:
                partes = base_name.split("_")
                ds = partes[-1]
                modelo = " ".join(partes[:-1])

            modelo = modelo.replace("_", " ")
            mod_low = modelo.lower()
            
            # Normalización
            if mod_low == 'gemma 9b': modelo = 'Gemma 9B'
            elif mod_low == 'deepseek': modelo = 'DeepSeek'
            elif mod_low == 'llama 70b': modelo = 'Llama 70B'
            elif mod_low == 'qwen 72b': modelo = 'Qwen 72B'
            elif mod_low == 'llama 3b': modelo = 'Llama 3B'
            elif mod_low == 'qwen 7b': modelo = 'Qwen 7B'
            elif mod_low == 'gemma 3 1b': modelo = 'Gemma 3 1B'
            elif mod_low == 'llama 1b': modelo = 'Llama 1B'
            elif mod_low == 'qwen3 1.7b': modelo = 'Qwen3 1.7B'
            elif mod_low == 'gemma 3 1b ft': modelo = 'Gemma 3 1B FT'
            elif mod_low == 'llama 1b ft': modelo = 'Llama 1B FT'
            elif mod_low == 'qwen3 1.7b ft': modelo = 'Qwen3 1.7B FT'
            elif mod_low == 'gan': modelo = 'GAN'

            df_temp['Modelo'] = modelo
            df_temp['Dataset'] = ds

            # Forzar conversión a numérico (buscando entre TODAS las posibles métricas)
            for col_metrica in METRICAS_COMPLETAS.keys():
                if col_metrica in df_temp.columns:
                    df_temp[col_metrica] = pd.to_numeric(df_temp[col_metrica], errors='coerce')
            
            df_total = pd.concat([df_total, df_temp], ignore_index=True)
        except Exception as e:
            print(f"Error cargando {archivo}: {e}")
    return df_total

def crear_carpetas():
    os.makedirs("Graficas_Individuales", exist_ok=True)
    os.makedirs("Graficas_Agrupadas", exist_ok=True)


def filtrar_outliers_iqr(serie, factor=3.0):
    """Filtra outliers extremos usando IQR. Devuelve serie filtrada."""
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    if iqr <= 0:
        # IQR es 0 → usar percentiles P1-P99
        lb = serie.quantile(0.01)
        ub = serie.quantile(0.99)
        if lb == ub:
            return serie
    else:
        lb = q1 - factor * iqr
        ub = q3 + factor * iqr
    filtered = serie[(serie >= lb) & (serie <= ub)]
    if len(filtered) < 5:
        return serie
    return filtered


def ajustar_xlim(ax, serie_filtrada):
    """Ajusta xlim al rango de datos filtrados con margen visual a ambos lados.
    No corta en pared — deja que la curva KDE fluya naturalmente."""
    vmin = serie_filtrada.min()
    vmax = serie_filtrada.max()
    rango = vmax - vmin
    if rango <= 0:
        rango = max(abs(vmax), 1)
    margen = rango * 0.08  # 8% de margen a cada lado
    ax.set_xlim(vmin - margen, vmax + margen)

def generar_individuales(df):
    generados = set() # Para evitar que una gráfica se repita si un modelo aparece en varias fases
    
    for escenario_nombre, config in ESCENARIOS.items():
        for ds in config["datasets"]:
            for modelo in config["modelos"]:
                
                # Creamos subcarpeta del modelo
                nombre_carpeta = modelo.replace(" ", "_")
                ruta_carpeta = os.path.join("Graficas_Individuales", nombre_carpeta)
                os.makedirs(ruta_carpeta, exist_ok=True)
                
                for col_metrica, nombre_metrica in config["metricas"].items():
                    clave_grafica = (modelo, ds, col_metrica)
                    if clave_grafica in generados: continue
                    generados.add(clave_grafica)

                    if col_metrica not in df.columns: continue

                    df_filtrado = df[(df['Dataset'] == ds) & (df['Modelo'] == modelo) & (df[col_metrica].notna())]
                    if df_filtrado.empty: continue

                    # Filtrar outliers extremos para que el KDE no se deforme
                    serie_limpia = filtrar_outliers_iqr(df_filtrado[col_metrica])
                    if len(serie_limpia) < 5: continue
                    df_plot = df_filtrado.loc[serie_limpia.index]

                    fig, ax = plt.subplots(figsize=(8, 5))
                    color = COLORES.get(modelo, '#888888')

                    sns.kdeplot(
                        data=df_plot, x=col_metrica, fill=True,
                        alpha=0.4, linewidth=2.0, color=color, ax=ax
                    )

                    # Eje X al rango filtrado con margen natural
                    ajustar_xlim(ax, serie_limpia)

                    ax.set_title(f"Distribución {nombre_metrica}\nModelo: {modelo} | Dataset: {ds}", fontsize=13, pad=15)
                    ax.set_xlabel(nombre_metrica, fontsize=12)
                    ax.set_ylabel("Densidad", fontsize=12)
                    ax.grid(True, linestyle='-', alpha=0.3, color='#888888')
                    ax.set_axisbelow(True)

                    plt.tight_layout()
                    nombre_archivo = f"{ruta_carpeta}/Indiv_{nombre_carpeta}_{col_metrica}_{ds}.png"
                    plt.savefig(nombre_archivo, dpi=300, facecolor='white')
                    plt.close()

def generar_agrupadas(df):
    for escenario_nombre, config in ESCENARIOS.items():
        for ds in config["datasets"]:
            for col_metrica, nombre_metrica in config["metricas"].items():
                if col_metrica not in df.columns: continue

                df_filtrado = df[(df['Dataset'] == ds) & (df['Modelo'].isin(config["modelos"])) & (df[col_metrica].notna())]
                
                # Comprobamos que haya datos de al menos un modelo para dibujar
                if df_filtrado.empty: continue

                # Filtrar outliers extremos por modelo para que el KDE no se deforme
                dfs_limpios = []
                all_series_limpias = []
                for m in config["modelos"]:
                    df_m = df_filtrado[df_filtrado['Modelo'] == m]
                    if df_m.empty: continue
                    serie_limpia = filtrar_outliers_iqr(df_m[col_metrica])
                    if len(serie_limpia) >= 5:
                        dfs_limpios.append(df_m.loc[serie_limpia.index])
                        all_series_limpias.append(serie_limpia)
                if not dfs_limpios: continue
                df_plot = pd.concat(dfs_limpios, ignore_index=True)
                serie_global = pd.concat(all_series_limpias, ignore_index=True)

                fig, ax = plt.subplots(figsize=(10, 6))

                sns.kdeplot(
                    data=df_plot, x=col_metrica, hue='Modelo', fill=True,           
                    alpha=0.25, linewidth=2.0, palette=COLORES, common_norm=False,   
                    ax=ax, hue_order=[m for m in config["modelos"] if m in df_plot['Modelo'].unique()]
                )

                # Eje X al rango filtrado con margen natural
                ajustar_xlim(ax, serie_global)

                # === EL TÍTULO LIMPIO ===
                ax.set_title(f"Comparativa de Distribución\n{nombre_metrica} - Dataset: {ds}", fontsize=14, pad=15)
                # ========================

                ax.set_xlabel(nombre_metrica, fontsize=12)
                ax.set_ylabel("Densidad (Frecuencia)", fontsize=12)
                ax.grid(True, linestyle='-', alpha=0.3, color='#888888')
                ax.set_axisbelow(True)

                if ax.get_legend() is not None:
                    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1), 
                                    framealpha=1.0, edgecolor='black', title='Modelos')
                    plt.tight_layout()

                # El nombre del archivo lo dejamos como estaba para que las siga organizando bien
                nombre_archivo = f"Graficas_Agrupadas/Comp_{escenario_nombre}_{col_metrica}_{ds}.png"
                plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="analisis")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: No se encuentra '{args.input_dir}'")
        exit(1)

    print("Cargando, limpiando y unificando nombres a 'Teleco'...")
    df = cargar_todos(args.input_dir)
    
    if df.empty:
        print("Error: No se han cargado datos.")
        exit(1)

    crear_carpetas()
    print("Generando gráficas individuales restringidas por escenario...")
    generar_individuales(df)
    print("Generando gráficas agrupadas restringidas por escenario...")
    generar_agrupadas(df)
    print("\n¡Listo! Generación completada con éxito.")