# pip install pandas matplotlib numpy seaborn scipy
"""
Generador Maestro de Q-Q Plots para TODAS las Fases del TFG.
- Filtra valores imposibles (>1.0) para evitar ejes deformes.
- Etiquetas claras para el tribunal.
- Subcarpetas por Fases y por Base de Datos.
- Títulos de fase corregidos y elegantes.

USO: python qq_plots_maestro.py --input-dir analisis
"""
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import glob
import os
import argparse

# Estilo visual Paper
plt.rcParams.update({
    'font.size': 11,
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

# 7 métricas para modelos comerciales/minis
METRICAS_COMPLETAS = {
    'ROUGE_L':      'ROUGE-L',
    'METEOR':       'METEOR',
    'BERTScore':    'BERTScore',
    'BLEU':         'BLEU Score',
    'Time_seconds': 'Tiempo (Segundos)',
    'Coste_USD':    'Coste (USD)',
    'CO2_gramos':   'CO2 (Gramos)'
}

# 5 métricas para GAN/FT
METRICAS_GAN_FT = {
    'ROUGE_L':      'ROUGE-L',
    'METEOR':       'METEOR',
    'BERTScore':    'BERTScore',
    'BLEU':         'BLEU Score',
    'Time_seconds': 'Tiempo (Segundos)'
}

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
    "4_GAN_vs_Finetuning": {
        "datasets": ["Teleco"], 
        "modelos":  ['GAN', 'Gemma 3 1B FT', 'Llama 1B FT', 'Qwen3 1.7B FT'],
        "metricas": METRICAS_GAN_FT
    }
}

# === TRADUCTOR DE TÍTULOS ELEGANTE ===
TITULOS_FASES = {
    "1_LLMs": "LLMs",
    "2_SLMs": "SLMs",
    "3_MiniSLMs_Teleco": "Mini-SLMs",
    "4_GAN_vs_Finetuning": "GAN y Fine-Tuning"
}
# ====================================

def cargar_todos(input_dir):
    df_total = pd.DataFrame()
    archivos = glob.glob(os.path.join(input_dir, "metricas_por_fila_*.csv"))
    
    for archivo in archivos:
        try:
            df_temp = pd.read_csv(archivo)
            base_name = os.path.basename(archivo).replace("metricas_por_fila_", "").replace(".csv", "")
            
            # Unificador de BBDD
            if "valtest" in base_name: ds = "Teleco"; modelo = base_name.replace("_valtest", "")
            elif "teleco" in base_name: ds = "Teleco"; modelo = base_name.replace("teleco_", "")
            elif "WebNLG" in base_name: ds = "WebNLG"; modelo = base_name.replace("_WebNLG", "")
            elif "ToTTo" in base_name: ds = "ToTTo"; modelo = base_name.replace("_ToTTo", "")
            elif "KELM" in base_name: ds = "KELM"; modelo = base_name.replace("_KELM", "")
            else:
                partes = base_name.split("_")
                ds = partes[-1]
                modelo = " ".join(partes[:-1])

            modelo = modelo.replace("_", " ")
            mod_low = modelo.lower()
            
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

            # 1. Forzar numéricos
            for col_metrica in METRICAS_COMPLETAS.keys():
                if col_metrica in df_temp.columns:
                    df_temp[col_metrica] = pd.to_numeric(df_temp[col_metrica], errors='coerce')
            
            # 2. ELIMINAR ERRORES OUTLIERS (>1.0) EN PUNTUACIONES
            metricas_puntuacion = ['ROUGE_L', 'METEOR', 'BERTScore', 'BLEU']
            for col in metricas_puntuacion:
                if col in df_temp.columns:
                    df_temp = df_temp[df_temp[col] <= 1.0]

            df_total = pd.concat([df_total, df_temp], ignore_index=True)
        except Exception as e:
            print(f"Error cargando {archivo}: {e}")
    return df_total

def generar_qq_plots_maestro(df):
    for escenario_nombre, config in ESCENARIOS.items():
        for ds in config["datasets"]:
            
            # Crear subcarpetas para tenerlo todo hiper-ordenado
            ruta_carpeta = f"Graficas_QQPlot/{escenario_nombre}/{ds}"
            os.makedirs(ruta_carpeta, exist_ok=True)
            
            for col_metrica, nombre_metrica in config["metricas"].items():
                if col_metrica not in df.columns: continue

                df_filtrado = df[(df['Dataset'] == ds) & (df['Modelo'].isin(config["modelos"])) & (df[col_metrica].notna())]
                if df_filtrado.empty: continue

                # Creamos el mosaico 2x2
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # === USAMOS EL TÍTULO TRADUCIDO ===
                titulo_fase = TITULOS_FASES.get(escenario_nombre, escenario_nombre)
                fig.suptitle(f"Q-Q Plot de Normalidad: {nombre_metrica}\nDataset: {ds} ({titulo_fase})", fontsize=16, y=0.95)
                # ==================================
                
                axes = axes.flatten()

                for idx, modelo in enumerate(config["modelos"]):
                    ax = axes[idx]
                    datos_modelo = df_filtrado[df_filtrado['Modelo'] == modelo][col_metrica].values
                    
                    if len(datos_modelo) > 5:
                        res = stats.probplot(datos_modelo, dist="norm", plot=ax)
                        
                        lineas = ax.get_lines()
                        color_modelo = COLORES.get(modelo, '#000000')
                        
                        lineas[0].set_markerfacecolor('none')
                        lineas[0].set_markeredgecolor(color_modelo)
                        lineas[0].set_markersize(4)
                        lineas[0].set_alpha(0.6)
                        
                        lineas[1].set_color('#444444')
                        lineas[1].set_linestyle('--')
                        lineas[1].set_linewidth(1.5)
                        
                        ax.set_title(modelo, fontsize=13, pad=10, color=color_modelo, fontweight='bold')
                    else:
                        ax.text(0.5, 0.5, "Datos insuficientes", ha='center', va='center')
                        ax.set_title(modelo)

                    ax.set_xlabel("Distribución Normal Teórica")
                    ax.set_ylabel(f"Puntuación Real ({nombre_metrica})")
                    ax.grid(True, linestyle=':', alpha=0.6)
                    ax.set_axisbelow(True)

                # Si el escenario tiene 3 modelos, ocultamos el cuarto hueco (abajo a la derecha)
                if len(config["modelos"]) == 3:
                    fig.delaxes(axes[3])

                plt.tight_layout()
                plt.subplots_adjust(top=0.88)
                
                nombre_archivo = f"{ruta_carpeta}/QQ_{col_metrica}.png"
                plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
                plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="analisis")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: No se encuentra '{args.input_dir}'")
        exit(1)

    print("Cargando y purificando datos (>1.0 filtrados)...")
    df = cargar_todos(args.input_dir)
    
    if df.empty:
        print("Error: No se han cargado datos.")
        exit(1)

    print("Generando TODOS los Q-Q Plots con títulos corregidos...")
    generar_qq_plots_maestro(df)
    print("\n¡Listísimo! Revisa la carpeta 'Graficas_QQPlot'.")