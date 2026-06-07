# pip install pandas numpy scipy scikit-posthocs matplotlib
"""
CD Diagrams mejorados + Reporte Post-Hoc:
  - Nombres de modelo más grandes
  - Leyenda LLM / SLM / Mini-SLM
  - Por separado: un CD por métrica × dataset
  - Juntos: las 4 métricas de calidad en un grid 2×2 por dataset
  - Reporte: Reporte_PostHoc.txt + resultados_posthoc_pares.csv

USO: python posthoc.py --input-dir analisis
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob, os, argparse
from itertools import combinations

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})

# ── Config ────────────────────────────────────────────────────────────────
METRICAS_CALIDAD = {
    'ROUGE_L':   'ROUGE-L',
    'METEOR':    'METEOR',
    'BERTScore': 'BERTScore',
    'BLEU':      'BLEU',
}

GRUPO_1 = {
    "nombre": "Grupo 1: LLM vs SLM vs Mini-SLM (9 modelos)",
    "datasets": ["WebNLG", "ToTTo", "KELM"],
    "modelos": [
        'DeepSeek', 'Llama 70B', 'Qwen 72B',
        'Gemma 9B', 'Llama 3B', 'Qwen 7B',
        'Gemma 3 1B', 'Llama 1B', 'Qwen3 1.7B',
    ],
}

GRUPO_2 = {
    "nombre": "Grupo 2: GAN vs Mini-SLM vs FT (7 modelos)",
    "datasets": ["Teleco"],
    "modelos": [
        'GAN',
        'Gemma 3 1B', 'Llama 1B', 'Qwen3 1.7B',
        'Gemma 3 1B FT', 'Llama 1B FT', 'Qwen3 1.7B FT',
    ],
}

ARCHIVO_MAPA = [
    ('webNLG_Gemma_3_1B',    'Gemma 3 1B', 'WebNLG'),
    ('webNLG_Llama_1B',      'Llama 1B',   'WebNLG'),
    ('webNLG_Qwen3_1.7B',    'Qwen3 1.7B', 'WebNLG'),
    ('totto_Gemma_3_1B',     'Gemma 3 1B', 'ToTTo'),
    ('totto_Llama_1B',       'Llama 1B',   'ToTTo'),
    ('totto_Qwen3_1.7B',     'Qwen3 1.7B', 'ToTTo'),
    ('kelm_stem_Gemma_3_1B', 'Gemma 3 1B', 'KELM'),
    ('kelm_stem_Llama_1B',   'Llama 1B',   'KELM'),
    ('kelm_stem_Qwen3_1.7B', 'Qwen3 1.7B', 'KELM'),
    ('DeepSeek_WebNLG',      'DeepSeek',   'WebNLG'),
    ('DeepSeek_ToTTo',       'DeepSeek',   'ToTTo'),
    ('DeepSeek_KELM',        'DeepSeek',   'KELM'),
    ('Llama_70B_WebNLG',     'Llama 70B',  'WebNLG'),
    ('Llama_70B_ToTTo',      'Llama 70B',  'ToTTo'),
    ('Llama_70B_KELM',       'Llama 70B',  'KELM'),
    ('Qwen_72B_WebNLG',      'Qwen 72B',   'WebNLG'),
    ('Qwen_72B_ToTTo',       'Qwen 72B',   'ToTTo'),
    ('Qwen_72B_KELM',        'Qwen 72B',   'KELM'),
    ('Gemma_9B_WebNLG',      'Gemma 9B',   'WebNLG'),
    ('Gemma_9B_ToTTo',       'Gemma 9B',   'ToTTo'),
    ('Gemma_9B_KELM',        'Gemma 9B',   'KELM'),
    ('Llama_3B_WebNLG',      'Llama 3B',   'WebNLG'),
    ('Llama_3B_ToTTo',       'Llama 3B',   'ToTTo'),
    ('Llama_3B_KELM',        'Llama 3B',   'KELM'),
    ('Qwen_7B_WebNLG',       'Qwen 7B',    'WebNLG'),
    ('Qwen_7B_ToTTo',        'Qwen 7B',    'ToTTo'),
    ('Qwen_7B_KELM',         'Qwen 7B',    'KELM'),
    # Grupo 2 — Teleco
    ('teleco_Gemma_3_1B',     'Gemma 3 1B',    'Teleco'),
    ('teleco_Llama_1B',       'Llama 1B',      'Teleco'),
    ('teleco_Qwen3_1.7B',     'Qwen3 1.7B',    'Teleco'),
    ('GAN_valtest',           'GAN',           'Teleco'),
    ('Gemma_3_1B_FT_valtest', 'Gemma 3 1B FT', 'Teleco'),
    ('Llama_1B_FT_valtest',   'Llama 1B FT',   'Teleco'),
    ('Qwen3_1.7B_FT_valtest', 'Qwen3 1.7B FT', 'Teleco'),
]

# Colores G1: LLM→azules, SLM→verdes, Mini-SLM→rojos/rosas
COLORES = {
    'DeepSeek':   '#1A5276',
    'Llama 70B':  '#2E86C1',
    'Qwen 72B':   '#7FB3D3',
    'Gemma 9B':   '#1E8449',
    'Llama 3B':   '#27AE60',
    'Qwen 7B':    '#76D7A3',
    'Gemma 3 1B': '#C0392B',
    'Llama 1B':   '#E91E8C',
    'Qwen3 1.7B': '#F1948A',
}

COLOR_DEFECTO_G1 = '#888888'

COLORES_LEYENDA = {
    'LLM':      '#2E86C1',
    'SLM':      '#27AE60',
    'Mini-SLM': '#E91E8C',
}

# Colores G2
COLORES_G2 = {
    'GAN':           '#17BECF',
    'Gemma 3 1B':    '#E91E8C',
    'Llama 1B':      '#66AA00',
    'Qwen3 1.7B':    '#B82E2E',
    'Gemma 3 1B FT': '#E67300',
    'Llama 1B FT':   '#8B0707',
    'Qwen3 1.7B FT': '#329262',
}

COLOR_DEFECTO_G2 = '#888888'

COLORES_LEYENDA_G2 = {
    'GAN':       '#17BECF',
    'Mini-SLM':  '#E91E8C',
    'Fine-Tuned':'#E67300',
}


def _paleta_completa(modelos_presentes, colores_ref, color_defecto):
    """
    Devuelve un dict con una entrada por cada modelo presente.
    Si el modelo no está en colores_ref, usa color_defecto.
    critical_difference_diagram requiere que TODOS los modelos tengan color.
    """
    return {m: colores_ref.get(m, color_defecto) for m in modelos_presentes}


# ── Carga datos ───────────────────────────────────────────────────────────
def cargar_todos(input_dir):
    df_total = pd.DataFrame()
    for archivo in glob.glob(os.path.join(input_dir, "metricas_por_fila_*.csv")):
        base = os.path.basename(archivo).replace("metricas_por_fila_", "").replace(".csv", "")
        modelo, dataset = None, None
        for patron, m, d in ARCHIVO_MAPA:
            if base == patron:
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
            print(f"  [WARN] No se pudo leer {archivo}: {e}")

    # Convertir métricas a numérico — filtrar por columna de forma independiente
    # (no acumular el filtro entre columnas, que elimina filas válidas)
    for c in list(METRICAS_CALIDAD.keys()):
        if c in df_total.columns:
            df_total[c] = pd.to_numeric(df_total[c], errors='coerce')
            # Solo BLEU está garantizado en [0,1]; BERTScore y METEOR pueden superar 1
            # → solo filtramos valores claramente erróneos (>100)
            df_total.loc[df_total[c] > 100, c] = np.nan

    return df_total


# ── Leyendas ──────────────────────────────────────────────────────────────
def añadir_leyenda(fig, bajo_grid=False):
    modelos_g1_ord = ['DeepSeek','Llama 70B','Qwen 72B',
                      'Gemma 9B','Llama 3B','Qwen 7B',
                      'Gemma 3 1B','Llama 1B','Qwen3 1.7B']
    handles_mod = [mpatches.Patch(color=COLORES[m], label=m) for m in modelos_g1_ord]
    y_mod = -0.03 if bajo_grid else -0.12
    fig.legend(handles=handles_mod, loc='lower center',
               bbox_to_anchor=(0.5, y_mod), ncol=9,
               fontsize=16, framealpha=0.95, edgecolor='#CCCCCC',
               title='Modelo', title_fontsize=16)


def añadir_leyenda_g2(fig, bajo_grid=False):
    modelos_g2_ord = ['GAN','Gemma 3 1B','Llama 1B','Qwen3 1.7B',
                      'Gemma 3 1B FT','Llama 1B FT','Qwen3 1.7B FT']
    handles_mod = [mpatches.Patch(color=COLORES_G2[m], label=m) for m in modelos_g2_ord]
    y_mod = -0.03 if bajo_grid else -0.12
    fig.legend(handles=handles_mod, loc='lower center',
               bbox_to_anchor=(0.5, y_mod), ncol=7,
               fontsize=16, framealpha=0.95, edgecolor='#CCCCCC',
               title='Modelo', title_fontsize=16)


# ── CD diagram individual G1 ──────────────────────────────────────────────
def cd_individual(datos, titulo, fname):
    df_m = pd.DataFrame(datos)
    ranks = df_m.rank(axis=1, ascending=False).mean()
    try:
        nemenyi = sp.posthoc_nemenyi_friedman(df_m)
    except Exception as e:
        print(f"  [WARN] Nemenyi error en '{titulo}': {e}"); return

    paleta = _paleta_completa(ranks.index.tolist(), COLORES, COLOR_DEFECTO_G1)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    try:
        sp.critical_difference_diagram(
            ranks=ranks, sig_matrix=nemenyi, ax=ax,
            label_fmt_left='{label} ({rank:.2f})',
            label_fmt_right='({rank:.2f}) {label}',
            label_props={'fontsize': 12, 'fontweight': 'bold'},
            crossbar_props={'linewidth': 2.5, 'color': '#333333'},
            marker_props={'s': 70, 'zorder': 10},
            elbow_props={'linewidth': 1.0},
            text_h_margin=0.04,
            color_palette=paleta,
        )
    except Exception as e:
        print(f"  [WARN] CD error en '{titulo}': {e}"); plt.close(); return

    ax.set_title(titulo, fontsize=13, pad=12)
    plt.tight_layout()
    añadir_leyenda(fig, bajo_grid=False)
    plt.savefig(fname, dpi=200, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"  OK: {fname}")


# ── CD grid 2×2 G1 ────────────────────────────────────────────────────────
def cd_grid_4metricas(datos_dict, ds, fname):
    metricas_orden = ['ROUGE_L', 'METEOR', 'BLEU', 'BERTScore']
    nombres = METRICAS_CALIDAD

    fig, axes = plt.subplots(2, 2, figsize=(22, 10))
    fig.suptitle(f'Dataset: {ds}', fontsize=17, fontweight='bold', y=1.01)

    for idx, col_met in enumerate(metricas_orden):
        ax = axes[idx // 2][idx % 2]
        datos = datos_dict.get(col_met)
        if datos is None:
            ax.set_visible(False); continue

        df_m = pd.DataFrame(datos)
        ranks = df_m.rank(axis=1, ascending=False).mean()
        try:
            nemenyi = sp.posthoc_nemenyi_friedman(df_m)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {e}', ha='center', transform=ax.transAxes); continue

        paleta = _paleta_completa(ranks.index.tolist(), COLORES, COLOR_DEFECTO_G1)

        try:
            sp.critical_difference_diagram(
                ranks=ranks,
                sig_matrix=nemenyi,
                ax=ax,
                label_fmt_left='{label} ({rank:.2f})',
                label_fmt_right='({rank:.2f}) {label}',
                label_props={'fontsize': 18, 'fontweight': 'bold'},
                crossbar_props={'linewidth': 2.5, 'color': '#333333'},
                marker_props={'s': 60, 'zorder': 10},
                elbow_props={'linewidth': 0.9},
                text_h_margin=0.04,
                color_palette=paleta,
            )
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {e}', ha='center', transform=ax.transAxes); continue

        ax.tick_params(axis='x', labelsize=16)
        letras = ['(a)', '(b)', '(c)', '(d)']
        ax.set_title(f"{letras[idx]} {nombres[col_met]}", fontsize=17, pad=10)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    añadir_leyenda(fig, bajo_grid=True)
    plt.savefig(fname, dpi=200, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"  OK grid: {fname}")


# ── CD diagram individual G2 ──────────────────────────────────────────────
def cd_individual_g2(datos, titulo, fname):
    df_m = pd.DataFrame(datos)
    ranks = df_m.rank(axis=1, ascending=False).mean()
    try:
        nemenyi = sp.posthoc_nemenyi_friedman(df_m)
    except Exception as e:
        print(f"  [WARN] Nemenyi error en '{titulo}': {e}"); return

    paleta = _paleta_completa(ranks.index.tolist(), COLORES_G2, COLOR_DEFECTO_G2)

    fig, ax = plt.subplots(figsize=(14, 6))
    try:
        sp.critical_difference_diagram(
            ranks=ranks, sig_matrix=nemenyi, ax=ax,
            label_fmt_left='{label} ({rank:.2f})',
            label_fmt_right='({rank:.2f}) {label}',
            label_props={'fontsize': 12, 'fontweight': 'bold'},
            crossbar_props={'linewidth': 2.5, 'color': '#333333'},
            marker_props={'s': 70, 'zorder': 10},
            elbow_props={'linewidth': 1.0},
            text_h_margin=0.08,
            color_palette=paleta,
        )
    except Exception as e:
        print(f"  [WARN] CD error en '{titulo}': {e}"); plt.close(); return

    ax.set_title(titulo, fontsize=13, pad=12)
    plt.tight_layout()
    añadir_leyenda_g2(fig, bajo_grid=False)
    plt.savefig(fname, dpi=200, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"  OK: {fname}")


# ── CD grid 2×2 G2 ────────────────────────────────────────────────────────
def cd_grid_4metricas_g2(datos_dict, fname):
    metricas_orden = ['ROUGE_L', 'METEOR', 'BLEU', 'BERTScore']
    nombres = METRICAS_CALIDAD

    fig, axes = plt.subplots(2, 2, figsize=(28, 14))
    fig.suptitle('Dataset: Teleco', fontsize=20, fontweight='bold', y=1.005)

    for idx, col_met in enumerate(metricas_orden):
        ax = axes[idx // 2][idx % 2]
        datos = datos_dict.get(col_met)
        if datos is None:
            ax.set_visible(False); continue

        df_m = pd.DataFrame(datos)
        ranks = df_m.rank(axis=1, ascending=False).mean()
        try:
            nemenyi = sp.posthoc_nemenyi_friedman(df_m)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {e}', ha='center', transform=ax.transAxes); continue

        paleta = _paleta_completa(ranks.index.tolist(), COLORES_G2, COLOR_DEFECTO_G2)

        try:
            sp.critical_difference_diagram(
                ranks=ranks, sig_matrix=nemenyi, ax=ax,
                label_fmt_left='{label} ({rank:.2f})',
                label_fmt_right='({rank:.2f}) {label}',
                label_props={'fontsize': 18, 'fontweight': 'bold'},
                crossbar_props={'linewidth': 2.5, 'color': '#333333'},
                marker_props={'s': 70, 'zorder': 10},
                elbow_props={'linewidth': 1.0},
                text_h_margin=0.08,
                color_palette=paleta,
            )
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {e}', ha='center', transform=ax.transAxes); continue

        ax.tick_params(axis='x', labelsize=18)
        letras = ['(a)', '(b)', '(c)', '(d)']
        ax.set_title(f"{letras[idx]} {nombres[col_met]}", fontsize=20, pad=10)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    añadir_leyenda_g2(fig, bajo_grid=True)
    plt.savefig(fname, dpi=200, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"  OK grid: {fname}")


# ── Métricas del reporte (incluye Tiempo y CO2 además de calidad) ─────────
METRICAS_REPORTE = {
    'ROUGE_L':      ('ROUGE-L',           'Mayor es mejor'),
    'METEOR':       ('METEOR',            'Mayor es mejor'),
    'BERTScore':    ('BERTScore',         'Mayor es mejor'),
    'BLEU':         ('BLEU',              'Mayor es mejor'),
    'Time_seconds': ('Tiempo (Segundos)', 'Menor es mejor'),
    'CO2_gramos':   ('CO₂ (Gramos)',      'Menor es mejor'),
}


# ── Reporte Post-Hoc ──────────────────────────────────────────────────────
def formatear_p(p):
    if p == 0.0 or p < 1e-300:
        return "< 0.001"
    elif p < 0.001:
        return "< 0.001"
    else:
        return f"{p:.4f}"


def estrellas_p(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else: return "n.s."


def clasificar_efecto(r_abs):
    if r_abs < 0.1: return "nulo"
    elif r_abs < 0.3: return "pequeño"
    elif r_abs < 0.5: return "mediano"
    else: return "grande"


def wilcoxon_r(a, b):
    """Calcula Wilcoxon signed-rank r = |Z| / √N. Robusto ante overflow y versiones de scipy."""
    try:
        n = len(a)
        if n < 10:
            return 0.0, 1.0

        result = stats.wilcoxon(a, b)

        # Intentar obtener z directamente (scipy >= 1.9)
        if hasattr(result, 'zstatistic'):
            z = abs(result.zstatistic)
        else:
            # Aproximación manual Z = (T - μ) / σ
            T = float(result.statistic)
            diff = a - b
            n_eff = float(np.sum(diff != 0))
            if n_eff < 2:
                return 0.0, result.pvalue
            mu = n_eff * (n_eff + 1.0) / 4.0
            sigma = np.sqrt(n_eff * (n_eff + 1.0) * (2.0 * n_eff + 1.0) / 24.0)
            if sigma == 0 or np.isnan(sigma):
                return 0.0, result.pvalue
            z = abs(T - mu) / sigma

        r = z / np.sqrt(float(n))
        if np.isnan(r) or np.isinf(r):
            r = 0.0
        return min(r, 1.0), result.pvalue
    except Exception:
        return 0.0, 1.0


def ejecutar_posthoc_completo(datos, grupo_nombre, dataset, metrica_col, metrica_nombre, direccion):
    """Ejecuta Nemenyi + Wilcoxon por pares. Devuelve dict con todo."""
    df_m = pd.DataFrame(datos)

    try:
        nemenyi = sp.posthoc_nemenyi_friedman(df_m)
    except Exception as e:
        print(f"  [WARN] Nemenyi error {metrica_nombre}/{dataset}: {e}")
        return None

    modelos = list(datos.keys())
    n_pareado = len(df_m)
    pares = []

    for m1, m2 in combinations(modelos, 2):
        a = datos[m1]
        b = datos[m2]
        delta = np.mean(a) - np.mean(b)
        p_adj = nemenyi.loc[m1, m2]
        sig = p_adj < 0.05
        r_val, _ = wilcoxon_r(a, b)
        efecto = clasificar_efecto(r_val)

        pares.append({
            'm1': m1, 'm2': m2,
            'delta': delta,
            'p_adj': p_adj,
            'sig': sig,
            'r': r_val,
            'efecto': efecto,
        })

    return {
        'grupo': grupo_nombre,
        'dataset': dataset,
        'metrica': metrica_nombre,
        'col': metrica_col,
        'direccion': direccion,
        'n_pareado': n_pareado,
        'n_pares': len(pares),
        'pares': pares,
    }


def generar_reporte_posthoc(todos_resultados, output_dir):
    """Genera Reporte_PostHoc.txt con formato Nemenyi + Wilcoxon (r)."""
    ruta = os.path.join(output_dir, "Reporte_PostHoc.txt")

    with open(ruta, "w", encoding="utf-8") as f:
        f.write("═"*70 + "\n")
        f.write("   POST-HOC: Nemenyi (p-adj) + Wilcoxon (r)\n")
        f.write("═"*70 + "\n")
        f.write(f"\nMetodología:\n")
        f.write(f"  • Todas las métricas → Nemenyi (p-adj) + Wilcoxon (r = Z/√N)\n")
        f.write(f"  • α = 0.05\n")
        f.write(f"  • r: nulo(<0.1) pequeño(<0.3) mediano(<0.5) grande(≥0.5)\n")
        f.write(f"  • d: nulo(<0.2) pequeño(<0.5) mediano(<0.8) grande(≥0.8)\n")

        grupo_actual = None
        ds_actual = None

        for r in todos_resultados:
            # Cabecera de grupo
            if r['grupo'] != grupo_actual:
                grupo_actual = r['grupo']
                ds_actual = None
                f.write(f"\n\n{'='*70}\n")
                f.write(f"  {grupo_actual.upper()}\n")
                f.write(f"{'='*70}\n")

            # Cabecera de dataset
            if r['dataset'] != ds_actual:
                ds_actual = r['dataset']
                f.write(f"\n{'─'*50}\n")
                f.write(f"  Dataset: {r['dataset']}\n")
                f.write(f"{'─'*50}\n")

            f.write(f"\n  ■ {r['metrica']} ({r['direccion']}) — Nemenyi + Wilcoxon (r)\n")
            f.write(f"    N = {r['n_pareado']} | Pares = {r['n_pares']}\n")
            f.write(f"\n")

            # Cabecera de tabla
            f.write(f"    {'A':<20s} {'B':<20s} {'Δ':>7s} {'p-adj':>11s} {'Sig':>4s} {'r':>7s} {'Efecto':>9s}\n")
            f.write(f"    {'─'*20} {'─'*20} {'─'*7} {'─'*11} {'─'*4} {'─'*7} {'─'*9}\n")

            n_sig = 0
            for p in r['pares']:
                sig_mark = "✓" if p['sig'] else "✗"
                estrellas = estrellas_p(p['p_adj'])
                delta_str = f"{p['delta']:+.4f}"
                r_str = f"{p['r']:.4f}"
                efecto_str = f"{p['efecto']:>9s}"

                f.write(f"    {p['m1']:<20s} {p['m2']:<20s} {delta_str:>7s} {formatear_p(p['p_adj']):>11s} {sig_mark:>4s} {r_str:>7s} {efecto_str} {estrellas}\n")

                if p['sig']:
                    n_sig += 1

            f.write(f"    Pares significativos: {n_sig}/{r['n_pares']}\n")

        # Resumen final
        total_pares = sum(r['n_pares'] for r in todos_resultados)
        total_sig = sum(sum(1 for p in r['pares'] if p['sig']) for r in todos_resultados)
        total_no_sig = total_pares - total_sig

        # Desglose tamaños de efecto (solo significativos)
        efecto_counts = {'grande': 0, 'mediano': 0, 'pequeño': 0, 'nulo': 0}
        for r in todos_resultados:
            for p in r['pares']:
                if p['sig']:
                    efecto_counts[p['efecto']] += 1

        f.write(f"\n\n{'═'*70}\n")
        f.write("   RESUMEN\n")
        f.write(f"{'═'*70}\n\n")
        f.write(f"  Total comparaciones: {total_pares}\n")
        f.write(f"  Significativas:      {total_sig}\n")
        f.write(f"  No significativas:   {total_no_sig}\n")
        f.write(f"\n  Tamaños de efecto (significativos):\n")
        for ef in ['grande', 'mediano', 'pequeño', 'nulo']:
            f.write(f"    {ef}: {efecto_counts[ef]}\n")

    print(f"✓ Reporte: {ruta}")

    # CSV con todos los pares (formato completo)
    filas_csv = []
    for r in todos_resultados:
        for p in r['pares']:
            filas_csv.append({
                'grupo': r['grupo'],
                'dataset': r['dataset'],
                'metrica': r['metrica'],
                'col': r['col'],
                'test': 'Nemenyi',
                'modelo_a': p['m1'],
                'modelo_b': p['m2'],
                'diff_media': p['delta'],
                'estadistico': '',
                'p_adj': p['p_adj'],
                'significativo': p['sig'],
                'efecto_valor': p['r'],
                'efecto_magnitud': p['efecto'],
                'estrellas': estrellas_p(p['p_adj']),
                'n_pareado': r['n_pareado'],
            })

    ruta_csv = os.path.join(output_dir, "Reporte_PostHoc.csv")
    pd.DataFrame(filas_csv).to_csv(ruta_csv, index=False, encoding="utf-8")
    print(f"✓ CSV:     {ruta_csv}")


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="analisis")
    parser.add_argument("--output-dir", default="CD_diagrams")
    args = parser.parse_args()

    # Carpeta de salida relativa al script, no al cwd
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    input_dir  = os.path.join(script_dir, args.input_dir)  if not os.path.isabs(args.input_dir)  else args.input_dir

    os.makedirs(output_dir, exist_ok=True)
    print(f"Guardando en: {output_dir}")

    print("Cargando datos...")
    df = cargar_todos(input_dir)
    if df.empty:
        print(f"  [ERROR] No se encontraron CSVs en '{input_dir}'. Comprueba la ruta.")
        exit(1)
    print(f"  {len(df):,} filas | {df['Modelo'].nunique()} modelos")
    print(f"  Modelos encontrados: {sorted(df['Modelo'].unique())}\n")

    # Forzar numérico en métricas extra (Time, CO2)
    for c in ['Time_seconds', 'CO2_gramos']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    modelos_g1 = GRUPO_1['modelos']
    todos_posthoc = []

    for ds in GRUPO_1['datasets']:
        print(f"\n=== {ds} ===")
        datos_grid = {}

        # CD diagrams: solo 4 métricas de calidad
        for col_met, nombre_met in METRICAS_CALIDAD.items():
            if col_met not in df.columns:
                continue

            sub = df[(df['Dataset'] == ds) & (df['Modelo'].isin(modelos_g1)) & df[col_met].notna()]
            mod_pres = [m for m in modelos_g1 if m in sub['Modelo'].unique()]
            if len(mod_pres) < 2:
                print(f"  [SKIP] {col_met}: solo {len(mod_pres)} modelos con datos")
                continue

            n = min(len(sub[sub['Modelo'] == m]) for m in mod_pres)
            datos = {m: sub[sub['Modelo'] == m][col_met].values[:n] for m in mod_pres}
            datos_grid[col_met] = datos

            fname = os.path.join(output_dir, f"CD_G1_{ds}_{col_met}.png")
            cd_individual(datos, f"{nombre_met} — {ds}", fname)

        fname_grid = os.path.join(output_dir, f"CD_G1_{ds}_calidad_grid.png")
        cd_grid_4metricas(datos_grid, ds, fname_grid)

        # Reporte G1: las 6 métricas (calidad + tiempo + CO2)
        for col_met, (nombre_met, direccion) in METRICAS_REPORTE.items():
            if col_met not in df.columns:
                continue

            sub = df[(df['Dataset'] == ds) & (df['Modelo'].isin(modelos_g1)) & df[col_met].notna()]
            mod_pres = [m for m in modelos_g1 if m in sub['Modelo'].unique()]
            if len(mod_pres) < 2:
                continue

            n = min(len(sub[sub['Modelo'] == m]) for m in mod_pres)
            datos = {m: sub[sub['Modelo'] == m][col_met].values[:n] for m in mod_pres}

            print(f"  Post-hoc: {nombre_met} ({ds}, N={n})...")
            r = ejecutar_posthoc_completo(datos, GRUPO_1['nombre'], ds, col_met, nombre_met, direccion)
            if r: todos_posthoc.append(r)

    # ── Grupo 2: Teleco ──
    print("\n=== Teleco (Grupo 2) ===")
    modelos_g2 = GRUPO_2['modelos']
    datos_grid_g2 = {}

    # CD diagrams: solo 4 métricas
    for col_met, nombre_met in METRICAS_CALIDAD.items():
        if col_met not in df.columns:
            continue

        sub = df[(df['Dataset'] == 'Teleco') & (df['Modelo'].isin(modelos_g2)) & df[col_met].notna()]
        mod_pres = [m for m in modelos_g2 if m in sub['Modelo'].unique()]
        if len(mod_pres) < 2:
            print(f"  [SKIP] {col_met}: solo {len(mod_pres)} modelos con datos")
            continue

        n = min(len(sub[sub['Modelo'] == m]) for m in mod_pres)
        datos = {m: sub[sub['Modelo'] == m][col_met].values[:n] for m in mod_pres}
        datos_grid_g2[col_met] = datos

        fname = os.path.join(output_dir, f"CD_G2_Teleco_{col_met}.png")
        cd_individual_g2(datos, f"{nombre_met} — Teleco", fname)

    fname_grid = os.path.join(output_dir, "CD_G2_Teleco_calidad_grid.png")
    cd_grid_4metricas_g2(datos_grid_g2, fname_grid)

    # Reporte G2: las 6 métricas
    for col_met, (nombre_met, direccion) in METRICAS_REPORTE.items():
        if col_met not in df.columns:
            continue

        sub = df[(df['Dataset'] == 'Teleco') & (df['Modelo'].isin(modelos_g2)) & df[col_met].notna()]
        mod_pres = [m for m in modelos_g2 if m in sub['Modelo'].unique()]
        if len(mod_pres) < 2:
            continue

        n = min(len(sub[sub['Modelo'] == m]) for m in mod_pres)
        datos = {m: sub[sub['Modelo'] == m][col_met].values[:n] for m in mod_pres}

        print(f"  Post-hoc: {nombre_met} (Teleco, N={n})...")
        r = ejecutar_posthoc_completo(datos, GRUPO_2['nombre'], 'Teleco', col_met, nombre_met, direccion)
        if r: todos_posthoc.append(r)

    # ── Generar reporte ──
    print(f"\n── Generando reporte post-hoc ({len(todos_posthoc)} bloques) ──")
    if todos_posthoc:
        generar_reporte_posthoc(todos_posthoc, output_dir)
    else:
        print("  [ERROR] No se generaron resultados post-hoc.")

    print("\nListo.")