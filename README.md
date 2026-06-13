# TFG — Diseño e implementación de una aplicación generativa de explicación de contenidos de telecomunicaciones de dominio específico.

[![Disponible en Google Play](https://img.shields.io/badge/Google%20Play-Teleco%20SLM-success?logo=google-play&logoColor=white)](https://play.google.com/store/apps/details?id=com.telecoslm.app)

Trabajo de Fin de Grado en Ingeniería de Sistemas de Telecomunicación (Universidad CEU San Pablo, Madrid).

Este proyecto compara LLMs, SLMs, Mini-SLMs, modelos fine-tuneados y un GAN generativo en tareas de generación de texto, evaluados sobre cuatro datasets: WebNLG, ToTTo, KELM y un corpus propio en español del ámbito de telecomunicaciones (*Teleco*, ~24.680 pares Q&A). El resultado final es **TelecoSLM**, una app Flutter que ejecuta el modelo seleccionado (Qwen3 1.7B FT, Q4_K_M GGUF) de forma completamente offline en Android y Windows.

## Estructura del repositorio

```
TFG/
├── LLM/                              # Pipeline de inferencia para LLMs (+70B params)
├── SLM/                              # Pipeline de inferencia para SLMs (~7-14B params)
├── mini-SLM/                        # Pipeline de inferencia para Mini-SLMs (~1-3B params)
├── GAN/                             # Pipeline de la GAN generativa
├── fine-tuning/                    # Fine-tuning LoRA y fusión de adaptadores
├── graficas_y_tests/               # Visualizaciones y análisis estadístico
├── app_teleco_slm/                 # App Flutter (Android + Windows)
├── base_teleco/                    # Dataset de Telecomunicaciones (~24.680 pares Q&A)
├── base_kelm/                      # Dataset KELM filtrado por temática STEM
├── splits/                         # Splits train/val/test por subtema
├── Manual de Usuario - TelecoSLM.pdf  # Manual de usuario de la app
├── requirements.txt                # Dependencias Python
├── .gitignore
└── README.md
```

## Requisitos e instalación

- Python 3.11
- (Opcional, para fine-tuning y GAN) GPU NVIDIA con CUDA
- Flutter / Dart (solo para compilar la app)

Instalación de dependencias de Python:

```bash
pip install -r requirements.txt
```

## Variables de entorno

Los pipelines de inferencia por API leen las claves de las siguientes variables de entorno (no van escritas en el código):

| Variable | Necesaria para |
|----------|----------------|
| `OPENROUTER_API_KEY` | Inferencia vía OpenRouter (LLM, SLM, Mini-SLM) |
| `TOGETHER_API_KEY`   | Inferencia vía Together AI (Mini-SLM y modelos fine-tuneados) |
| `HF_TOKEN`           | Descarga de modelos privados de Hugging Face (fine-tuning) |

En Windows (PowerShell), de forma persistente:

```powershell
setx OPENROUTER_API_KEY "tu_clave"
setx TOGETHER_API_KEY "tu_clave"
setx HF_TOKEN "tu_token"
```

Tras usar `setx` hay que abrir una terminal nueva. Para definirlas solo en la sesión actual: `$env:OPENROUTER_API_KEY="tu_clave"` (PowerShell) o `set OPENROUTER_API_KEY=tu_clave` (CMD).

En Linux / macOS:

```bash
export OPENROUTER_API_KEY="tu_clave"
export TOGETHER_API_KEY="tu_clave"
export HF_TOKEN="tu_token"
```

## Uso

El flujo general es: generar los splits, lanzar la inferencia de cada familia de modelos, calcular las métricas y, por último, generar gráficas y tests.

```bash
# 1. Splits por subtema (evita data leakage)
python base_teleco/hacer_splits.py

# 2. Inferencia (ejemplos; cada pipeline acepta --help)
python LLM/pipelineLLM.py --dataset totto --model deepseek
python SLM/pipelineSLM.py --dataset webnlg --model llama
python mini-SLM/pipelineminiSLM.py --dataset kelm --model qwen --evaluar
python SLM/pipelineSLMbbdd.py --model llama          # SLMs sobre dataset de Telecomunicaciones
python mini-SLM/pipelineminiSLMbbdd.py --model qwen --evaluar  # Mini-SLMs sobre dataset de Telecomunicaciones

# 3. Entrenamiento propio
python GAN/pipeline_gan.py --epochs 100
python fine-tuning/pipeline_finetuning.py --dataset all

# 4. Métricas y análisis
python graficas_y_tests/calcula_metricas.py --input-dir resultados
python graficas_y_tests/posthoc.py --input-dir resultados
python graficas_y_tests/boxplots.py --input-dir resultados
```

Cada script admite `--help` para ver todas sus opciones.

## LLM/

Pipeline de inferencia para modelos grandes (DeepSeek-V3, Llama 3.3 70B, Qwen 2.5 72B) vía API (OpenRouter / Together AI) sobre los tres datasets genéricos (WebNLG, ToTTo, KELM).

## SLM/

Pipeline de inferencia para modelos medianos (Gemma 2 9B, Llama 3.2 3B, Qwen 2.5 7B) vía API (OpenRouter / Together AI) sobre los tres datasets genéricos y el dataset de Telecomunicaciones.

## mini-SLM/

Pipeline de inferencia para modelos pequeños (Gemma 3 1B, Llama 3.2 1B, Qwen 3 1.7B) vía API (OpenRouter / Together AI) sobre los tres datasets genéricos y el dataset de Telecomunicaciones. Incluye versiones con limpieza de tags `<think>` para Qwen3 1.7B.

## GAN/

Pipeline completo de la GAN generativa de texto (Conditional SeqGAN con Gumbel-Softmax, ~430M params). Incluye el script de entrenamiento, inferencia sobre dataset de Telecomunicaciones y KELM, cálculo de métricas y generación de curvas de aprendizaje.

## fine-tuning/

Fine-tuning LoRA de Gemma 3 1B, Llama 3.2 1B y Qwen3 1.7B sobre el dataset Teleco (LR=2e-4, rank=8, alpha=16, dropout=0.05, 3 epochs). Incluye fusión de adaptadores LoRA con los modelos base, inferencia sobre dataset de Telecomunicaciones y KELM, y cálculo de métricas.

## graficas_y_tests/

Generación de gráficas y tests estadísticos sobre los resultados de los pipelines.

- Boxplots, violin plots, bar plots, diagramas de diferencia crítica (CD), Q-Q plots y gráficas de tiempos de inferencia.
- Tests de Friedman, Nemenyi post-hoc, Wilcoxon con tamaño de efecto r e intervalos de confianza bootstrap al 95%.
- Dos grupos experimentales: **Grupo 1** (LLM vs SLM vs Mini-SLM, WebNLG/ToTTo/KELM) y **Grupo 2** (Mini-SLM base vs Mini-SLM fine-tuned vs GAN, Teleco).

## app_teleco_slm/

App Flutter multiplataforma que permite consultar información de telecomunicaciones offline.

- **Android**: usa `llama_flutter_android` para inferencia nativa del GGUF.
- **Windows**: usa `llama-server.exe` vía HTTP local.
- Filtro de tokens de modo thinking con auto-retry.
- System prompt few-shot optimizado para respuestas en español.

La app está publicada en Google Play: **[Teleco SLM](https://play.google.com/store/apps/details?id=com.telecoslm.app)**.

La guía de instalación y uso de la app está en [Manual de Usuario - TelecoSLM.pdf](Manual%20de%20Usuario%20-%20TelecoSLM.pdf).

> **Nota:** El archivo `.gguf` del modelo no está incluido por su tamaño. Para obtenerlo, descárgalo desde [Hugging Face](https://huggingface.co/arturofierrop/qwen-3-1.7B-teleco-slm-GGUF).

## base_teleco/

Dataset propio en español construido a partir del plan de estudios de telecomunicaciones de la Universidad CEU San Pablo (~24.680 pares Q&A). Incluye el script `hacer_splits.py` para generar los splits train/val/test por subtema (estratificado por asignatura), de modo que cada subtema queda en un único split y se evita el data leakage.

## base_kelm/

Dataset KELM filtrado por temática STEM (~60.000 filas). Incluye el script de filtrado `kelm.py` que selecciona entradas relacionadas con informática, telecomunicaciones, electrónica, ingeniería y ciencias aplicadas.

## Datasets

| Dataset | Idioma | Pares | Descripción |
|---------|--------|-------|-------------|
| WebNLG | EN | ~13.000 | Tripletas RDF → texto |
| ToTTo | EN | ~120.000 | Tablas → descripciones |
| KELM | EN | ~60.000 | Grafos de conocimiento → texto (filtrado STEM) |
| Teleco | ES | ~24.680 | Plan de estudios del Grado en Ingeniería de Sistemas de Telecomunicación CEU → Q&A |

## Tecnologías principales

- Python 3.11, PyTorch, Hugging Face Transformers, PEFT, llama.cpp
- Flutter / Dart
- APIs: OpenRouter, Together AI

## Autor y tutores

**Autor:** Arturo Fierro Pardo
Grado en Ingeniería de Sistemas de Telecomunicación, Universidad CEU San Pablo (Madrid).

**Tutores:** Ana Sanmartín Domenech y Pablo Pérez Tirador.