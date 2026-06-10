# TFG — Diseño e implementación de una aplicación generativa de explicación de contenidos de telecomunicaciones de dominio específico

Trabajo de Fin de Grado en Ingeniería de Sistemas de Telecomunicación (Universidad CEU San Pablo, Madrid).

Este proyecto compara LLMs, SLMs, Mini-SLMs, modelos fine-tuneados y un GAN generativo en tareas de generación de texto, evaluados sobre cuatro datasets: WebNLG, ToTTo, KELM y un corpus propio en español del ámbito de telecomunicaciones (*Teleco*, ~24.680 pares Q&A). El resultado final es **TelecoSLM**, una app Flutter que ejecuta el modelo seleccionado (Qwen3 1.7B FT, Q4_K_M GGUF) de forma completamente offline en Android y Windows.

## Estructura del repositorio

```
TFG/
├── pipelines/            # Scripts de inferencia para cada familia de modelos
├── graficas_y_tests/     # Visualizaciones y análisis estadístico
├── fine-tuning/          # Fine-tuning LoRA, fusión de adaptadores
└── app_teleco_slm/       # App Flutter (Android + Windows)
```

## pipelines/

Scripts de inferencia que ejecutan cada modelo sobre los cuatro datasets y generan los CSV de resultados con métricas BLEU, ROUGE-L, METEOR y BERTScore.

- `pipelineLLM.py` — Modelos grandes (+70B params) vía API (OpenRouter / Together AI) en los tres datasets genéricos.
- `pipelineSLM.py` — Modelos medianos (~7-14B params) vía API (OpenRouter / Together AI) en los tres datasets genéricos.
- `pipelineSLMbbdd.py` — Evaluación de SLMs sobre el dataset Teleco vía API (OpenRouter / Together AI).
- `pipelineMiniSLM.py` — Modelos pequeños (~1-3B params) vía API (OpenRouter / Together AI) en los tres datasets genéricos.
- `pipelineminiSLMbbdd.py` — Evaluación de Mini-SLMs sobre el dataset Teleco vía API (OpenRouter / Together AI).
- `pipeline_gan.py` — Evaluación de la GAN sobre el dataset Telecomunicaciones.
- `pipeline_finetuning.py` — Evaluación de los modelos fine-tuned sobre el dataset Telecomunicaciones y KELM. Genera plots de las curvas de aprendizaje de cada modelo fine-tuned.
- `gan_teleco_v3.py` — GAN generativa de texto para le dataset de Telecomunicaciones. Genera plots de las curvas de aprendizaje del modelo.

## graficas_y_tests/

Generación de gráficas y tests estadísticos sobre los resultados de los pipelines.

- Boxplots, violin plots, bar plots, diagramas de diferencia crítica (CD), Q-Q plots y gráficas de tiempos de inferencia.
- Tests de Shapiro-Wilk, Friedman, Nemenyi post-hoc, Wilcoxon con tamaño de efecto r e intervalos de confianza bootstrap al 95%.
- Dos grupos experimentales: **Grupo 1** (9 modelos, WebNLG/ToTTo/KELM) y **Grupo 2** (7 modelos incluyendo GAN y fine-tuned, Teleco).

## fine-tuning/

Entrenamiento y preparación de los modelos fine-tuneados.

- Fusión de adaptadores LoRA con el modelo base.

## app_teleco_slm/

App Flutter multiplataforma que permite consultar información de telecomunicaciones offline.

- **Android**: usa `llama_flutter_android` para inferencia nativa del GGUF.
- **Windows**: usa `llama-server.exe` vía HTTP local.
- Filtro de tokens de modo thinking con auto-retry.
- System prompt few-shot optimizado para respuestas en español.

> **Nota:** El archivo `.gguf` del modelo no está incluido por su tamaño. Para generarlo, descárgalo desde [https://huggingface.co/arturofierrop/qwen-3-1.7B-teleco-slm-GGUF].

## Datasets

| Dataset | Idioma | Pares | Descripción |
|---------|--------|-------|-------------|
| WebNLG | EN | ~13.000 | Tripletas RDF → texto |
| ToTTo | EN | ~120.000 | Tablas → descripciones |
| KELM | EN | ~15.000 | Grafos de conocimiento → texto |
| Teleco | ES | ~24.680 | Currículo de telecomunicaciones CEU → Q&A |

## Tecnologías principales

- Python 3.10+, PyTorch, Hugging Face Transformers, PEFT, llama.cpp
- Flutter / Dart
- APIs: OpenRouter, Together AI

## Autor

Arturo — Grado en Ingeniería de Sistemas de Telecomunicación, Universidad CEU San Pablo (Madrid).