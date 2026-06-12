"""
Fusiona los adaptadores LoRA con sus modelos base y exporta los modelos
completos (merged) listos para inferencia o cuantizacion.

Para cada Mini-SLM (Gemma 3 1B, Llama 3.2 1B, Qwen3 1.7B): carga el modelo base
(local o descargandolo de Hugging Face), aplica el adaptador LoRA, lo fusiona con
merge_and_unload y guarda el resultado en su carpeta -merged.

Requisitos:
    pip install transformers peft torch

Uso:
    python fusionar_lora.py
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELOS = [
    {
        "nombre": "Gemma 3 1B",
        "base": os.path.join(BASE_DIR, "gemma-3-1b-base"),
        "hf_id": "google/gemma-3-1b-it",
        "lora": os.path.join(BASE_DIR, "lora_adapter_gemma"),
        "output": os.path.join(BASE_DIR, "gemma-3-1b-merged"),
    },
    {
        "nombre": "Llama 3.2 1B",
        "base": os.path.join(BASE_DIR, "llama-3.2-1b-base"),
        "hf_id": "meta-llama/Llama-3.2-1B-Instruct",
        "lora": os.path.join(BASE_DIR, "lora_adapter_llama"),
        "output": os.path.join(BASE_DIR, "llama-3.2-1b-merged"),
    },
    {
        "nombre": "Qwen3 1.7B",
        "base": os.path.join(BASE_DIR, "qwen3-1.7b-base"),
        "hf_id": "Qwen/Qwen3-1.7B",
        "lora": os.path.join(BASE_DIR, "lora_adapter_qwen"),
        "output": os.path.join(BASE_DIR, "qwen3-1.7b-merged"),
    },
]


def fusionar(modelo):
    nombre = modelo["nombre"]
    print(f"\n{nombre}")

    if not os.path.exists(modelo["base"]):
        print(f"Base no encontrada en {modelo['base']}")
        print(f"Descargando {modelo['hf_id']} de Hugging Face...")
        base = AutoModelForCausalLM.from_pretrained(
            modelo["hf_id"], torch_dtype=torch.float16
        )
        tok = AutoTokenizer.from_pretrained(modelo["hf_id"])
        os.makedirs(modelo["base"], exist_ok=True)
        base.save_pretrained(modelo["base"])
        tok.save_pretrained(modelo["base"])
        print("Base descargada y guardada.")
    else:
        print(f"Cargando modelo base desde {modelo['base']}...")
        base = AutoModelForCausalLM.from_pretrained(
            modelo["base"], torch_dtype=torch.float16
        )
        tok = AutoTokenizer.from_pretrained(modelo["base"])

    print("Cargando adaptador LoRA...")
    base = PeftModel.from_pretrained(base, modelo["lora"])

    print("Fusionando LoRA con modelo base...")
    merged = base.merge_and_unload()

    print(f"Guardando en {modelo['output']}...")
    os.makedirs(modelo["output"], exist_ok=True)
    merged.save_pretrained(modelo["output"])
    tok.save_pretrained(modelo["output"])

    del merged, base, tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"{nombre} fusionado correctamente.")


if __name__ == "__main__":
    for m in MODELOS:
        fusionar(m)
    print("\nLos 3 modelos se han fusionado correctamente.")