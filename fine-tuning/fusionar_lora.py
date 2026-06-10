from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

base_path = r"C:\Users\artur\Desktop\TFG\fine-tuning\gemma-3-1b-base"
lora_path = r"C:\Users\artur\Desktop\TFG\fine-tuning\resultados_fine-tuning\gemma-3-1b\lora_adapter"
output_path = r"C:\Users\artur\Desktop\TFG\fine-tuning\gemma-3-1b-merged"

print("Cargando modelo base...")
model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(base_path)

print("Cargando adaptador LoRA...")
model = PeftModel.from_pretrained(model, lora_path)

print("Fusionando LoRA con modelo base...")
model = model.merge_and_unload()

print(f"Guardando modelo fusionado en {output_path}...")
os.makedirs(output_path, exist_ok=True)
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print("Modelo fusionado guardado.")