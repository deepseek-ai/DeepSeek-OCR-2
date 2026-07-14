import os
import sys
from pathlib import Path

# ============== MONKEY PATCH ==============
import transformers.models.llama.modeling_llama as modeling_llama

if not hasattr(modeling_llama, "LlamaFlashAttention2"):
    print(">>> Patching LlamaFlashAttention2 (missing in transformers 4.51+)...")
    class LlamaFlashAttention2(modeling_llama.LlamaAttention):
        pass
    modeling_llama.LlamaFlashAttention2 = LlamaFlashAttention2
    print(">>> Patch appliqué avec succès")
# ==========================================

import torch
from transformers import AutoModel, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print(f"torch: {torch.__version__}")
print(f"Device capability: {torch.cuda.get_device_capability()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# ========== CHARGEMENT 100% LOCAL ==========
model_path = "./DeepSeek-OCR-2-weights"
print(f"\n>>> Loading model STRICTEMENT depuis le dossier local : {model_path}")
print("    (local_files_only=True → aucun re-téléchargement)")

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True
)

model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    _attn_implementation="eager",
    device_map="cuda",
    local_files_only=True          # ← CRITIQUE
).eval()

print("\n" + "="*70)
print("=== STRUCTURE DU MODÈLE ===")
print(model)

print("\n=== ATTRIBUTS PRINCIPAUX ===")
print("type(model.model)                 :", type(model.model))
print("hasattr(model.model, 'sam_model')  :", hasattr(model.model, 'sam_model'))
print("hasattr(model.model, 'qwen2_model'):", hasattr(model.model, 'qwen2_model'))
print("hasattr(model.model, 'projector')  :", hasattr(model.model, 'projector'))

print("\n=== DeepEncoder V2 (qwen2_model) ===")
print(model.model.qwen2_model)

print("\n=== Projector (896 → 1280) ===")
print(model.model.projector)

print("\n=== Queries (important pour N tokens) ===")
print("query_1024:", model.model.qwen2_model.query_1024.weight.shape)
print("query_768 :", model.model.qwen2_model.query_768.weight.shape)

print("\n=== Config vision ===")
print(model.config.vision_config)

print("\n=== Projector config ===")
print(model.config.projector_config)

print("\n✅ Modèle chargé avec succès en 100% local !")
print("Copie-colle TOUTE la sortie ici.")
