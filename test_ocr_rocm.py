from transformers import AutoModel, AutoTokenizer
import torch
import os

# Memory management for ROCm
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HIP_VISIBLE_DEVICES"] = '0'

model_name = 'deepseek-ai/DeepSeek-OCR-2'

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print("Loading model with eager attention (ROCm compatible)...")
# Use eager attention and device_map for better memory management
model = AutoModel.from_pretrained(
    model_name, 
    _attn_implementation='eager',
    trust_remote_code=True, 
    use_safetensors=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="auto"  # Automatically distribute model across available GPUs
)

print(f"Model loaded on: {torch.cuda.get_device_name(0)}")
print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

# Test prompt for document to markdown
prompt = "<image>\n<|grounding|>Convert the document to markdown. "

# Create a test image or use existing one
import sys
if len(sys.argv) > 1:
    image_file = sys.argv[1]
else:
    # Create a simple test with PIL
    from PIL import Image
    print("No image provided, creating test image...")
    test_img = Image.new('RGB', (768, 768), color='white')
    image_file = '/tmp/test_image.png'
    test_img.save(image_file)
    print(f"Created test image: {image_file}")

output_path = '/tmp/ocr_output'
os.makedirs(output_path, exist_ok=True)

print(f"Processing image: {image_file}")
print(f"Output path: {output_path}")
print("Starting inference...")

res = model.infer(
    tokenizer, 
    prompt=prompt, 
    image_file=image_file, 
    output_path=output_path, 
    base_size=1024, 
    image_size=768, 
    crop_mode=True, 
    save_results=True
)

print("Inference completed!")
print(f"Results saved to: {output_path}")
