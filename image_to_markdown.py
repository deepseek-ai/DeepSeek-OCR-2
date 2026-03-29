#!/usr/bin/env python3
"""
DeepSeek-OCR-2 Image to Markdown Converter
Optimized for AMD ROCm (RX 9070 XT)

Usage:
    python image_to_markdown.py <image_path> [output_dir]

Example:
    python image_to_markdown.py document.png ./output
"""

from transformers import AutoModel, AutoTokenizer
import torch
import os
import sys

# Use ONLY dedicated GPUs (RX 9070 XT = gfx1201), exclude iGPU (gfx1036)
os.environ["HIP_VISIBLE_DEVICES"] = "0"
# ROCm memory optimization
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

MODEL_NAME = 'deepseek-ai/DeepSeek-OCR-2'
DEFAULT_OUTPUT_DIR = '/tmp/ocr_output'

def load_model():
    """Load model and tokenizer with ROCm optimizations"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        trust_remote_code=True
    )
    
    print("Loading model (this may take a few minutes)...")
    model = AutoModel.from_pretrained(
        MODEL_NAME, 
        _attn_implementation='eager',  # ROCm compatible
        trust_remote_code=True, 
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"}  # Force use of first dedicated GPU only
    )
    
    model.eval()
    print(f"✓ Model loaded on: {torch.cuda.get_device_name(0)}")
    print(f"  GPU memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    return tokenizer, model

def convert_image_to_markdown(tokenizer, model, image_path, output_dir, prompt=None):
    """Convert single image to markdown"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if prompt is None:
        prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    
    print(f"Processing: {image_path}")
    
    results = model.infer(
        tokenizer, 
        prompt=prompt, 
        image_file=image_path, 
        output_path=output_dir, 
        base_size=1024, 
        image_size=768, 
        crop_mode=True, 
        save_results=True
    )
    
    # Find and return the markdown file
    md_file = os.path.join(output_dir, 'result.mmd')
    if os.path.exists(md_file):
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    return None

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Error: No image path provided")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_DIR
    
    # Load model (cached on first run)
    tokenizer, model = load_model()
    
    # Convert
    try:
        markdown = convert_image_to_markdown(tokenizer, model, image_path, output_dir)
        print(f"\n✓ Conversion complete!")
        print(f"Output directory: {output_dir}")
        if markdown:
            print(f"\nExtracted text preview:")
            print("-" * 50)
            print(markdown[:500] if len(markdown) > 500 else markdown)
            print("-" * 50)
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
