# DeepSeek-OCR-2 on AMD ROCm - Setup Complete ✓

## Hardware Configuration
- **GPUs:** 2× AMD Radeon RX 9070 XT (gfx1201) + iGPU (excluded)
- **ROCm:** 7.2.1.70201
- **PyTorch:** 2.9.1+rocm6.3
- **Python:** 3.12.9
- **Transformers:** 4.46.3

## Quick Start

### 1. Activate Virtual Environment
```bash
cd /home/jules/Documents/DeepSeek-OCR-2
source .venv/bin/activate
```

### 2. Convert Image to Markdown
```bash
# Single image conversion
python image_to_markdown.py /path/to/your/image.png ./output

# Example with test image
python image_to_markdown.py ~/Documents/DeepSeek-OCR/ocr_benchmark_images/Carlito_scale060_ar0p25_align-center_327x992.png ./output
```

### 3. Check Output
```bash
cat ./output/result.mmd
```

## Verified Working Example

**Test Image:** Attention is All You Need paper excerpt  
**Output Quality:** ✓ Excellent text extraction with formatting  
**Processing Time:** ~30-60 seconds per page  
**GPU Memory:** 6.41 GB on RX 9070 XT

## Technical Details

### Key Configuration
- **HIP_VISIBLE_DEVICES=0**: Excludes iGPU (gfx1036), uses only dedicated GPU (gfx1201)
- **device_map={"": "cuda:0"}**: Forces model to first dedicated GPU
- **_attn_implementation='eager'**: ROCm-compatible attention (no flash-attn needed)
- **torch.bfloat16**: Optimized precision for AMD GPUs

### Why This Works
The critical fix was excluding the integrated GPU (gfx1036) which lacks proper rocBLAS libraries:
```python
os.environ["HIP_VISIBLE_DEVICES"] = "0"  # Only RX 9070 XT
device_map={"": "cuda:0"}  # Explicit GPU assignment
```

## Performance Notes

### Current Configuration (Transformers)
- **Single image:** 30-60 seconds
- **GPU utilization:** ~6.4 GB VRAM
- **Quality:** High (preserves formatting, citations, equations)

### Optional: vLLM for Batch Processing
If you need to process 100s of images faster:
- **Potential speedup:** 10-50x for batch processing
- **Setup time:** 6-10 hours (building vLLM 0.18.0 from source)
- **Recommendation:** Only if transformers is too slow for your needs

## Troubleshooting

### rocBLAS Error (gfx1036)
**Error:** `rocBLAS error: Cannot read ... for GPU arch : gfx1036`  
**Solution:** Already fixed in `image_to_markdown.py` with `HIP_VISIBLE_DEVICES=0`

### Out of Memory
```bash
# Model uses ~6.4GB VRAM
# Close other GPU applications
```

### Model Loading Issues
```bash
# Clear cache if needed
rm -rf ~/.cache/huggingface/hub/deepseek-ai--DeepSeek-OCR-2
```

## Environment Verification
```bash
source .venv/bin/activate
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'ROCm: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'GPU Count: {torch.cuda.device_count()}')
"
```

Expected output:
```
PyTorch: 2.9.1+rocm6.3
ROCm: True
GPU: AMD Radeon RX 9070 XT
GPU Count: 3
```

## Files Created
- `image_to_markdown.py` - Production converter script
- `test_ocr_rocm.py` - Test/debug script
- `ROCm_SETUP_COMPLETE.md` - This documentation

---

**Setup completed:** 2026-03-29  
**Status:** ✓ WORKING - Image to Markdown conversion functional  
**Total setup time:** ~2 hours (including ROCm upgrade)
