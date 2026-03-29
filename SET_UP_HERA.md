# DeepSeek-OCR-2 Setup Guide for Hera

**Machine:** hera (Ubuntu 24.04.4 LTS)  
**Hardware:** 2× AMD Radeon RX 9070 XT (gfx1201) + AMD Ryzen 9 9950X3D  
**Date Created:** 2026-03-29  
**Status:** ✅ Verified Working

---

## Quick Start (If ROCm 7.2.1 Already Installed)

```bash
# 1. Clone repository
git clone https://github.com/deepseek-ai/DeepSeek-OCR-2.git
cd DeepSeek-OCR-2

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install PyTorch for ROCm (DO NOT use CUDA version!)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3

# 4. Install requirements
pip install -r requirements.txt

# 5. Test with provided script
python image_to_markdown.py /path/to/image.png ./output
```

---

## Complete Setup from Scratch

### Prerequisites Check

Verify your hardware and OS:
```bash
# Should show AMD Radeon RX 9070 XT
lspci -nn | grep -i vga

# Should show Ubuntu 24.04
cat /etc/os-release

# Should show kernel 6.17+
uname -a
```

### Step 1: Install/Upgrade ROCm to 7.2.1

**CRITICAL:** Your RX 9070 XT (gfx1201) requires ROCm 7.2.1. Do not use 7.1.x!

```bash
# Download ROCm 7.2.1 installer
cd /tmp
wget https://repo.radeon.com/amdgpu-install/7.2.1/ubuntu/noble/amdgpu-install_7.2.1.70201-1_all.deb

# Install
sudo apt install ./amdgpu-install_7.2.1.70201-1_all.deb -y
sudo apt update
sudo apt install python3-setuptools python3-wheel -y

# Add user to GPU groups
sudo usermod -a -G render,video $LOGNAME

# Install ROCm
sudo apt install rocm -y

# REBOOT REQUIRED
sudo reboot
```

After reboot, verify ROCm installation:
```bash
# Should show ROCm 7.2.1
hipcc --version

# Should show your GPUs
rocm-smi --showproductname
```

Expected output:
```
HIP version: 7.2.xxxxx
GPU[0]: AMD Radeon RX 9070 XT (gfx1201)
GPU[1]: AMD Radeon RX 9070 XT (gfx1201)
```

### Step 2: Clone Repository and Setup Python Environment

```bash
cd /home/jules/Documents
git clone https://github.com/deepseek-ai/DeepSeek-OCR-2.git
cd DeepSeek-OCR-2

# Create virtual environment with Python 3.12
python -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip wheel
```

### Step 3: Install PyTorch for ROCm

**⚠️ CRITICAL: DO NOT INSTALL CUDA VERSION!**

```bash
# CORRECT: ROCm version for AMD GPUs
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'ROCm: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
PyTorch: 2.9.1+rocm6.3
ROCm: True
GPU: AMD Radeon RX 9070 XT
```

**If you see `CUDA` or `False` for ROCm, you installed the wrong version!**
```bash
# Fix: uninstall and reinstall
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
```

### Step 4: Install Project Requirements

```bash
pip install -r requirements.txt
```

This installs:
- transformers==4.46.3
- tokenizers==0.20.3
- PyMuPDF, img2pdf, einops, easydict, addict, Pillow, numpy

### Step 5: Install Additional Dependencies

```bash
# Required for device_map support
pip install accelerate
```

### Step 6: Test Installation

```bash
# Run test script with a sample image
python image_to_markdown.py ~/Documents/DeepSeek-OCR/ocr_benchmark_images/Carlito_scale060_ar0p25_align-center_327x992.png ./test_output

# Check output
cat ./test_output/result.mmd
```

If you see extracted markdown text, the setup is successful! ✅

---

## Critical Configuration Notes

### 1. GPU Selection (Prevents rocBLAS Crashes)

The integrated GPU (gfx1036) lacks proper rocBLAS support. **Always exclude it:**

```python
# In all Python scripts, add at the top:
import os
os.environ["HIP_VISIBLE_DEVICES"] = "0"  # Use only dedicated GPU
```

### 2. Device Map Configuration

When loading the model, explicitly specify the GPU:

```python
model = AutoModel.from_pretrained(
    model_name,
    device_map={"": "cuda:0"},  # NOT "auto" (which includes iGPU)
    ...
)
```

### 3. Attention Implementation

Use `eager` attention (ROCm compatible), NOT `flash_attention_2`:

```python
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation='eager',  # ✓ ROCm compatible
    # _attn_implementation='flash_attention_2',  # ✗ CUDA only
    ...
)
```

---

## Usage Examples

### Single Image Conversion
```bash
source .venv/bin/activate
python image_to_markdown.py document.png ./output
cat ./output/result.mmd
```

### Batch Processing (Shell Script)
```bash
#!/bin/bash
source .venv/bin/activate
for img in /path/to/images/*.png; do
    output_dir="./output_$(basename $img .png)"
    python image_to_markdown.py "$img" "$output_dir"
done
```

### Custom Prompt
```python
# For different OCR tasks
prompt = "<image>\nFree OCR."  # Simple OCR without layout
# or
prompt = "<image>\n<|grounding|>Convert the document to markdown."  # With layout
```

---

## Troubleshooting

### Error: rocBLAS error for GPU arch gfx1036
**Cause:** Using integrated GPU instead of dedicated RX 9070 XT  
**Solution:**
```python
os.environ["HIP_VISIBLE_DEVICES"] = "0"
device_map={"": "cuda:0"}
```

### Error: CUDA out of memory
**Cause:** Model requires ~6.4GB VRAM  
**Solution:**
- Close other GPU applications
- Ensure only one GPU is being used (not split across GPUs)

### Error: ModuleNotFoundError: transformers
**Solution:**
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Error: torch.cuda.is_available() returns False
**Cause:** Installed CUDA version instead of ROCm  
**Solution:**
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
```

### Model Download Issues
```bash
# Clear Hugging Face cache
rm -rf ~/.cache/huggingface/hub/deepseek-ai--DeepSeek-OCR-2

# Re-run script (will re-download)
```

---

## Performance Benchmarks

**Tested on hera (2026-03-29):**

| Image Type | Size | Processing Time | Output Quality |
|------------|------|----------------|----------------|
| Document (text) | 327×992 | ~30-45s | Excellent |
| Paper excerpt | 768×1024 | ~45-60s | Excellent |
| Complex layout | 1024×1024 | ~60-90s | Good |

**GPU Memory Usage:** ~6.41 GB on RX 9070 XT

---

## Version Information (Known Working)

```
ROCm: 7.2.1.70201
PyTorch: 2.9.1+rocm6.3
Transformers: 4.46.3
Python: 3.12.9
Ubuntu: 24.04.4 LTS
Kernel: 6.17.0-19-generic
```

**DO NOT CHANGE THESE VERSIONS** unless you know what you're doing. Different versions may break compatibility.

---

## Optional: vLLM for Maximum Performance

**Note:** Only do this if transformers inference is too slow for your needs.

The provided `vllm-0.8.5+cu118-*.whl` is **CUDA-only** and won't work. To use vLLM with ROCm:

```bash
# Build vLLM from source (6-10 hours)
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.18.0

# Set ROCm environment
export ROCM_PATH=/opt/rocm
export HSA_OVERRIDE_GFX_VERSION=12.0.1

# Install
pip install -e .
```

**Warning:** Requires modifying DeepSeek's vLLM code for API compatibility. Not recommended unless you need batch processing speed.

---

## Quick Reference

### Activate Environment
```bash
cd /home/jules/Documents/DeepSeek-OCR-2
source .venv/bin/activate
```

### Verify Installation
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Run Conversion
```bash
python image_to_markdown.py <image_path> [output_dir]
```

### Check ROCm
```bash
rocm-smi --showproductname
hipcc --version
```

---

## Files Reference

- `image_to_markdown.py` - Main conversion script (ready to use)
- `test_ocr_rocm.py` - Test/debug script
- `ROCm_SETUP_COMPLETE.md` - Technical setup documentation
- `SET_UP_HERA.md` - This file (setup guide for hera)

---

**Setup Time:** ~2 hours (including ROCm installation)  
**Last Verified:** 2026-03-29 on hera  
**Maintainer:** Jules
