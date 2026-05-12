# setup.ps1 — One-shot install for the object replacement pipeline
# Run from your project folder with the venv activated:
#   .\setup.ps1
#
# Note: PowerShell mangles nested quotes inside `python -c "..."`, so we
# write tiny temp scripts to disk and execute them. Robust and readable.

$ErrorActionPreference = "Stop"

Write-Host "Step 1: PyTorch with CUDA 12.6 (GTX 1660 Ti compatible)..." -ForegroundColor Cyan
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

Write-Host "`nStep 2: Verifying CUDA..." -ForegroundColor Cyan
@'
import torch
assert torch.cuda.is_available(), "CUDA not detected. Update your NVIDIA driver and reboot."
print("OK:", torch.cuda.get_device_name(0))
print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), "GB")
'@ | Out-File -FilePath "_check_cuda.py" -Encoding utf8
python _check_cuda.py
Remove-Item _check_cuda.py

Write-Host "`nStep 3: Project dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "`nStep 4: SAM 2 from GitHub..." -ForegroundColor Cyan
pip install "git+https://github.com/facebookresearch/sam2.git"

Write-Host "`nStep 5: Smoke test (all imports)..." -ForegroundColor Cyan
@'
from ultralytics import YOLO
from diffusers import StableDiffusionInpaintPipeline
from sam2.sam2_image_predictor import SAM2ImagePredictor
print("All imports OK")
'@ | Out-File -FilePath "_check_imports.py" -Encoding utf8
python _check_imports.py
Remove-Item _check_imports.py

Write-Host "`nDone! Try:" -ForegroundColor Green
Write-Host '  python replace.py --image YOUR_PHOTO.jpg --find apple --replace "ripe orange" --out result.jpg' -ForegroundColor Yellow