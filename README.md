# Object Replacer (YOLO-World + SAM 2 + SD 1.5 Inpaint)

Replace objects in any photo with generative AI. Tell it what to **find** and what to **replace** it with.

```
python replace.py --image basket.jpg --find apple --replace "ripe orange" --out result.jpg
```

Tested on a GTX 1660 Ti (6 GB VRAM).

---

## One-time setup (Windows + VS Code)

### 1. NVIDIA driver
Update to the latest GeForce driver from GeForce Experience or nvidia.com/drivers, then restart.
Verify with `nvidia-smi` in Command Prompt — you should see your GPU listed.

### 2. Python 3.11
Install from python.org. **Check "Add Python to PATH"** during install.

### 3. Project
```powershell
mkdir C:\Projects\object-replacer
cd C:\Projects\object-replacer
# copy replace.py, requirements.txt, setup.ps1 into this folder
code .
```

### 4. Virtual environment (in VS Code's terminal)
```powershell
python -m venv venv
venv\Scripts\activate
```
VS Code: `Ctrl+Shift+P` -> "Python: Select Interpreter" -> pick `.\venv\Scripts\python.exe`.

If PowerShell blocks the activation script, run this once in an admin PowerShell:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### 5. Install everything
```powershell
.\setup.ps1
```

This downloads ~5 GB of model weights on first run (cached to `~\.cache\huggingface`).

---

## Usage

```powershell
python replace.py --image input.jpg --find "apple" --replace "ripe orange" --out result.jpg
```

| Flag         | Default | Notes                                                          |
|--------------|---------|----------------------------------------------------------------|
| `--image`    | —       | Input photo                                                    |
| `--find`     | —       | What to detect. Any English noun. "red car", "dog", "wine glass" |
| `--replace`  | —       | What to generate in its place. Be descriptive.                 |
| `--out`      | result.jpg | Output path                                                    |
| `--conf`     | 0.05    | Detection threshold. Raise to be stricter, lower to catch more |
| `--dilate`   | 12      | Mask growth in pixels. Bigger = hides edges better but eats surroundings |
| `--feather`  | 21      | Edge softness. Must be odd                                     |
| `--strength` | 0.99    | How much to change inside the mask. 0.99 = full replace, 0.85 = preserve shape |
| `--steps`    | 30      | SD inference steps. 20 fast / 30 balanced / 50 max quality     |
| `--seed`     | 42      | Change for different variations of the same prompt             |

### What gets saved
- `result.jpg` — the final image
- `debug/01_input.jpg` ... `debug/06_final.jpg` — every stage
- `debug/_overview.png` — 2x3 grid of all stages, useful for debugging quality issues

---

## Examples

**Apples to oranges**
```powershell
python replace.py --image fruit.jpg --find apple --replace "fresh orange citrus fruit" --out fruit_oranges.jpg
```

**Replace cars with horses**
```powershell
python replace.py --image street.jpg --find car --replace "brown horse standing" --out street_horses.jpg --dilate 20
```

**Shape-preserving (apple keeps round shape, just becomes a peach)**
```powershell
python replace.py --image fruit.jpg --find apple --replace "ripe peach" --strength 0.85 --out peaches.jpg
```

---

## Tuning quality

In order of impact:

1. **Prompt the replacement well.** Add context: `"ripe orange citrus fruit, photorealistic, sharp focus"` beats `"orange"`.
2. **Dilate more if you see ghosts.** Original-object edges leaking through? Bump `--dilate 12` to `--dilate 20`.
3. **Lower strength for similar-shape swaps.** Apple -> peach with `--strength 0.85` keeps geometry; apple -> banana needs `--strength 0.99`.
4. **More steps for fine textures.** `--steps 50` if skin/fur/grain matters.
5. **Different seeds.** If one result is bad, try `--seed 100`, `--seed 7`, etc. Cheap rerolls.

---

## If you hit "CUDA out of memory"

In `replace.py`, find this line:
```python
# pipe.enable_model_cpu_offload()
```
Uncomment it. The pipeline becomes 2-3x slower but uses ~2 GB less VRAM.

Also try downscaling the input — anything above 1024px gets resized to 768 anyway for SD 1.5.

---

## Going further

- **Faces**: SD 1.5 is mediocre at faces. For identity-preserving face edits, look at `InstantID` or `IP-Adapter FaceID`. Heavier; needs an RTX card.
- **SDXL quality with 6 GB**: not really possible without offloading the UNet, which makes each image take ~1 minute. Stay on SD 1.5 for development.
- **Speed**: install `xformers` for ~30% speedup (`pip install xformers`).
- **Better detection**: if YOLO-World misses, swap to Grounding DINO. Heavier but more accurate on unusual prompts.
