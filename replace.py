"""
replace.py — Detect objects in an image and replace them with something else
            using YOLO-World (detection) + SAM 2 (segmentation) + SD 1.5 Inpaint.

USAGE
-----
  python replace.py --image input.jpg --find apple --replace "ripe orange" --out result.jpg

  python replace.py --image street.jpg --find "car" --replace "horse-drawn carriage" --out result.jpg 

Designed to run on a 6GB VRAM GPU (e.g. GTX 1660 Ti) by loading one model at a
time and freeing CUDA memory between stages.

PIPELINE
--------
  load image
    -> [STAGE 1] YOLO-World open-vocabulary detection on `--find`
    -> [STAGE 2] SAM 2 segmentation from each bounding box
    -> [STAGE 3] union + dilate masks
    -> [STAGE 4] Stable Diffusion 1.5 Inpainting using `--replace` as prompt
    -> [STAGE 5] feather edges & blend onto original
    -> save result + per-stage debug images

The first run downloads all model weights (~5 GB) from Hugging Face / Ultralytics.
"""

from __future__ import annotations
import argparse
import gc
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


#memory helpers 

def free_vram(*objs):
    """Delete model objects and clear the CUDA cache."""
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def vram_report(tag: str):
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"   [VRAM] {tag}: used={used:.2f} GB  peak={peak:.2f} GB")


#  stage 1 detection with YOLO-World (open-vocabulary)

def detect(image_path: str, find_classes: list[str], conf_threshold: float = 0.05):
    """
    YOLO-World accepts arbitrary class names at inference time. No fine-tuning
    needed — you just tell it what to look for.

    Returns: list of dicts {"label", "bbox": (x1,y1,x2,y2), "confidence": float}
    """
    print(f"\n[1/5] DETECTION — looking for: {find_classes}")
    from ultralytics import YOLO

    # YOLOv8x-worldv2 is the open-vocab variant.
    model = YOLO("yolov8x-worldv2.pt")
    model.set_classes(find_classes)

    results = model.predict(
        image_path,
        conf=conf_threshold,
        iou=0.5,
        verbose=False,
        device=0 if torch.cuda.is_available() else "cpu",
    )[0]

    detections = []
    for box in results.boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        label_idx = int(box.cls[0])
        detections.append({
            "label": find_classes[label_idx],
            "bbox": tuple(xyxy.tolist()),
            "confidence": float(box.conf[0]),
        })

    print(f"      Found {len(detections)} matching object(s)")
    for d in detections:
        print(f"        - {d['label']}  conf={d['confidence']:.2f}  bbox={d['bbox']}")

    vram_report("after YOLO-World")
    free_vram(model)
    return detections


#  stage 2  segmentation with SAM 2
def segment(image_rgb: np.ndarray, detections: list) -> np.ndarray:
    """
    For each bbox, ask SAM 2 for a tight binary mask. Returns one combined
    binary mask (H x W, uint8 0/255) covering all detections.
    """
    print(f"\n[2/5] SEGMENTATION — running SAM 2 on {len(detections)} box(es)")
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-base-plus")
    predictor.set_image(image_rgb)

    H, W = image_rgb.shape[:2]
    combined = np.zeros((H, W), dtype=np.uint8)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for det in detections:
            box = np.array(det["bbox"])
            masks, scores, _ = predictor.predict(box=box, multimask_output=False)
            mask = (masks[0] > 0).astype(np.uint8) * 255
            combined = np.maximum(combined, mask)
            print(f"        - segmented {det['label']}  iou_score={scores[0]:.2f}  "
                  f"pixels={int(mask.sum() / 255)}")

    vram_report("after SAM 2")
    free_vram(predictor)
    return combined



#  stage 3 mask post-processing

def refine_mask(mask: np.ndarray, dilation: int = 12) -> np.ndarray:
    """
    Clean up the raw SAM mask before inpainting. The raw mask may have holes or be too tight, 
    which can cause inpaint failures or visible artifacts.
    """
    print(f"\n[3/5] MASK REFINEMENT — closing holes + dilating by {dilation}px")
    #step 1: close small holes inside the mask
    close_k = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=2)
    #step 2: dilate to expand the masked area
    dilate_k = np.ones((dilation, dilation), np.uint8)
    return cv2.dilate(mask, dilate_k, iterations=1)


#  stage 4 inpainting with stable diffusion 1.5
#my device has 6GB VRAM, which is too tight for heavier models than SD 1.5. 
def inpaint(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    prompt: str,
    negative_prompt: str = "blurry, distorted, low quality, deformed, cartoon, text, watermark",
    strength: float = 0.99,
    guidance: float = 7.5,
    steps: int = 30,
    seed: int = 42,
) -> np.ndarray:
    
    

    
    print(f"\n[4/5] INPAINTING — SD 1.5 Inpaint")
    print(f"      prompt: \"{prompt}\"")
    print(f"      steps={steps}  guidance={guidance}  strength={strength}  seed={seed}")

    from diffusers import StableDiffusionInpaintPipeline


    # here i faced a problem with the GTX 1660 Ti card — the inpainted output was coming out all black, 
    # so I did my research and consulted LLMs to solve it and the comments below is regarding that.
    # GTX 16xx (1650, 1660, 1660 Ti, 1660 Super) FIX:
    # These Turing-minor (TU116/TU117) cards have a buggy fp16 implementation
    # — well documented in the SD community. The symptoms are black or green
    # images even when no errors are raised. The reliable fix is to load the
    # entire pipeline in fp32. Cost: ~2x VRAM, but your 6GB card can handle it.
    # Other cards (RTX 20+, RTX 30+) use fp16 normally.


    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0).lower()
        gtx_16xx = any(x in gpu_name for x in ["gtx 1650", "gtx 1660", "gtx 16 "])
    else:
        gtx_16xx = False

    if gtx_16xx:
        print(f"      Detected {torch.cuda.get_device_name(0)} — "
              f"loading pipeline in fp32 (TU116 has broken fp16)")
        dtype = torch.float32
    else:
        dtype = torch.float16

    # NOTE: the original "runwayml/stable-diffusion-inpainting" repo was taken
    # down in 2024. The model is now hosted under the stable-diffusion-v1-5 org.
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-inpainting",
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cuda")

    # Belt + suspenders against the diffusers "all-black output" bug:
    # disable every post-processing hook that could zero the image.
    pipe.safety_checker = None
    if hasattr(pipe, "watermark"):
        pipe.watermark = None

    # Memory-saving toggles. Helpful even on fp32 to stay under 6GB.
    # Note: the new API is pipe.vae.enable_slicing() (the pipe-level wrapper
    # is deprecated as of diffusers 0.31+).
    pipe.enable_attention_slicing()
    pipe.vae.enable_slicing()
    # If you OOM, uncomment the next line. Moves UNet to CPU between steps —
    # ~2-3x slower but uses much less VRAM.
    # pipe.enable_model_cpu_offload()

    # SD 1.5 is trained at 512x512. The pipeline accepts arbitrary sizes but
    # quality drops outside ~768. We pad to a multiple of 8 and resize down
    # if the long edge is over 768.
    orig_h, orig_w = image_rgb.shape[:2]
    long_edge = max(orig_h, orig_w)
    if long_edge > 768:
        scale = 768 / long_edge
        new_w = int(round(orig_w * scale / 8) * 8)
        new_h = int(round(orig_h * scale / 8) * 8)
    else:
        new_w = int(round(orig_w / 8) * 8)
        new_h = int(round(orig_h / 8) * 8)

    img_pil  = Image.fromarray(image_rgb).resize((new_w, new_h), Image.LANCZOS)
    mask_pil = Image.fromarray(mask).resize((new_w, new_h), Image.NEAREST)

    generator = torch.Generator("cuda").manual_seed(seed)
    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=img_pil,
        mask_image=mask_pil,
        num_inference_steps=steps,
        guidance_scale=guidance,
        strength=strength,
        generator=generator,
    ).images[0]

    # Resize back to original
    out = out.resize((orig_w, orig_h), Image.LANCZOS)
    out_arr = np.array(out)

    # Guard against the "black image" failure mode. If we silently feathered a
    # black image onto the input, the user would see a dark blob and not know
    # the inpaint failed. Detect this loudly.
    if out_arr.mean() < 5:
        print("\n  !! WARNING: inpaint returned an all-black image.")
        print("     Likely causes:")
        print("       (a) safety/NSFW filter triggered — try a different --seed")
        print("       (b) VRAM OOM during UNet pass — uncomment "
              "pipe.enable_model_cpu_offload() in replace.py")
        print("       (c) corrupted model download — delete the HuggingFace "
              "cache folder and re-run\n")

    vram_report("after SD inpaint")
    free_vram(pipe)
    return out_arr


#  stage 5  Feather + blend
def blend(original_rgb: np.ndarray, inpainted_rgb: np.ndarray,
          mask: np.ndarray, feather: int = 21) -> np.ndarray:
    """
    Even with a dilated mask, the boundary is a giveaway. the mask is feathered
    edge so the inpainted region fades into the original at the border.
    """
    print(f"\n[5/5] BLEND — feathering edges (kernel={feather}px)")
    soft = cv2.GaussianBlur(mask, (feather, feather), 0).astype(np.float32) / 255.0
    soft = soft[..., None]
    out = (inpainted_rgb.astype(np.float32) * soft +
           original_rgb.astype(np.float32) * (1 - soft))
    return out.astype(np.uint8)



# debug visualization
def save_debug_grid(stages: dict, out_dir: Path):
    """Save individual stage images + a 2x3 overview grid."""
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, img in stages.items():
        if img.ndim == 2:
            cv2.imwrite(str(out_dir / f"{name}.png"), img)
        else:
            cv2.imwrite(str(out_dir / f"{name}.jpg"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    titles = ["1. Input", "2. Detections", "3. SAM mask", "4. Dilated mask",
              "5. Raw inpaint", "6. Final (feathered)"]
    keys   = ["01_input", "02_detections", "03_mask_raw", "04_mask_dilated",
              "05_inpaint_raw", "06_final"]
    for ax, k, t in zip(axes.flat, keys, titles):
        if k in stages:
            im = stages[k]
            if im.ndim == 2:
                ax.imshow(im, cmap="gray")
            else:
                ax.imshow(im)
        ax.set_title(t); ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_dir / "_overview.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


# main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",   required=True, help="Input image path")
    ap.add_argument("--find",    required=True, help="What to detect, e.g. 'apple'")
    ap.add_argument("--replace", required=True, help="What to replace with, e.g. 'orange'")
    ap.add_argument("--out",     default="result.jpg", help="Output path")
    ap.add_argument("--conf",    type=float, default=0.05, help="Detection confidence threshold")
    ap.add_argument("--dilate",  type=int,   default=12, help="Mask dilation in pixels")
    ap.add_argument("--feather", type=int,   default=21, help="Edge feather kernel size (odd)")
    ap.add_argument("--strength", type=float, default=0.99, help="Inpaint denoising strength")
    ap.add_argument("--steps",   type=int,   default=30, help="SD inference steps")
    ap.add_argument("--seed",    type=int,   default=42)
    ap.add_argument("--debug-dir", default="debug", help="Where to save per-stage images")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — this will be very slow on CPU.")
    else:
        print(f"Using device: {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB VRAM)")

    image_path = Path(args.image)
    if not image_path.exists():
        sys.exit(f"Input image not found: {image_path}")

    image_bgr = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    print(f"Loaded {image_path.name}  shape={image_rgb.shape}")

    #stage 1: detect
    find_classes = [args.find]
    detections = detect(str(image_path), find_classes, conf_threshold=args.conf)
    if not detections:
        sys.exit(f"No '{args.find}' detected. Try lowering --conf or rewording.")

    #building a viz of the detection boxes
    vis_boxes = image_rgb.copy()
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(vis_boxes, (x1, y1), (x2, y2), (255, 50, 50), 3)
        cv2.putText(vis_boxes, f"{d['label']} {d['confidence']:.2f}",
                    (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 50, 50), 2)

    #stage 2: segment
    mask_raw = segment(image_rgb, detections)

    #stage 3: refine
    mask = refine_mask(mask_raw, dilation=args.dilate)

    #stage 4: inpaint
    inpainted = inpaint(
        image_rgb, mask, prompt=args.replace,
        strength=args.strength, steps=args.steps, seed=args.seed,
    )

    #stage 5: blend
    final = blend(image_rgb, inpainted, mask, feather=args.feather)

    #Saving
    cv2.imwrite(args.out, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
    print(f"\nSaved final result -> {args.out}")

    debug_dir = Path(args.debug_dir)
    save_debug_grid({
        "01_input":         image_rgb,
        "02_detections":    vis_boxes,
        "03_mask_raw":      mask_raw,
        "04_mask_dilated":  mask,
        "05_inpaint_raw":   inpainted,
        "06_final":         final,
    }, debug_dir)
    print(f"Saved debug images   -> {debug_dir}/")


if __name__ == "__main__":
    main()