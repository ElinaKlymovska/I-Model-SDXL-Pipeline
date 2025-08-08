#!/usr/bin/env python3
"""
Headless batch pipeline for hyper-stylized face enhancement using SDXL models
via WebUI HTTP API, with optional ControlNet and ADetailer post-processing.

Usage (on the pod):
  python3 pipelines/process_faces.py --model gonzalomo_xl --per-image 1
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
from typing import List, Optional

from PIL import Image

from hyperrealistic.pipelines.config import CFG, PATHS, ensure_directories
from hyperrealistic.pipelines.utils_face import detect_faces, expand_square_crop, crop_image_to_box
from hyperrealistic.pipelines.prompting import build_positive, build_negative
from hyperrealistic.pipelines.webui_client import WebUI


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def run_for_image(webui: WebUI, model_ckpt: str, style_prompt: str, src_path: Path,
                  out_dir: Path, args) -> List[Path]:
    image = Image.open(src_path).convert("RGB")
    faces = detect_faces(image)
    if not faces:
        return []

    # choose largest face
    faces_sorted = sorted(faces, key=lambda b: (b.x2 - b.x1) * (b.y2 - b.y1), reverse=True)
    main_box = faces_sorted[0]
    expanded = expand_square_crop(main_box, image.width, image.height, CFG.crop_expand_ratio)
    face_img = crop_image_to_box(image, expanded, max(CFG.width, CFG.height))

    pos = build_positive(style_prompt, extra=args.prompt_extra)
    neg = build_negative(extra=args.negative_extra)

    b64 = pil_to_b64(face_img)

    control_units = []
    if CFG.enable_controlnet and not args.disable_controlnet:
        if CFG.control_mode == "canny":
            canny = webui.preprocess_canny(b64, CFG.canny_low_threshold, CFG.canny_high_threshold)
            control_units.append({
                "input_image": canny,
                "module": "canny",
                "model": "",  # use any installed canny ControlNet XL
                "weight": CFG.control_weight,
                "guidance_start": CFG.control_guidance_start,
                "guidance_end": CFG.control_guidance_end,
                "resize_mode": CFG.control_resize_mode,
            })
        else:
            depth = webui.preprocess_depth(b64)
            control_units.append({
                "input_image": depth,
                "module": "depth_midas",
                "model": "",
                "weight": CFG.control_weight,
                "guidance_start": CFG.control_guidance_start,
                "guidance_end": CFG.control_guidance_end,
                "resize_mode": CFG.control_resize_mode,
            })

    # Prepare img2img payload; include ADetailer only if script exists
    payload = {
        "denoising_strength": CFG.denoising_strength,
        "prompt": pos,
        "negative_prompt": neg,
        "sampler_name": CFG.sampler,
        "steps": CFG.steps,
        "cfg_scale": CFG.cfg_scale,
        "width": CFG.width,
        "height": CFG.height,
        "seed": args.seed if args.seed is not None else CFG.seed,
        "batch_size": CFG.batch_size,
        "n_iter": args.per_image,
        "init_images": [b64],
        # ControlNet args if extension present
        "alwayson_scripts": {
            "controlnet": {
                "args": control_units
            }
        } if control_units else {},
    }

    if webui.has_img2img_script("ADetailer"):
        payload.update({
            "script_name": "ADetailer",
            "script_args": [
                [
                    {
                        "ad_model": CFG.adetailer_model,
                        "ad_denoising_strength": CFG.adetailer_denoising_strength,
                        "ad_inpaint_only_masked": CFG.adetailer_inpaint_only_masked,
                        "ad_mask_blur": CFG.adetailer_mask_blur,
                    }
                ]
            ],
        })

    # Ensure model is selected (resolve to WebUI model title if needed)
    webui.set_model(model_ckpt)
    # Force-disable xformers attention if WebUI still tries to use it via opts
    try:
        webui._post("/sdapi/v1/options", {"sd_vae_half": False, "use_old_emphasis_behavior": False, "usexformers": False})
    except Exception:
        pass

    result = webui.img2img(**payload)
    outputs: List[Path] = []
    for idx, b64png in enumerate(result.get("images", [])):
        data = base64.b64decode(b64png.split(",")[-1])
        out_path = out_dir / f"{src_path.stem}_gen{idx+1}.png"
        with open(out_path, "wb") as f:
            f.write(data)
        outputs.append(out_path)
    return outputs


def process_character(webui: WebUI, model_key: str, per_image: int, character_dir: Path, out_root: Path, args) -> int:
    from .config import PipelineConfig
    model = CFG.models[model_key]
    ckpt = model.checkpoint_filename
    style_prompt = model.style_prompt

    images = sorted([p for p in character_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}])
    if len(images) < CFG.min_images_per_character:
        return 0

    out_dir = out_root / character_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    num_saved = 0
    for img_path in images:
        outputs = run_for_image(webui, ckpt, style_prompt, img_path, out_dir, args)
        num_saved += len(outputs)
    return num_saved


def main():
    parser = argparse.ArgumentParser(description="Hyper-stylized face enhancer (headless)")
    parser.add_argument("--model", choices=list(CFG.models.keys()), default="gonzalomo_xl")
    parser.add_argument("--per-image", type=int, default=1, help="number of images to generate per source")
    parser.add_argument("--input", type=str, default=str(PATHS.input_dir))
    parser.add_argument("--output", type=str, default=str(PATHS.output_dir))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prompt-extra", type=str, default="")
    parser.add_argument("--negative-extra", type=str, default="")
    parser.add_argument("--disable-controlnet", action="store_true")

    args = parser.parse_args()

    ensure_directories()
    webui = WebUI(CFG.webui_base_url)

    input_root = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    total = 0
    for character_dir in sorted([p for p in input_root.iterdir() if p.is_dir()]):
        saved = process_character(webui, args.model, args.per_image, character_dir, output_root, args)
        total += saved

    print(json.dumps({"saved": total, "output_root": str(output_root)}))


if __name__ == "__main__":
    main()


