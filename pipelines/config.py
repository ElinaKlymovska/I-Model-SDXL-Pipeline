"""
Central configuration for the headless face enhancement pipeline.

All paths are designed for RunPod with workspace mounted at /workspace.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class Paths:
    workspace: Path = Path("/workspace")
    data_root: Path = workspace / "data"
    input_dir: Path = data_root / "input"
    output_dir: Path = data_root / "outputs"
    logs_dir: Path = workspace / "logs"
    models_dir: Path = workspace / "models" / "Stable-diffusion"
    controlnet_dir: Path = workspace / "models" / "ControlNet"


@dataclass
class ModelSpec:
    name: str
    # Filename of the checkpoint in models_dir
    checkpoint_filename: str
    # Optional recommended positive prompt booster
    style_prompt: str


@dataclass
class PipelineConfig:
    # WebUI API endpoint
    webui_base_url: str = "http://127.0.0.1:7860"

    # Default generation parameters
    width: int = 1024
    height: int = 1024
    steps: int = 30
    cfg_scale: float = 6.5
    sampler: str = "DPM++ 2M Karras"
    denoising_strength: float = 0.62
    seed: int = -1  # random per image
    batch_size: int = 1
    n_iter: int = 1

    # ADetailer
    adetailer_model: str = "face_yolov8n.pt"  # lightweight face detector
    adetailer_denoising_strength: float = 0.4
    adetailer_inpaint_only_masked: bool = True
    adetailer_mask_blur: int = 4

    # Face crop parameters (applied before img2img)
    crop_expand_ratio: float = 1.4  # expand bbox to include hair/contour
    min_face_size_px: int = 128

    # ControlNet
    enable_controlnet: bool = True
    control_mode: str = "canny"  # canny or depth
    control_weight: float = 0.65
    control_guidance_start: float = 0.0
    control_guidance_end: float = 1.0
    control_resize_mode: str = "Crop"
    canny_low_threshold: int = 100
    canny_high_threshold: int = 200

    # Characters are folders under input_dir; set minimum images per character
    min_images_per_character: int = 5

    # Mapping of friendly names to model specs
    models: Dict[str, ModelSpec] = field(default_factory=lambda: {
        # The filenames are expected to exist under Paths.models_dir.
        # Ensure these are fp16 .safetensors.
        "gonzalomo_xl": ModelSpec(
            name="Gonzalomo XL / Fluxpony",
            checkpoint_filename="GonzalomoXL_Fluxpony_fp16.safetensors",
            style_prompt=(
                "hyper-realistic, ultra-detailed, editorial beauty, glossy skin, soft studio lighting, "
                "subtle makeup, cinematic color grading, instagram influencer aesthetic, premium portrait"
            ),
        ),
        "babes_illustrious": ModelSpec(
            name="Babes Illustrious SDXL",
            checkpoint_filename="BabesIllustrious_SDXL_fp16.safetensors",
            style_prompt=(
                "glamour portrait, luminous skin, high-end photo retouch, volumetric light, sharp eyes, "
                "vivid yet tasteful colors, elegant, editorial beauty"
            ),
        ),
    })


PATHS = Paths()
CFG = PipelineConfig()


def ensure_directories() -> None:
    PATHS.output_dir.mkdir(parents=True, exist_ok=True)
    PATHS.logs_dir.mkdir(parents=True, exist_ok=True)
    PATHS.models_dir.mkdir(parents=True, exist_ok=True)
    PATHS.controlnet_dir.mkdir(parents=True, exist_ok=True)


