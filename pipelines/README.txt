Minimal notes:
- Place fp16 .safetensors models into /workspace/models/Stable-diffusion
  * GonzalomoXL_Fluxpony_fp16.safetensors
  * BabesIllustrious_SDXL_fp16.safetensors
- Start WebUI API: python3 pipelines/start_webui.py
- Run: python3 pipelines/process_faces.py --model gonzalomo_xl --per-image 1
- Inputs: /workspace/data/input/<Character>/*.png|jpg|jpeg|webp
- Outputs: /workspace/data/outputs/<Character>/

