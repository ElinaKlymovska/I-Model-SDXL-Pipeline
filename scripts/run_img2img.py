"""
run_img2img.py
Runs SDXL img2img inference without ADetailer using Stable Diffusion WebUI API.
"""

import requests
import os
import base64

WEBUI_API_URL = os.getenv("WEBUI_API_URL", "http://127.0.0.1:7860")

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def save_image_from_base64(b64_data, output_path):
    import base64
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(b64_data.split(",", 1)[-1]))

def run_img2img(image_path, output_path, prompt, negative_prompt, model="epiCRealismXL", denoise_strength=0.4):
    print(f"🎨 Running SDXL img2img on: {image_path}")
    payload = {
        "init_images": [encode_image(image_path)],
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "denoising_strength": denoise_strength,
        "sampler_name": "DPM++ 2M Karras",
        "cfg_scale": 7,
        "steps": 30,
        "width": 768,
        "height": 1024,
        "override_settings": {
            "sd_model_checkpoint": model
        }
    }

    try:
        response = requests.post(f"{WEBUI_API_URL}/sdapi/v1/img2img", json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        if "images" in result:
            save_image_from_base64(result["images"][0], output_path)
            print(f"✅ Img2Img result saved: {output_path}")
        else:
            print("❌ Img2Img failed:", result)
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to WebUI API. Make sure Stable Diffusion WebUI is running on localhost:7860")
        print("💡 Run: python utils/runpod_launcher.py to start WebUI")
    except requests.exceptions.Timeout:
        print("❌ Request timed out. Generation might take longer for large images.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path to save output image")
    parser.add_argument("--prompt", required=True, help="Prompt for generation")
    parser.add_argument("--negative", default="", help="Negative prompt")
    parser.add_argument("--model", default="epiCRealismXL", help="SDXL checkpoint to use")
    parser.add_argument("--denoise", type=float, default=0.4, help="Denoising strength")
    args = parser.parse_args()

    run_img2img(args.input, args.output, args.prompt, args.negative, args.model, args.denoise)
