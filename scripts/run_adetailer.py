"""
run_adetailer.py
Applies ADetailer facial enhancement using Stable Diffusion WebUI API.
"""

import requests
import os
import base64
from PIL import Image
from io import BytesIO

WEBUI_API_URL = os.getenv("WEBUI_API_URL", "http://127.0.0.1:7860")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def save_image_from_base64(b64_data, output_path):
    image_data = base64.b64decode(b64_data.split(",", 1)[-1])
    with open(output_path, "wb") as f:
        f.write(image_data)

def run_adetailer(image_path, output_path, model="epiCRealismXL"):
    print(f"🔍 Running ADetailer on: {image_path}")
    payload = {
        "init_images": [encode_image(image_path)],
        "adetailer_confidence": 0.3,
        "adetailer_model": "face_yolov8n.pt",
        "adetailer_prompt": "",
        "adetailer_denoising_strength": 0.4,
        "denoising_strength": 0.4,
        "alwayson_scripts": {
            "ADetailer": {
                "args": [
                    {
                        "ad_model": "face_yolov8n.pt",
                        "ad_prompt": "",
                        "ad_confidence": 0.3,
                        "ad_denoising_strength": 0.4,
                        "ad_mask_merge_invert": "None"
                    }
                ]
            }
        },
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
            print(f"✅ ADetailer output saved: {output_path}")
        else:
            print("❌ ADetailer failed:", result)
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
    parser.add_argument("--model", default="epiCRealismXL", help="Model name from WebUI")
    args = parser.parse_args()

    run_adetailer(args.input, args.output, args.model)
