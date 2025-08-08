#!/usr/bin/env python3
"""
Start AUTOMATIC1111 Stable Diffusion WebUI in API-only mode suitable for RunPod.
Assumes repo exists at /workspace/stable-diffusion-webui and that required
extensions are installed: ControlNet and ADetailer.
"""
import os
import subprocess
import time
from pathlib import Path


def main():
    webui_dir = Path("/workspace/stable-diffusion-webui")
    assert webui_dir.exists(), "WebUI directory not found. Install dependencies first."

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Force our args to avoid accidental inherited flags like --xformers
    env["COMMANDLINE_ARGS"] = "--api --listen --port 7860 --enable-insecure-extension-access --no-half-vae --opt-sdp-attention"
    # Ensure xformers path is disabled even if installed
    env["XFORMERS_DISABLED"] = "1"

    # Ensure extensions
    (webui_dir / "extensions").mkdir(exist_ok=True)
    # ControlNet
    cn = webui_dir / "extensions" / "sd-webui-controlnet"
    if not cn.exists():
        subprocess.run(["git", "clone", "https://github.com/Mikubill/sd-webui-controlnet", str(cn)], check=True)
    # ADetailer
    ad = webui_dir / "extensions" / "adetailer"
    if not ad.exists():
        subprocess.run(["git", "clone", "https://github.com/Bing-su/adetailer", str(ad)], check=True)

    # Launch
    cmd = ["bash", "-lc", "cd /workspace/stable-diffusion-webui && exec python3 launch.py --skip-torch-cuda-test"]
    subprocess.Popen(cmd, env=env)
    print("WebUI starting on :7860 (API mode)")
    time.sleep(5)


if __name__ == "__main__":
    main()


