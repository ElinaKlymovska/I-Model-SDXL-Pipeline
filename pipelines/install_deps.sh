#!/usr/bin/env bash
set -euo pipefail

echo "[deps] Installing Python deps for pipeline..."
python3 -m pip install --upgrade pip
python3 -m pip install insightface==0.7.3 face_recognition==1.3.0 opencv-python-headless==4.10.0.84 pillow==10.4.0 requests==2.32.3 numpy==1.26.4

echo "[deps] Ensuring WebUI repo exists..."
if [ ! -d /workspace/stable-diffusion-webui ]; then
  git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui /workspace/stable-diffusion-webui
fi

echo "[deps] Done."


