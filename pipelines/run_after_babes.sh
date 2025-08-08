#!/usr/bin/env bash
set -euo pipefail

LOG_DIR=/workspace/logs
MODELS_DIR=/workspace/models/Stable-diffusion
TARGET="$MODELS_DIR/BabesIllustrious_SDXL_fp16.safetensors"
GONZ="$MODELS_DIR/GonzalomoXL_Fluxpony_fp16.safetensors"

mkdir -p "$LOG_DIR"

echo "[auto] Waiting for BabesIllustrious to finish downloading..." | tee -a "$LOG_DIR/auto_orchestrator.log"

last=-1
while true; do
  if [ -f "$TARGET" ]; then
    size=$(stat -c %s "$TARGET" 2>/dev/null || stat -f %z "$TARGET" 2>/dev/null || echo 0)
    echo "[auto] current size: $size bytes" | tee -a "$LOG_DIR/auto_orchestrator.log"
    if [ "$size" -gt 1000000000 ] && [ "$size" = "$last" ]; then
      break
    fi
    last=$size
  fi
  sleep 20
done

echo "[auto] Babes model ready. Starting WebUI..." | tee -a "$LOG_DIR/auto_orchestrator.log"
cd /workspace/stable-diffusion-webui
[ -d extensions/sd-webui-controlnet ] || git clone https://github.com/Mikubill/sd-webui-controlnet extensions/sd-webui-controlnet
[ -d extensions/adetailer ] || git clone https://github.com/Bing-su/adetailer extensions/adetailer
export COMMANDLINE_ARGS="--xformers --api --listen --port 7860 --enable-insecure-extension-access"
setsid python3 launch.py </dev/null > "$LOG_DIR/webui.log" 2>&1 & echo $! > "$LOG_DIR/webui.pid"

for i in $(seq 1 120); do
  if curl -s http://127.0.0.1:7860/sdapi/v1/progress >/dev/null 2>&1; then
    echo "[auto] WebUI API is up." | tee -a "$LOG_DIR/auto_orchestrator.log"
    break
  fi
  sleep 5
  if [ $i -eq 120 ]; then echo "[auto] WebUI did not come up in time" | tee -a "$LOG_DIR/auto_orchestrator.log"; fi
done

echo "[auto] Running Gonzalomo XL batch..." | tee -a "$LOG_DIR/auto_orchestrator.log"
python3 /workspace/hyperrealistic/pipelines/process_faces.py --model gonzalomo_xl --per-image 1 --input /workspace/data/input --output /workspace/data/outputs > "$LOG_DIR/batch_gonz.log" 2>&1 || true

echo "[auto] Running Babes Illustrious batch..." | tee -a "$LOG_DIR/auto_orchestrator.log"
python3 /workspace/hyperrealistic/pipelines/process_faces.py --model babes_illustrious --per-image 1 --input /workspace/data/input --output /workspace/data/outputs > "$LOG_DIR/batch_babes.log" 2>&1 || true

echo "[auto] All batches finished." | tee -a "$LOG_DIR/auto_orchestrator.log"


