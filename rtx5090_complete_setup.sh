#!/bin/bash
echo "🔥 RTX 5090 COMPLETE SETUP SCRIPT"
echo "=================================="

# Крок 1: Перевірка RTX 5090
echo "📊 Крок 1: Перевірка RTX 5090..."
nvidia-smi
echo ""
df -h
echo ""

# Крок 2: Перехід в робочу директорію
echo "📂 Крок 2: Перехід в /workspace..."
cd /workspace
pwd
ls -la
echo ""

# Крок 3: Створення проекту та requirements
echo "📋 Крок 3: Створення проекту I-Model..."
mkdir -p I-Model && cd I-Model

cat > requirements.txt << EOF
torch==2.1.0
torchvision==0.16.0
diffusers==0.21.4
transformers==4.30.0
accelerate==0.23.0
xformers==0.0.22
requests>=2.31.0
python-dotenv>=1.0.0
PyYAML>=6.0.0
Pillow>=11.0.0
numpy>=1.26.0
opencv-python>=4.8.0
EOF

echo "✅ Requirements.txt створено!"
echo ""

# Крок 4: Встановлення Python залежностей
echo "🔧 Крок 4: Встановлення Python залежностей..."
pip install -r requirements.txt
echo "✅ Dependencies встановлено!"
echo ""

# Крок 5: Клонування Stable Diffusion WebUI
echo "📥 Крок 5: Клонування Stable Diffusion WebUI..."
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git sd-webui
cd sd-webui
echo "✅ WebUI клоновано!"
echo ""

# Крок 6: Встановлення ADetailer
echo "🎯 Крок 6: Встановлення ADetailer extension..."
git clone https://github.com/Bing-su/adetailer.git extensions/adetailer
echo "✅ ADetailer встановлено!"
echo ""

# Крок 7: Створення директорій для моделей
echo "📁 Крок 7: Створення директорій для моделей..."
mkdir -p models/{Stable-diffusion,adetailer}
echo "✅ Директорії створено!"
echo ""

# Крок 8: Завантаження SDXL моделей
echo "📥 Крок 8: Завантаження SDXL моделей..."
cd models/Stable-diffusion

echo "  📥 Завантаження RealVisXL Lightning..."
wget -O RealVisXL_Lightning.safetensors "https://civitai.com/api/download/models/798204?type=Model&format=SafeTensor"

echo "  📥 Завантаження epiCRealism XL..."
wget -O epiCRealism_XL.safetensors "https://civitai.com/api/download/models/1920523?type=Model&format=SafeTensor"

echo "✅ SDXL моделі завантажено!"
echo ""

# Крок 9: Завантаження ADetailer моделей
echo "📥 Крок 9: Завантаження ADetailer моделей..."
cd ../adetailer

echo "  📥 Завантаження face_yolov8n.pt..."
wget https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt

echo "  📥 Завантаження face_yolov8m.pt..."
wget https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt

echo "✅ ADetailer моделі завантажено!"
echo ""

# Крок 10: Повернення в корінь WebUI
cd ../../../
pwd

# Крок 11: Створення launch скрипта з RTX 5090 оптимізаціями
echo "🚀 Крок 10: Створення launch script..."
cat > launch_rtx5090.py << EOF
import os
import subprocess

# RTX 5090 оптимізації
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

# Перехід в директорію WebUI
os.chdir("sd-webui")

# Запуск з RTX 5090 оптимізаціями
print("🔥 Запуск Stable Diffusion WebUI для RTX 5090...")
print("🌐 WebUI буде доступний на: https://bzwxjffsuvuucj-7860.proxy.runpod.net")
print("")

subprocess.run([
    "python", "launch.py",
    "--xformers",                    # Оптимізації пам'яті
    "--opt-split-attention",         # Розділена увага
    "--api",                         # API доступ
    "--listen",                      # Прослуховування всіх IP
    "--port", "7860",               # Порт
    "--enable-insecure-extension-access"  # Доступ до extensions
])
EOF

echo "✅ Launch script створено!"
echo ""

echo "🎉 SETUP ЗАВЕРШЕНО!"
echo "=================="
echo "📊 Статистика:"
echo "  💾 Використано диску: $(du -sh . | cut -f1)"
echo "  🎯 GPU: RTX 5090 (32GB VRAM)"
echo "  📁 Розташування: $(pwd)"
echo ""
echo "🚀 ДЛЯ ЗАПУСКУ WebUI:"
echo "python launch_rtx5090.py"
echo ""
echo "🌐 ПІСЛЯ ЗАПУСКУ ВІДКРИЙТЕ:"
echo "https://bzwxjffsuvuucj-7860.proxy.runpod.net"
echo ""
echo "⚡ Очікувані швидкості RTX 5090:"
echo "  • SDXL 1024x1024: ~8-12 секунд"
echo "  • Batch 4 images: можливо!"
echo "  • Resolution до 1536x1536: без проблем!"