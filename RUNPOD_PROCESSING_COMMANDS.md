# 🚀 RunPod Processing Commands

## Готові команди для обробки зображень на RunPod

### 1️⃣ Quick Start RunPod Setup
```bash
# На RunPod Terminal:
wget https://raw.githubusercontent.com/ElinaKlymovska/I-Model-SDXL-Pipeline/develop/runpod_quick_start.sh
chmod +x runpod_quick_start.sh
bash runpod_quick_start.sh
# Оберіть опцію 5 (Deploy Face Correction)
```

### 2️⃣ Автоматичний Deployment
```bash
git clone -b develop https://github.com/ElinaKlymovska/I-Model-SDXL-Pipeline.git
cd I-Model-SDXL-Pipeline
python scripts/deploy_runpod_face_correction.py --preset professional --demo
```

### 3️⃣ Manual Setup з всіма моделями
```bash
cd I-Model-SDXL-Pipeline
python utils/runpod_launcher.py --preset professional --face-models --demo
```

### 4️⃣ Обробка ВАШИХ зображень

#### 📁 Upload ваші фотографії:
```bash
# Завантажте зображення в data/input/
mkdir -p data/input
# (Upload your photos via RunPod file manager або scp)
```

#### 🖼️ Одне зображення (з config defaults):
```bash
python scripts/demo_pipeline.py \
  --input data/input/your_photo.jpg \
  --output ./results/
```

#### 📦 Batch processing всіх зображень:
```bash
python scripts/demo_pipeline.py \
  --batch data/input/ \
  --output ./results/ \
  --model copax_realistic_xl \
  --enhancement medium \
  --quality balanced
```

#### 🎭 High Quality Professional:
```bash
python scripts/demo_pipeline.py \
  --batch data/input/ \
  --output ./results/ \
  --model proteus_xl \
  --detail-model newreality_xl \
  --enhancement strong \
  --quality aggressive
```

#### ⚡ Enhanced ADetailer тільки (швидко):
```bash
python scripts/enhanced_adetailer.py \
  --input data/input/photo.jpg \
  --output enhanced.jpg \
  --model copax_realistic_xl \
  --quality balanced
```

### 5️⃣ Поточні Config Defaults
```yaml
📦 SDXL модель: copax_realistic_xl
👁️ Face detection: face_yolov8m.pt  
🎭 Character type: female_portrait
⚡ Enhancement: medium
🎨 Quality: balanced
🌐 WebUI Port: 3000
```

### 6️⃣ WebUI Access
```
https://[your-pod-id]-3000.proxy.runpod.net
```

### 7️⃣ Download Results
```bash
# Пакування результатів:
cd results/
tar -czf enhanced_photos_$(date +%Y%m%d).tar.gz *
# Download via RunPod file manager
```

---

## 🎯 Ready to Go Commands

```bash
# 1. Швидкий setup:
bash runpod_quick_start.sh

# 2. Обробка всіх фото:
python scripts/demo_pipeline.py --batch data/input/ --output ./results/

# 3. Professional качество:
python scripts/demo_pipeline.py --batch data/input/ --output ./results/ \
  --model proteus_xl --enhancement strong --quality aggressive
```

**Система готова до професійної обробки фотографій! 🎭✨**