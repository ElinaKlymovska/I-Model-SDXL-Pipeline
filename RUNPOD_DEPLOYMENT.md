# 🚀 RunPod Deployment Guide

## Enhanced Face Correction Pipeline на RunPod

### 🎯 Швидкий Старт (1 хвилина)

1. **Створіть RunPod Pod:**
   - GPU: A40, RTX 6000 Ada, або RTX A6000 (рекомендовано)
   - Docker: `runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04`
   - Container Disk: 20GB+
   - Network Volume: 50GB+ (рекомендовано)

2. **Запустіть Quick Start:**
   ```bash
   # У Terminal RunPod
   wget https://raw.githubusercontent.com/ElinaKlymovska/I-Model-SDXL-Pipeline/develop/runpod_quick_start.sh
   chmod +x runpod_quick_start.sh
   bash runpod_quick_start.sh
   ```

3. **Оберіть опцію 5** (Deploy Face Correction) для повного автоматичного налаштування

### 🎭 Автоматичний Deployment

```bash
# Клонувати проект
git clone -b develop https://github.com/ElinaKlymovska/I-Model-SDXL-Pipeline.git
cd I-Model-SDXL-Pipeline

# Автоматичний deployment з демо
python scripts/deploy_runpod_face_correction.py --preset professional --demo
```

### ⚙️ Configuration Management

Всі налаштування тепер зберігаються в config файлах:

- **`config/pipeline_settings.yaml`** - Основні налаштування pipeline
- **`config/models.yaml`** - Конфігурація моделей
- **`config/prompt_settings.yaml`** - Налаштування промптів та якості
- **`.env`** - Тільки sensitive дані (API ключі)

### 🎯 Топові SDXL Моделі

- **copax_realistic_xl** - Спеціаліст з портретної фотографії (default)
- **proteus_xl** - Передова деталізація та фотореалізм
- **newreality_xl** - Експерт з деталей обличчя та текстури шкіри
- **epicrealism_xl** - Epic реалізм та високі деталі

### 🎭 Використання Face Correction

#### 1. Демонстрації
```bash
# Всі демо
python scripts/demo_face_correction.py

# Порівняння моделей
python scripts/demo_face_correction.py --demo models
```

#### 2. Одне Зображення (з defaults з config)
```bash
# Базове покращення (використовує config defaults)
python scripts/demo_pipeline.py --input photo.jpg --output ./results/

# Налаштування перевизначають config
python scripts/demo_pipeline.py --input photo.jpg --output ./results/ \
  --character male_portrait --enhancement strong
```

#### 3. Enhanced ADetailer
```bash
# Використовує defaults з config/pipeline_settings.yaml
python scripts/enhanced_adetailer.py --input photo.jpg --output enhanced.jpg
```

### 📦 Model Presets (з config)

Presets визначені в `config/pipeline_settings.yaml`:

| Preset | Моделі | Швидкість | Якість |
|--------|--------|-----------|--------|
| `basic` | epicrealism_xl | ⚡⚡⚡ | ⭐⭐⭐ |
| `advanced` | epicrealism_xl + copax_realistic_xl | ⚡⚡ | ⭐⭐⭐⭐ |
| `professional` | 4 топові моделі | ⚡ | ⭐⭐⭐⭐⭐ |

### ⚙️ Configuration Override

Можна перевизначити будь-які налаштування:

```bash
# CLI аргументи перевизначають config
python scripts/demo_pipeline.py --input photo.jpg --output ./results/ \
  --model newreality_xl \
  --face-model face_yolov8l.pt \
  --enhancement extreme \
  --quality aggressive

# Або змінити defaults у config/pipeline_settings.yaml
```

### 🚀 Ready to Go!

Після завершення setup ваша система готова до професійної корекції лиця з конфігурацією що зберігається в YAML файлах!

🌐 **WebUI**: `https://[pod-id]-3000.proxy.runpod.net`
🎭 **Demo**: `python scripts/demo_face_correction.py`
⚙️ **Config**: редагуйте `config/pipeline_settings.yaml` для зміни defaults