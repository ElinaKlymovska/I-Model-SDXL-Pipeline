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

### ⚙️ Ручне Налаштування

```bash
# 1. Клонувати та встановити
git clone -b develop https://github.com/ElinaKlymovska/I-Model-SDXL-Pipeline.git
cd I-Model-SDXL-Pipeline
pip install -r requirements_runpod.txt

# 2. Налаштування з професійними моделями
python utils/runpod_launcher.py --setup-only --preset professional --face-models

# 3. Завантажити моделі
python utils/runpod_launcher.py --download-only --preset professional --face-models

# 4. Запустити WebUI
python utils/runpod_launcher.py --launch
```

### 🌐 Доступ до WebUI

- **URL**: `https://[ваш-pod-id]-3000.proxy.runpod.net`
- **Local**: `http://localhost:3000` (якщо використовуєте SSH)

### 🎨 Використання Face Correction

#### 1. Демонстрації
```bash
# Всі демо
python scripts/demo_face_correction.py

# Порівняння моделей
python scripts/demo_face_correction.py --demo models

# Тест рівнів покращення
python scripts/demo_face_correction.py --demo levels
```

#### 2. Одне Зображення
```bash
# Базове покращення
python scripts/demo_pipeline.py --input photo.jpg --output ./results/

# Професійний портрет
python scripts/demo_pipeline.py --input photo.jpg --output ./results/ \
  --character female_portrait --model copax_realistic_xl \
  --enhancement strong --quality aggressive
```

#### 3. Пакетна Обробка
```bash
# Пакетна обробка з високою якістю
python scripts/demo_pipeline.py --batch ./photos/ --output ./results/ \
  --model proteus_xl --detail-model newreality_xl \
  --enhancement strong --quality aggressive
```

#### 4. Тільки ADetailer
```bash
# Швидка корекція лиця
python scripts/enhanced_adetailer.py --input photo.jpg --output enhanced.jpg \
  --model copax_realistic_xl --face-model face_yolov8m.pt --quality balanced
```

### 📦 Доступні Пресети

| Preset | Моделі | Швидкість | Якість |
|--------|--------|-----------|--------|
| `basic` | epicrealism_xl | ⚡⚡⚡ | ⭐⭐⭐ |
| `advanced` | epicrealism_xl + copax_realistic_xl | ⚡⚡ | ⭐⭐⭐⭐ |
| `professional` | 4 топові моделі | ⚡ | ⭐⭐⭐⭐⭐ |

### 🎯 Топові SDXL Моделі

- **copax_realistic_xl** - Спеціаліст з портретної фотографії
- **proteus_xl** - Передова деталізація та фотореалізм
- **newreality_xl** - Експерт з деталей обличчя та текстури шкіри
- **epicrealism_xl** - Epic реалізм та високі деталі

### 👁️ Face Detection Моделі

- **face_yolov8s.pt** - Збалансований (рекомендовано)
- **face_yolov8m.pt** - Висока точність
- **face_yolov8l.pt** - Максимальна якість (повільніше)
- **face_yolov8x.pt** - Професійна точність (найповільніше)

### 🔧 Налаштування Якості

#### Enhancement Levels:
- `light` - М'яке покращення, швидко
- `medium` - Збалансоване покращення (рекомендовано)
- `strong` - Сильне покращення
- `extreme` - Максимальне покращення

#### Quality Presets:
- `conservative` - Безпечні зміни, мінімальні артефакти
- `balanced` - Гарний баланс якості/швидкості (рекомендовано)
- `aggressive` - Максимальна якість, повільніше

### 🎭 Типи Персонажів

- `female_portrait` - Жіночий портрет (рекомендовано)
- `male_portrait` - Чоловічий портрет
- `child_portrait` - Дитячий портрет

### 🚀 Performance Tips

1. **GPU Рекомендації:**
   - A40 (48GB VRAM) - ідеально ✅
   - RTX 6000 Ada (48GB VRAM) - чудово ✅
   - RTX A6000 (48GB VRAM) - відмінно ✅
   - RTX 4090 (24GB VRAM) - добре ⚠️

2. **Швидкість vs Якість:**
   - Швидко: `--preset basic --enhancement light --quality conservative`
   - Збалансовано: `--preset advanced --enhancement medium --quality balanced`
   - Максимум: `--preset professional --enhancement strong --quality aggressive`

3. **Network Volume:**
   - Використовуйте Network Volume для зберігання моделей
   - Економить час на перезавантаження
   - Моделі: ~25GB, загалом потрібно 50GB+

### 🔄 Restart та Troubleshooting

```bash
# Перезапуск WebUI
python utils/runpod_launcher.py --launch

# Швидкий старт
bash start_auto_processing.sh

# Оновити проект
git pull origin develop

# Примусово CPU режим (якщо проблеми з GPU)
python utils/runpod_launcher.py --force-cpu --launch
```

### 📊 Моніторинг

- **GPU використання**: `nvidia-smi`
- **WebUI логи**: дивіться Terminal під час запуску
- **Process статус**: `ps aux | grep python`

### 🎉 Ready to Go!

Після завершення setup ваша система готова до професійної корекції лиця! 

🌐 **WebUI**: `https://[pod-id]-3000.proxy.runpod.net`
🎭 **Demo**: `python scripts/demo_face_correction.py`