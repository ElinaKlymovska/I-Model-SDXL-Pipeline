# 🚀 RunPod QuickStart Guide - MyNeuralKingdom

## 📋 Передумови

1. **RunPod акаунт** з доступом до GPU pods
2. **API ключ** RunPod 
3. **SSH ключ** для підключення

## ⚡ Швидкий старт

### 1. Налаштування локального середовища

```bash
# Клонуйте репозиторій
git clone <your-repo-url>
cd MyNeuralKingdom

# Створіть .env файл з вашим API ключем
echo "RUNPOD_API_KEY=your_runpod_api_key_here" > .env

# Встановіть залежності для деплойменту
pip install paramiko python-dotenv requests
```

### 2. Створення RunPod Pod

1. Перейдіть на [RunPod Dashboard](https://www.runpod.io/console/pods)
2. Натисніть **"Deploy"** 
3. Оберіть **GPU Pod** (рекомендовано: RTX A4000+)
4. Оберіть template: **PyTorch 2.0** або **RunPod PyTorch**
5. Налаштуйте:
   - **Container Disk**: 50GB+
   - **Volume**: 100GB+ (для моделей)
   - **Expose HTTP Ports**: 8888 (Jupyter)
6. Натисніть **"Deploy"**

### 3. Отримання Pod ID

```bash
# Знайдіть ваш Pod ID в RunPod Dashboard
# Оновіть runpod/config.py:
pod_id = "your_pod_id_here"
```

### 4. Автоматичний деплоймент

```bash
# Запустіть автоматичний деплоймент
python scripts/deploy_runpod.py
```

Цей скрипт автоматично:
- ✅ Підключиться до вашого pod
- ✅ Завантажить код проєкту
- ✅ Встановить всі залежності
- ✅ Завантажить SDXL моделі (~21GB)
- ✅ Налаштує Jupyter notebook
- ✅ Запустить тести

### 5. Використання

#### Через Jupyter Notebook
```bash
# Знайдіть Jupyter URL в output деплойменту або в RunPod dashboard
https://<port>-<pod_id>.proxy.runpod.net
```

#### Через SSH
```bash
# Підключіться через SSH (порт див. в dashboard)
ssh root@<pod_ip> -p <ssh_port>

cd /workspace
python -c "
from pipelines.inference.enhanced_image_pipeline import EnhancedImagePipeline
pipeline = EnhancedImagePipeline()
pipeline.setup()
print('✅ Pipeline ready!')
"
```

## 🎯 Приклад використання

### Швидкий тест

```python
# У Jupyter notebook або SSH сесії
import sys
sys.path.append('/workspace')

from pipelines.inference.enhanced_image_pipeline import EnhancedImagePipeline
from PIL import Image

# Ініціалізація
pipeline = EnhancedImagePipeline()
pipeline.setup()

# Завантаження моделі
pipeline.load_model('epicrealism_xl')

# Завантаження зображення
image = Image.open('/runpod-volume/data/your_image.jpg')

# Покращення
results = pipeline.run(
    input_data=image,
    model_name='epicrealism_xl',
    enhancement_config='moderate_enhancement',
    output_dir='/runpod-volume/outputs',
    create_comparison=True
)

print(f"✅ Enhanced! Found {results['metadata']['total_faces_detected']} faces")
```

## 📂 Структура файлів у RunPod

```
/workspace/                 # Код проєкту
├── pipelines/             # AI pipeline
├── configs/              # Конфігурації
└── scripts/              # Деплоймент скрипти

/runpod-volume/           # Постійне сховище  
├── models/              # SDXL моделі (~21GB)
├── cache/               # Кеш HuggingFace
├── data/                # Вхідні зображення
├── outputs/             # Результати
└── temp/                # Тимчасові файли
```

## 🔧 Доступні моделі

1. **epicrealism_xl** - найкращий для портретів (6.9GB)
2. **realvis_xl_lightning** - швидкий (8 кроків, 6.9GB) 
3. **juggernaut_xl** - висока деталізація (6.9GB)

## 🎮 Пресети конфігурацій

- `subtle_enhancement` - мінімальні зміни
- `moderate_enhancement` - збалансований (рекомендовано)
- `strong_enhancement` - сильне покращення
- `portrait_focus` - оптимізовано для портретів

## 📊 Моніторинг ресурсів

```python
# Перевірка системи
status = pipeline.get_pipeline_status()
print("GPU Memory:", status['resource_status']['memory']['gpu_memory'])
print("Loaded models:", status['model_manager']['loaded_models'])
```

## 💡 Корисні поради

1. **Завантажуйте зображення** у `/runpod-volume/data/`
2. **Результати** збережуться у `/runpod-volume/outputs/`
3. **Батч-обробка** для кількох зображень:
   ```python
   pipeline.optimize_for_batch_processing()
   results = pipeline.run(input_data=['img1.jpg', 'img2.jpg', ...])
   ```
4. **Переключення моделей**:
   ```python
   pipeline.switch_model('realvis_xl_lightning', unload_current=True)
   ```

## ⚠️ Troubleshooting

### Pod не запускається
- Перевірте баланс RunPod
- Оберіть іншу доступну GPU

### Помилки завантаження моделей
- Переконайтеся, що є 25GB+ вільного місця
- Перезапустіть деплоймент

### Jupyter недоступний
- Перевірте експоновані порти в налаштуваннях pod
- Використайте URL з RunPod dashboard

## 💰 Вартість

- **RTX A4000** (~$0.34/год) - рекомендовано
- **RTX 4090** (~$0.69/год) - найшвидший
- **A100** (~$1.89/год) - максимальна потужність

## 🎉 Готово!

Ваш MyNeuralKingdom готовий до покращення зображень з потужністю хмарних GPU! 🚀

Для питань та підтримки див. документацію або створіть issue у репозиторії.