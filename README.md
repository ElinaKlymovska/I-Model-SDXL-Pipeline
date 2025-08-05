# 🏰 MyNeuralKingdom - AI Image Enhancement Pipeline

**Професійна система покращення зображень на базі SDXL з інтелектуальним виявленням облич**

## ✨ Особливості

🤖 **SDXL Image Enhancement**
- 3 оптимізовані SDXL моделі для різних сценаріїв
- Автоматичне виявлення та покращення облич
- Батч-обробка множини зображень
- Гнучкі пресети конфігурацій

🏗️ **Enterprise Architecture**
- Service-oriented design з dependency injection
- Модульна архітектура з чіткими інтерфейсами
- Comprehensive logging та performance tracking
- Resource monitoring та optimization

☁️ **Cloud Ready**
- Повна інтеграція з RunPod
- Автоматичний деплоймент та налаштування
- Persistent storage для моделей
- Jupyter notebook інтерфейс

👤 **Smart Face Detection**
- MediaPipe (основний детектор)
- OpenCV fallback детектор
- Автоматичне перемикання детекторів
- Спеціалізована обробка лиць

## 🚀 Швидкий старт з RunPod

### 1. Підготовка
```bash
git clone <your-repo>
cd MyNeuralKingdom

# Створіть .env файл
echo "RUNPOD_API_KEY=your_api_key" > .env
```

### 2. Запуск в RunPod
```bash
# Автоматичний деплоймент
python scripts/deploy_runpod.py
```

### 3. Використання
- 📓 **Jupyter**: `https://<port>-<pod_id>.proxy.runpod.net`
- 🖼️ **Завантажте зображення**: `/runpod-volume/data/`
- 🎨 **Результати**: `/runpod-volume/outputs/`

**Детальні інструкції**: [RUNPOD_QUICKSTART.md](RUNPOD_QUICKSTART.md)

## 🧪 Локальне тестування (без GPU)

```bash
# Тест виявлення облич
python -c "
from services.face_detection import FaceDetectionService
from core.config import get_config_manager
from PIL import Image

detector = FaceDetectionService(get_config_manager())
detector.setup()

# Тест на зображенні
image = Image.open('data/input/your_image.jpg')
faces = detector.detect_faces(image)
print(f'Знайдено {len(faces)} облич')
"
```

## 🤖 Доступні моделі

| Модель | Призначення | Швидкість | VRAM | Розмір |
|--------|-------------|-----------|------|--------|
| **epicrealism_xl** | Портрети | 25 кроків | 10GB | 6.9GB |
| **realvis_xl_lightning** | Швидкий | 8 кроків | 8GB | 6.9GB |
| **juggernaut_xl** | Деталізація | 30 кроків | 12GB | 6.9GB |

## ⚙️ Пресети конфігурацій

```python
# Доступні пресети
"subtle_enhancement"     # Мінімальні зміни
"moderate_enhancement"   # Збалансований (рекомендовано)
"strong_enhancement"     # Сильне покращення  
"portrait_focus"         # Оптимізовано для портретів
```

## 💻 Приклад використання

```python
from pipelines.inference.enhanced_image_pipeline import EnhancedImagePipeline
from core.config import get_config_manager
from core.container import create_default_container

# Ініціалізація
pipeline = EnhancedImagePipeline(get_config_manager(), create_default_container())
pipeline.setup()

# Завантаження моделі
pipeline.load_model('epicrealism_xl')

# Покращення зображення
results = pipeline.run(
    input_data='path/to/image.jpg',
    model_name='epicrealism_xl',
    enhancement_config='moderate_enhancement',
    output_dir='outputs/',
    create_comparison=True
)

print(f"✅ Enhanced! Found {results['metadata']['total_faces_detected']} faces")
```

## 🔧 Архітектура проєкту

```
MyNeuralKingdom/
├── core/                    # Основні компоненти
│   ├── config.py           # Конфігурація системи
│   ├── container.py        # Dependency injection
│   └── interfaces.py       # Абстракції та протоколи
├── services/               # Бізнес логіка
│   ├── face_detection.py   # Виявлення облич
│   ├── image_enhancer.py   # SDXL покращення
│   ├── model_manager.py    # Керування моделями
│   └── resource_manager.py # Моніторинг ресурсів
├── pipelines/             # ML pipeline
│   └── inference/
│       └── enhanced_image_pipeline.py
├── runpod/               # RunPod інтеграція
│   ├── manager.py        # SSH, файли, команди
│   └── config.py         # Хмарні налаштування
└── scripts/
    └── deploy_runpod.py  # Автоматичний деплоймент
```

## 📊 Ресурси

### Мінімальні вимоги RunPod:
- **GPU**: RTX A4000+ (10GB+ VRAM)
- **Storage**: 100GB+ volume для моделей
- **Memory**: 16GB+ RAM

### Рекомендовані конфігурації:
- **RTX A4000** (~$0.34/год) - оптимально
- **RTX 4090** (~$0.69/год) - швидкий
- **A100** (~$1.89/год) - максимальна потужність

## 🧪 Тестування в RunPod

```bash
# Швидкий тест всієї системи
python runpod_test.py
```

Цей скрипт перевірить:
- ✅ GPU та CUDA
- ✅ Pipeline ініціалізацію  
- ✅ Завантаження моделей
- ✅ Покращення зображень

## 📚 Документація

- [📖 RunPod QuickStart](RUNPOD_QUICKSTART.md) - Повний гід з деплойменту
- [🏗️ Architecture Guide](docs/ARCHITECTURE.md) - Архітектура системи
- [🔧 Configuration](docs/CONFIGURATION.md) - Налаштування та конфігурація
- [🚀 API Reference](docs/API.md) - Програмний інтерфейс

## 🤝 Contributing

1. Fork репозиторій
2. Створіть feature branch (`git checkout -b feature/amazing-feature`)
3. Commit зміни (`git commit -m 'Add amazing feature'`)
4. Push branch (`git push origin feature/amazing-feature`)
5. Створіть Pull Request

## 📄 Ліцензія

Цей проєкт використовує [MIT License](LICENSE).

## 🎯 Roadmap

- [ ] 🌐 Web UI інтерфейс
- [ ] 🔄 Real-time processing
- [ ] 📱 Mobile app integration
- [ ] 🎨 Style transfer моделі
- [ ] 🤗 HuggingFace Spaces deployment
- [ ] 📊 Advanced analytics dashboard

## 🆘 Підтримка

- 📧 **Email**: support@myneuralkingdom.com
- 💬 **Discord**: [Join our community](https://discord.gg/myneuralkingdom)
- 📋 **Issues**: [GitHub Issues](https://github.com/yourrepo/issues)
- 📖 **Docs**: [Documentation Site](https://docs.myneuralkingdom.com)

---

**MyNeuralKingdom** - Ваше королівство штучного інтелекту для покращення зображень! 🏰✨