#!/usr/bin/env python3
"""
RunPod Quick Test Script
Швидка перевірка функціональності в хмарному середовищі RunPod
"""

import sys
import time
import logging
from pathlib import Path
from PIL import Image
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_environment():
    """Перевірка середовища RunPod"""
    print("🔍 Перевірка середовища RunPod...")
    
    # Check CUDA
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB VRAM)")
    else:
        print("❌ CUDA недоступна!")
        return False
    
    # Check paths
    workspace = Path("/workspace")
    volume = Path("/runpod-volume")
    
    if workspace.exists():
        print(f"✅ Workspace: {workspace}")
    else:
        print(f"❌ Workspace не знайдено: {workspace}")
        
    if volume.exists():
        print(f"✅ Volume: {volume}")
    else:
        print(f"❌ Volume не знайдено: {volume}")
    
    return True

def test_pipeline():
    """Тест основного pipeline"""
    print("\n🧪 Тестування Image Enhancement Pipeline...")
    
    try:
        # Add workspace to path
        sys.path.append("/workspace")
        
        from core.config import get_config_manager
        from core.container import create_default_container
        from pipelines.inference.enhanced_image_pipeline import EnhancedImagePipeline
        
        # Initialize pipeline
        config_manager = get_config_manager()
        container = create_default_container()
        pipeline = EnhancedImagePipeline(config_manager, container)
        
        # Setup
        if not pipeline.setup():
            print("❌ Pipeline setup failed")
            return False
        
        print("✅ Pipeline ініціалізовано")
        
        # Check models
        available_models = config_manager.list_available_models()
        print(f"📚 Доступні моделі: {', '.join(available_models)}")
        
        # Check status
        status = pipeline.get_pipeline_status()
        print(f"🔧 Face detection: {status['face_detection']['current_detector']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline тест failed: {e}")
        return False

def test_model_loading():
    """Тест завантаження моделі"""
    print("\n⬇️ Тестування завантаження моделі...")
    
    try:
        sys.path.append("/workspace")
        from pipelines.inference.enhanced_image_pipeline import EnhancedImagePipeline
        from core.config import get_config_manager
        from core.container import create_default_container
        
        pipeline = EnhancedImagePipeline(get_config_manager(), create_default_container())
        pipeline.setup()
        
        # Try to load fastest model
        model_name = "realvis_xl_lightning"
        print(f"🚀 Завантаження {model_name}...")
        
        start_time = time.time()
        if pipeline.load_model(model_name):
            load_time = time.time() - start_time
            print(f"✅ Модель завантажена за {load_time:.1f}s")
            
            # Check memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated(0) / 1e9
                print(f"📊 GPU memory використано: {memory_used:.1f}GB")
            
            return True
        else:
            print(f"❌ Не вдалося завантажити {model_name}")
            return False
            
    except Exception as e:
        print(f"❌ Model loading тест failed: {e}")
        return False

def test_enhancement():
    """Тест покращення зображення"""
    print("\n🖼️ Тестування покращення зображення...")
    
    try:
        sys.path.append("/workspace")
        from pipelines.inference.enhanced_image_pipeline import EnhancedImagePipeline
        from core.config import get_config_manager
        from core.container import create_default_container
        
        pipeline = EnhancedImagePipeline(get_config_manager(), create_default_container())
        pipeline.setup()
        
        # Load model
        model_name = "realvis_xl_lightning"
        if not pipeline.load_model(model_name):
            print(f"❌ Не вдалося завантажити модель")
            return False
        
        # Create test image
        test_image = Image.new("RGB", (512, 512), color=(100, 150, 200))
        
        # Test enhancement with minimal settings
        print("🎨 Запуск швидкого тесту...")
        start_time = time.time()
        
        results = pipeline.run(
            input_data=test_image,
            model_name=model_name,
            enhancement_config={
                "strength": 0.1,
                "num_inference_steps": 4,  # Very fast test
                "guidance_scale": 6.0
            },
            output_dir="/runpod-volume/outputs/test"
        )
        
        process_time = time.time() - start_time
        
        if results["status"] == "success":
            print(f"✅ Тест покращення успішний за {process_time:.1f}s")
            metadata = results["metadata"]
            print(f"📊 Оброблено: {metadata['total_images']} зображень")
            print(f"👤 Знайдено облич: {metadata['total_faces_detected']}")
            return True
        else:
            print(f"❌ Тест покращення failed: {results.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Enhancement тест failed: {e}")
        return False

def create_sample_images():
    """Створення тестових зображень"""
    print("\n📸 Створення тестових зображень...")
    
    output_dir = Path("/runpod-volume/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test images
    colors = [
        ("red_test.jpg", (255, 100, 100)),
        ("green_test.jpg", (100, 255, 100)),
        ("blue_test.jpg", (100, 100, 255))
    ]
    
    for filename, color in colors:
        image = Image.new("RGB", (512, 512), color=color)
        image_path = output_dir / filename
        image.save(image_path)
        print(f"✅ Створено: {image_path}")
    
    print(f"📁 Тестові зображення збережено в {output_dir}")

def main():
    """Головна функція тестування"""
    print("🚀 RunPod MyNeuralKingdom - Швидкий тест")
    print("=" * 50)
    
    tests = [
        ("Environment Check", test_environment),
        ("Pipeline Initialization", test_pipeline),
        ("Model Loading", test_model_loading),
        ("Enhancement Test", test_enhancement)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results[test_name] = success
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"\n{status}: {test_name}")
        except Exception as e:
            print(f"\n❌ FAILED: {test_name} - {e}")
            results[test_name] = False
    
    # Create sample images regardless of test results
    create_sample_images()
    
    # Summary
    print(f"\n{'='*20} SUMMARY {'='*20}")
    passed = sum(results.values())
    total = len(results)
    
    print(f"Тестів пройдено: {passed}/{total}")
    
    for test_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {test_name}")
    
    if passed == total:
        print(f"\n🎉 Всі тести пройдено! MyNeuralKingdom готовий до роботи в RunPod!")
        print(f"🚀 Можете завантажувати зображення в /runpod-volume/data/ та запускати покращення!")
    else:
        print(f"\n⚠️ Деякі тести не пройдено. Перевірте налаштування та спробуйте знову.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)