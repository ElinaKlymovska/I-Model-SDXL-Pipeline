#!/usr/bin/env python3
"""
SDXL Models Download Script for RunPod
Завантажує всі необхідні SDXL моделі на persistent volume
"""

import os
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
import time

def setup_environment():
    """Налаштування середовища"""
    os.environ["TRANSFORMERS_CACHE"] = "/runpod-volume/cache"
    os.environ["HF_HOME"] = "/runpod-volume/cache"
    os.environ["TORCH_HOME"] = "/runpod-volume/cache"
    
    print("🌍 Environment variables set")
    print(f"Cache directory: {os.environ['HF_HOME']}")

def check_gpu():
    """Перевірка GPU"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        return True
    else:
        print("❌ CUDA не доступна!")
        return False

def download_model(model_name, model_id, model_path):
    """Завантажити конкретну модель"""
    print(f"\n🔽 Завантаження {model_name}...")
    print(f"Model ID: {model_id}")
    print(f"Save path: {model_path}")
    
    start_time = time.time()
    
    try:
        # Download model
        pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            cache_dir="/runpod-volume/cache"
        )
        
        # Save to persistent volume
        pipeline.save_pretrained(model_path)
        
        # Check model size
        total_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(model_path)
            for filename in filenames
        ) / (1024**3)  # Convert to GB
        
        load_time = time.time() - start_time
        
        print(f"✅ {model_name} завантажено!")
        print(f"   Розмір: {total_size:.1f}GB")
        print(f"   Час: {load_time:.1f}s")
        
        # Free memory
        del pipeline
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Помилка завантаження {model_name}: {e}")
        return False

def main():
    """Головна функція"""
    print("🏰 MyNeuralKingdom - SDXL Models Downloader")
    print("=" * 50)
    
    # Setup
    setup_environment()
    
    if not check_gpu():
        print("⚠️ GPU не доступна, але завантаження продовжується...")
    
    # Models to download
    models = {
        "epicrealism_xl": "stablediffusionapi/epic-realism-xl",
        "realvis_xl_lightning": "SG161222/RealVisXL_V5.0_Lightning", 
        "juggernaut_xl": "RunDiffusion/Juggernaut-XL-v9"
    }
    
    print(f"\n📚 Завантаження {len(models)} SDXL моделей...")
    
    successful = 0
    failed = 0
    total_start_time = time.time()
    
    for model_name, model_id in models.items():
        model_path = f"/runpod-volume/models/{model_name}"
        
        # Check if already exists
        if os.path.exists(model_path) and os.listdir(model_path):
            print(f"⏭️ {model_name} вже існує, пропускаємо...")
            successful += 1
            continue
        
        # Download model
        if download_model(model_name, model_id, model_path):
            successful += 1
        else:
            failed += 1
    
    total_time = time.time() - total_start_time
    
    # Summary
    print(f"\n📊 Підсумок завантаження:")
    print(f"✅ Успішно: {successful}/{len(models)}")
    print(f"❌ Помилки: {failed}/{len(models)}")
    print(f"⏱️ Загальний час: {total_time/60:.1f} хвилин")
    
    if successful == len(models):
        print("\n🎉 Всі моделі завантажено успішно!")
        print("🚀 MyNeuralKingdom готовий до роботи!")
    else:
        print(f"\n⚠️ Завантаження завершено з помилками")
    
    # Check total size
    models_dir = "/runpod-volume/models"
    if os.path.exists(models_dir):
        total_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(models_dir)
            for filename in filenames
        ) / (1024**3)
        print(f"💾 Загальний розмір моделей: {total_size:.1f}GB")

if __name__ == "__main__":
    main()