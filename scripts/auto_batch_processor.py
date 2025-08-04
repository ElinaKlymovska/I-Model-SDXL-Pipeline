#!/usr/bin/env python3
"""
Автоматичний batch процесор для фотографій
Завантажує фото, обробляє з ADetailer + SDXL, автоматично скачує результати
"""

import os
import sys
import json
import yaml
import asyncio
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import shutil
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime

# Додамо шлях до scripts для імпорту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from run_adetailer import run_adetailer
    from run_img2img import run_img2img
except ImportError as e:
    logging.error(f"Failed to import pipeline modules: {e}")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class AutoBatchProcessor:
    def __init__(self, config_path: str = "config/batch_config.yaml"):
        """Ініціалізація автоматичного процесора"""
        self.config_path = config_path
        self.config = self.load_config()
        self.prompt_settings = self.load_prompt_settings()
        self.results = []
        
        # Створити необхідні директорії
        self.setup_directories()
        
    def load_config(self) -> Dict[str, Any]:
        """Завантажити конфігурацію batch обробки"""
        default_config = {
            "input_dir": "assets/input",
            "output_dir": "assets/output",
            "temp_dir": "assets/temp",
            "download_dir": "assets/downloads",
            "supported_formats": [".jpg", ".jpeg", ".png", ".webp", ".bmp"],
            "default_model": "epicrealism_xl",
            "default_preset": "professional_headshot",
            "default_enhancement": "medium",
            "max_workers": 2,  # Для A40 можна 2-3 одночасно
            "auto_cleanup": True,
            "generate_metadata": True
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                default_config.update(user_config)
            except Exception as e:
                logging.warning(f"Помилка завантаження конфігурації: {e}")
                
        return default_config
    
    def load_prompt_settings(self) -> Dict[str, Any]:
        """Завантажити налаштування промптів"""
        try:
            with open("config/prompt_settings.yaml", 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Помилка завантаження prompt_settings.yaml: {e}")
            return {}
    
    def setup_directories(self):
        """Створити необхідні директорії"""
        dirs = [
            self.config["input_dir"],
            self.config["output_dir"], 
            self.config["temp_dir"],
            self.config["download_dir"]
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"📁 Директорія готова: {dir_path}")
    
    def find_input_images(self, input_path: str) -> List[str]:
        """Знайти всі зображення для обробки"""
        images = []
        supported_exts = self.config["supported_formats"]
        
        if os.path.isfile(input_path):
            if any(input_path.lower().endswith(ext) for ext in supported_exts):
                images.append(input_path)
        elif os.path.isdir(input_path):
            for ext in supported_exts:
                images.extend(Path(input_path).glob(f"*{ext}"))
                images.extend(Path(input_path).glob(f"*{ext.upper()}"))
            images = [str(img) for img in images]
        
        logging.info(f"🖼️  Знайдено {len(images)} зображень для обробки")
        return images
    
    def get_processing_params(self, preset: str, model: str, enhancement: str) -> Dict[str, Any]:
        """Отримати параметри обробки на основі preset"""
        params = {
            "model": model,
            "preset": preset,
            "enhancement": enhancement
        }
        
        # Отримати prompt з preset
        if preset in self.prompt_settings.get("presets", {}):
            preset_config = self.prompt_settings["presets"][preset]
            params["positive_prompt"] = preset_config.get("positive", "")
            params["negative_prompt"] = preset_config.get("negative", "")
            params["enhancement_level"] = preset_config.get("enhancement_level", enhancement)
        else:
            # Базові промпти
            params["positive_prompt"] = self.prompt_settings.get("default_prompts", {}).get("base_positive", "")
            params["negative_prompt"] = self.prompt_settings.get("default_prompts", {}).get("base_negative", "")
        
        # Додати model-specific налаштування
        if model in self.prompt_settings.get("model_specific", {}):
            model_config = self.prompt_settings["model_specific"][model]
            params["positive_prompt"] += model_config.get("positive_suffix", "")
            params["negative_prompt"] += model_config.get("negative_suffix", "")
            params["cfg_scale"] = model_config.get("recommended_cfg", 7)
            params["steps"] = model_config.get("recommended_steps", 30)
        
        # Додати enhancement level налаштування
        if enhancement in self.prompt_settings.get("enhancement_levels", {}):
            enh_config = self.prompt_settings["enhancement_levels"][enhancement]
            params["denoising_strength"] = enh_config.get("denoising_strength", 0.4)
            params["cfg_scale"] = enh_config.get("cfg_scale", 7)
            params["steps"] = enh_config.get("steps", 30)
        
        return params
    
    def process_single_image(self, image_path: str, params: Dict[str, Any], 
                           output_suffix: str = "") -> Dict[str, Any]:
        """Обробити одне зображення"""
        start_time = time.time()
        image_name = Path(image_path).stem
        
        # Генерувати назви файлів
        temp_path = os.path.join(self.config["temp_dir"], f"{image_name}_adetailer{output_suffix}.png")
        output_path = os.path.join(self.config["output_dir"], f"{image_name}_enhanced{output_suffix}.png")
        
        result = {
            "input_path": image_path,
            "output_path": output_path,
            "temp_path": temp_path,
            "success": False,
            "error": None,
            "processing_time": 0,
            "params": params,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            logging.info(f"🎨 Обробка: {image_name}")
            
            # Step 1: ADetailer
            logging.info(f"  🧠 Крок 1: ADetailer для {image_name}")
            run_adetailer(image_path, temp_path, params["model"])
            
            if not os.path.exists(temp_path):
                raise RuntimeError("ADetailer не створив результат")
            
            # Step 2: SDXL img2img
            logging.info(f"  🎨 Крок 2: SDXL img2img для {image_name}")
            run_img2img(
                temp_path, 
                output_path, 
                params["positive_prompt"], 
                params["negative_prompt"], 
                params["model"]
            )
            
            if not os.path.exists(output_path):
                raise RuntimeError("SDXL img2img не створив результат")
            
            # Очистити temp файл
            if self.config["auto_cleanup"] and os.path.exists(temp_path):
                os.remove(temp_path)
            
            result["success"] = True
            result["processing_time"] = time.time() - start_time
            
            logging.info(f"✅ {image_name} оброблено за {result['processing_time']:.1f}с")
            
        except Exception as e:
            result["error"] = str(e)
            result["processing_time"] = time.time() - start_time
            logging.error(f"❌ Помилка обробки {image_name}: {e}")
            
            # Очистити артефакти при помилці
            for path in [temp_path, output_path]:
                if os.path.exists(path):
                    os.remove(path)
        
        return result
    
    def batch_process(self, input_path: str, preset: str = None, 
                     model: str = None, enhancement: str = None,
                     max_workers: int = None) -> List[Dict[str, Any]]:
        """Batch обробка зображень"""
        
        # Використати значення за замовчуванням
        preset = preset or self.config["default_preset"]
        model = model or self.config["default_model"]
        enhancement = enhancement or self.config["default_enhancement"]
        max_workers = max_workers or self.config["max_workers"]
        
        # Знайти всі зображення
        images = self.find_input_images(input_path)
        
        if not images:
            logging.warning("Жодного зображення не знайдено!")
            return []
        
        # Отримати параметри обробки
        params = self.get_processing_params(preset, model, enhancement)
        
        logging.info(f"🚀 Почату batch обробку {len(images)} зображень")
        logging.info(f"📋 Параметри: {preset}/{model}/{enhancement}")
        
        # Обробити зображення паралельно
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_image = {
                executor.submit(self.process_single_image, img, params, f"_{i:03d}"): img
                for i, img in enumerate(images)
            }
            
            for future in future_to_image:
                result = future.result()
                results.append(result)
                self.results.append(result)
        
        # Генерувати звіт
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful
        
        logging.info(f"📊 Batch обробка завершена: {successful} успішно, {failed} помилок")
        
        # Зберегти метадані
        if self.config["generate_metadata"]:
            self.save_metadata(results, preset, model, enhancement)
        
        return results
    
    def save_metadata(self, results: List[Dict[str, Any]], preset: str, 
                     model: str, enhancement: str):
        """Зберегти метадані обробки"""
        metadata = {
            "batch_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "preset": preset,
            "model": model,
            "enhancement": enhancement,
            "total_images": len(results),
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
            "total_time": sum(r["processing_time"] for r in results),
            "results": results
        }
        
        metadata_path = os.path.join(self.config["output_dir"], f"batch_metadata_{metadata['batch_id']}.json")
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logging.info(f"📄 Метадані збережено: {metadata_path}")
    
    def create_download_package(self, results: List[Dict[str, Any]], 
                               package_name: str = None) -> str:
        """Створити пакет для скачування"""
        if not package_name:
            package_name = f"enhanced_photos_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        package_dir = os.path.join(self.config["download_dir"], package_name)
        os.makedirs(package_dir, exist_ok=True)
        
        # Копіювати успішні результати
        copied = 0
        for result in results:
            if result["success"] and os.path.exists(result["output_path"]):
                dest_path = os.path.join(package_dir, os.path.basename(result["output_path"]))
                shutil.copy2(result["output_path"], dest_path)
                copied += 1
        
        # Створити архів
        archive_path = f"{package_dir}.zip"
        shutil.make_archive(package_dir.rstrip('.zip'), 'zip', package_dir)
        
        # Очистити тимчасову директорію
        shutil.rmtree(package_dir)
        
        logging.info(f"📦 Пакет створено: {archive_path} ({copied} файлів)")
        return archive_path

def main():
    parser = argparse.ArgumentParser(description="Автоматичний batch процесор фотографій")
    
    parser.add_argument("input", help="Шлях до зображення або директорії з зображеннями")
    parser.add_argument("--preset", default=None, help="Preset для обробки (professional_headshot, artistic_portrait, etc.)")
    parser.add_argument("--model", default=None, help="SDXL модель (epicrealism_xl, realvisxl_v5_lightning, etc.)")
    parser.add_argument("--enhancement", default=None, help="Рівень покращення (light, medium, strong, extreme)")
    parser.add_argument("--workers", type=int, default=None, help="Кількість паралельних процесів")
    parser.add_argument("--package", action="store_true", help="Створити ZIP пакет для скачування")
    parser.add_argument("--config", default="config/batch_config.yaml", help="Шлях до конфігураційного файлу")
    
    args = parser.parse_args()
    
    # Створити процесор
    processor = AutoBatchProcessor(args.config)
    
    # Запустити batch обробку
    results = processor.batch_process(
        args.input,
        preset=args.preset,
        model=args.model,
        enhancement=args.enhancement,
        max_workers=args.workers
    )
    
    # Створити пакет для скачування
    if args.package and results:
        processor.create_download_package(results)
    
    # Показати підсумок
    successful = sum(1 for r in results if r["success"])
    print(f"\n🎉 Обробка завершена: {successful}/{len(results)} успішно")
    
    if successful > 0:
        print(f"📁 Результати в: {processor.config['output_dir']}")
        if args.package:
            print(f"📦 ZIP пакет в: {processor.config['download_dir']}")

if __name__ == "__main__":
    main()