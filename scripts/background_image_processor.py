#!/usr/bin/env python3
"""
Background Image Processor for RunPod
Запускається на RunPod і обробляє всі фотографії у фоновому режимі
"""

import os
import sys
import json
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from pipelines.inference.image_enhancement import ImageEnhancementPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/workspace/background_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BackgroundImageProcessor:
    """Фоновий процесор зображень для RunPod"""
    
    def __init__(self, session_id: str = None):
        """
        Ініціалізація процесора
        
        Args:
            session_id: Унікальний ідентифікатор сесії
        """
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.pipeline = None
        self.start_time = time.time()
        
        # Шляхи
        self.input_dir = Path("/workspace/data/input")
        self.base_output_dir = Path("/workspace/data/outputs")
        self.session_output_dir = self.base_output_dir / f"session_{self.session_id}"
        
        # Створимо папки
        self.session_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Статистика
        self.stats = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "total_images": 0,
            "processed_images": 0,
            "failed_images": 0,
            "processing_time": 0,
            "model_name": None,
            "results": []
        }
        
        logger.info(f"🚀 Ініціалізовано BackgroundImageProcessor для сесії: {self.session_id}")
        logger.info(f"📁 Вихідна папка: {self.session_output_dir}")

    def find_input_images(self) -> List[Path]:
        """Знайти всі зображення для обробки"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff']
        images = []
        
        if not self.input_dir.exists():
            logger.warning(f"❌ Папка з вхідними зображеннями не існує: {self.input_dir}")
            return images
        
        for ext in image_extensions:
            images.extend(self.input_dir.glob(f"*{ext}"))
            images.extend(self.input_dir.glob(f"*{ext.upper()}"))
        
        logger.info(f"📂 Знайдено {len(images)} зображень для обробки")
        return sorted(images)

    def setup_pipeline(self, model_name: str = "epicrealism_xl") -> bool:
        """Налаштувати pipeline для обробки"""
        try:
            logger.info("🔧 Налаштування pipeline...")
            
            self.pipeline = ImageEnhancementPipeline()
            
            if not self.pipeline.setup():
                logger.error("❌ Не вдалося налаштувати pipeline")
                return False
            
            logger.info(f"📦 Завантаження моделі: {model_name}")
            if not self.pipeline.load_model(model_name):
                logger.error(f"❌ Не вдалося завантажити модель: {model_name}")
                return False
            
            self.stats["model_name"] = model_name
            logger.info("✅ Pipeline успішно налаштовано")
            return True
            
        except Exception as e:
            logger.error(f"❌ Помилка налаштування pipeline: {e}")
            logger.error(traceback.format_exc())
            return False

    def process_single_image(self, image_path: Path, index: int, total: int) -> Dict[str, Any]:
        """Обробити одне зображення"""
        result = {
            "input_path": str(image_path),
            "output_path": None,
            "status": "failed",
            "error": None,
            "processing_time": 0,
            "timestamp": datetime.now().isoformat()
        }
        
        start_time = time.time()
        
        try:
            logger.info(f"🖼️ Обробка [{index+1}/{total}]: {image_path.name}")
            
            # Обробка зображення
            output_subdir = self.session_output_dir / f"image_{index+1:02d}_{image_path.stem}"
            output_subdir.mkdir(exist_ok=True)
            
            # Конфігурація для обробки
            enhancement_config = {
                "strength": 0.3,
                "guidance_scale": 7.0,
                "num_inference_steps": 20,
                "face_enhancement_strength": 0.4
            }
            
            # Запуск обробки
            processing_result = self.pipeline.run(
                input_data=str(image_path),
                model_name=self.stats["model_name"],
                enhancement_config=enhancement_config,
                output_dir=str(output_subdir)
            )
            
            if processing_result["status"] == "success":
                result["status"] = "success"
                result["output_path"] = str(output_subdir)
                result["enhanced_images"] = processing_result.get("enhanced_images", [])
                self.stats["processed_images"] += 1
                logger.info(f"✅ Успішно оброблено: {image_path.name}")
            else:
                result["error"] = processing_result.get("error", "Unknown error")
                self.stats["failed_images"] += 1
                logger.error(f"❌ Помилка обробки {image_path.name}: {result['error']}")
                
        except Exception as e:
            result["error"] = str(e)
            self.stats["failed_images"] += 1
            logger.error(f"❌ Виняток при обробці {image_path.name}: {e}")
            logger.error(traceback.format_exc())
        
        result["processing_time"] = time.time() - start_time
        return result

    def process_all_images(self, model_name: str = "epicrealism_xl") -> Dict[str, Any]:
        """Обробити всі зображення"""
        logger.info("🚀 Початок фонової обробки зображень")
        
        # Знайти всі зображення
        images = self.find_input_images()
        if not images:
            return {"status": "error", "message": "Не знайдено зображень для обробки"}
        
        self.stats["total_images"] = len(images)
        
        # Налаштувати pipeline
        if not self.setup_pipeline(model_name):
            return {"status": "error", "message": "Не вдалося налаштувати pipeline"}
        
        # Обробити кожне зображення
        for i, image_path in enumerate(images):
            result = self.process_single_image(image_path, i, len(images))
            self.stats["results"].append(result)
            
            # Зберегти проміжні статистики
            self.save_stats()
        
        # Фінальна статистика
        self.stats["end_time"] = datetime.now().isoformat()
        self.stats["processing_time"] = time.time() - self.start_time
        self.save_stats()
        
        # Очистити pipeline
        if self.pipeline:
            self.pipeline.cleanup()
        
        logger.info(f"🎉 Завершено обробку! Успішно: {self.stats['processed_images']}, Помилки: {self.stats['failed_images']}")
        
        return {
            "status": "success",
            "session_id": self.session_id,
            "stats": self.stats,
            "output_dir": str(self.session_output_dir)
        }

    def save_stats(self):
        """Зберегти статистику"""
        stats_file = self.session_output_dir / "processing_stats.json"
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"❌ Помилка збереження статистики: {e}")

    def create_summary_report(self) -> str:
        """Створити підсумковий звіт"""
        report = f"""
🎯 ЗВІТ ПРО ОБРОБКУ ЗОБРАЖЕНЬ
Session ID: {self.session_id}
================================

📊 Статистика:
• Всього зображень: {self.stats['total_images']}
• Успішно оброблено: {self.stats['processed_images']}
• Помилки: {self.stats['failed_images']}
• Модель: {self.stats['model_name']}
• Час обробки: {self.stats.get('processing_time', 0):.1f} секунд

📁 Результати збережено в: {self.session_output_dir}

🖼️ Деталі обробки:
"""
        
        for i, result in enumerate(self.stats['results'], 1):
            status_icon = "✅" if result['status'] == 'success' else "❌"
            report += f"{status_icon} {i:2d}. {Path(result['input_path']).name} ({result['processing_time']:.1f}s)\n"
            if result['error']:
                report += f"     Помилка: {result['error']}\n"
        
        return report


def main():
    """Головна функція"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Background Image Processor")
    parser.add_argument("--model", default="epicrealism_xl", help="Model name to use")
    parser.add_argument("--session-id", help="Custom session ID")
    args = parser.parse_args()
    
    # Створити процесор
    processor = BackgroundImageProcessor(session_id=args.session_id)
    
    # Запустити обробку
    result = processor.process_all_images(model_name=args.model)
    
    if result["status"] == "success":
        # Створити звіт
        report = processor.create_summary_report()
        logger.info(report)
        
        # Зберегти звіт у файл
        report_file = processor.session_output_dir / "processing_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n🎉 Обробка завершена успішно!")
        print(f"📁 Результати: {processor.session_output_dir}")
        print(f"📋 Session ID: {processor.session_id}")
        
        return 0
    else:
        logger.error(f"❌ Обробка не вдалася: {result.get('message')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())