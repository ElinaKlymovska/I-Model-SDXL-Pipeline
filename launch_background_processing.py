#!/usr/bin/env python3
"""
Background Processing Launcher
Локальний скрипт для запуску фонової обробки зображень на RunPod та скачування результатів
"""

import os
import sys
import time
import json
import tarfile
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUTS_DIR = DATA_DIR / "outputs"

# SSH connection details (from config)
sys.path.append(str(PROJECT_ROOT))
from runpod.config import config

SSH_HOST = config.ssh_host
SSH_PORT = config.ssh_port
SSH_USER = "root"
SSH_KEY = "~/.ssh/id_ed25519"


class BackgroundProcessingLauncher:
    """Менеджер для запуску фонової обробки зображень"""
    
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.local_session_dir = OUTPUTS_DIR / f"session_{self.session_id}"
        self.remote_session_dir = f"/workspace/data/outputs/session_{self.session_id}"
        self.processing_log = self.local_session_dir / "processing.log"
        
        # Створити локальні папки
        self.local_session_dir.mkdir(parents=True, exist_ok=True)
        OUTPUTS_DIR.mkdir(exist_ok=True)
        
        print(f"🎯 Нова сесія обробки: {self.session_id}")
        print(f"📁 Локальна папка: {self.local_session_dir}")

    def run_ssh_command(self, command: str, show_output: bool = True) -> tuple[bool, str, str]:
        """Виконати команду через SSH"""
        ssh_cmd = f"ssh {SSH_USER}@{SSH_HOST} -p {SSH_PORT} -i {SSH_KEY} '{command}'"
        
        if show_output:
            print(f"🔧 Executing: {command}")
        
        try:
            result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)
            
            if show_output and result.stdout:
                print(result.stdout)
            if result.stderr:
                print(f"⚠️ stderr: {result.stderr}")
                
            return result.returncode == 0, result.stdout, result.stderr
        except Exception as e:
            print(f"❌ SSH command failed: {e}")
            return False, "", str(e)

    def upload_file(self, local_path: str, remote_path: str) -> bool:
        """Завантажити файл через SCP"""
        scp_cmd = f"scp -P {SSH_PORT} -i {SSH_KEY} {local_path} {SSH_USER}@{SSH_HOST}:{remote_path}"
        
        print(f"📤 Uploading {local_path} -> {remote_path}")
        
        try:
            result = subprocess.run(scp_cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Uploaded {local_path}")
                return True
            else:
                print(f"❌ Upload failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Upload error: {e}")
            return False

    def download_directory(self, remote_path: str, local_path: str) -> bool:
        """Скачати папку через SCP"""
        scp_cmd = f"scp -r -P {SSH_PORT} -i {SSH_KEY} {SSH_USER}@{SSH_HOST}:{remote_path} {local_path}"
        
        print(f"📥 Downloading {remote_path} -> {local_path}")
        
        try:
            result = subprocess.run(scp_cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Downloaded to {local_path}")
                return True
            else:
                print(f"❌ Download failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Download error: {e}")
            return False

    def check_connection(self) -> bool:
        """Перевірити SSH підключення"""
        print("🔍 Перевірка підключення до RunPod...")
        success, _, _ = self.run_ssh_command("echo 'Connection OK'", show_output=False)
        
        if success:
            print("✅ Підключення успішне")
            return True
        else:
            print("❌ Не вдалося підключитися до RunPod")
            print("Перевірте чи запущений pod та чи правильні SSH реквізити")
            return False

    def upload_input_images(self) -> bool:
        """Завантажити вхідні зображення на RunPod"""
        print("📤 Завантаження вхідних зображень...")
        
        if not INPUT_DIR.exists() or not list(INPUT_DIR.glob("*")):
            print(f"❌ Не знайдено зображень у {INPUT_DIR}")
            return False
        
        # Створити архів з зображеннями
        archive_path = f"input_images_{self.session_id}.tar.gz"
        
        with tarfile.open(archive_path, "w:gz") as tar:
            for image_file in INPUT_DIR.glob("*"):
                if image_file.is_file() and image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff']:
                    tar.add(image_file, arcname=image_file.name)
        
        # Завантажити архів
        if not self.upload_file(archive_path, f"/workspace/{archive_path}"):
            return False
        
        # Розпакувати на RunPod
        success, _, _ = self.run_ssh_command(f"cd /workspace && mkdir -p data/input && tar --no-same-owner -xzf {archive_path} -C data/input/")
        
        # Видалити локальний архів
        os.remove(archive_path)
        
        if success:
            print("✅ Зображення успішно завантажені")
            return True
        else:
            print("❌ Помилка розпакування зображень")
            return False

    def upload_scripts(self) -> bool:
        """Завантажити скрипти на RunPod"""
        print("📤 Завантаження скриптів...")
        
        # Створити архів з проектом
        archive_path = f"myneuralkingdom_{self.session_id}.tar.gz"
        
        include_items = [
            "core/",
            "services/", 
            "pipelines/",
            "scripts/",
            "configs/",
            "runpod/",
            "*.py",
            "*.txt",
            "*.json"
        ]
        
        with tarfile.open(archive_path, "w:gz") as tar:
            for item in include_items:
                if "*" in item:
                    import glob
                    for file_path in glob.glob(item):
                        if os.path.isfile(file_path):
                            tar.add(file_path)
                else:
                    if os.path.exists(item):
                        tar.add(item)
        
        # Завантажити та розпакувати
        if not self.upload_file(archive_path, f"/workspace/{archive_path}"):
            return False
        
        success, _, _ = self.run_ssh_command(f"cd /workspace && tar --no-same-owner -xzf {archive_path}")
        
        # Видалити локальний архів
        os.remove(archive_path)
        
        if success:
            print("✅ Скрипти успішно завантажені")
            return True
        else:
            print("❌ Помилка завантаження скриптів")
            return False

    def start_background_processing(self, model_name: str = "epicrealism_xl") -> bool:
        """Запустити фонову обробку на RunPod"""
        print(f"🚀 Запуск фонової обробки з моделлю: {model_name}")
        
        # Створити команду для фонового запуску
        cmd = f"cd /workspace && nohup python scripts/background_image_processor.py --model {model_name} --session-id {self.session_id} > background_processing_{self.session_id}.log 2>&1 &"
        
        success, output, _ = self.run_ssh_command(cmd)
        
        if success:
            print("✅ Фонова обробка запущена")
            print(f"📋 Session ID: {self.session_id}")
            print("🔍 Для моніторингу використовуйте:")
            print(f"   ssh {SSH_USER}@{SSH_HOST} -p {SSH_PORT} -i {SSH_KEY}")
            print(f"   tail -f /workspace/background_processing_{self.session_id}.log")
            return True
        else:
            print("❌ Помилка запуску фонової обробки")
            return False

    def monitor_processing(self, timeout_minutes: int = 60) -> bool:
        """Моніторити процес обробки"""
        print(f"👀 Моніторинг обробки (максимум {timeout_minutes} хвилин)...")
        
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        while time.time() - start_time < timeout_seconds:
            # Перевірити чи існує файл статистики
            success, output, _ = self.run_ssh_command(
                f"test -f {self.remote_session_dir}/processing_stats.json && echo 'EXISTS' || echo 'NOT_EXISTS'",
                show_output=False
            )
            
            if "EXISTS" in output:
                # Перевірити чи завершилася обробка
                success, stats_content, _ = self.run_ssh_command(
                    f"cat {self.remote_session_dir}/processing_stats.json",
                    show_output=False
                )
                
                if success and stats_content:
                    try:
                        stats = json.loads(stats_content)
                        if "end_time" in stats:
                            print("✅ Обробка завершена!")
                            return True
                        else:
                            processed = stats.get("processed_images", 0)
                            total = stats.get("total_images", 0)
                            print(f"📊 Прогрес: {processed}/{total} зображень")
                    except json.JSONDecodeError:
                        pass
            
            print("⏳ Очікування... (натисніть Ctrl+C для припинення моніторингу)")
            time.sleep(30)  # Перевіряти кожні 30 секунд
        
        print("⏰ Тайм-аут моніторингу. Обробка може все ще працювати.")
        return False

    def download_results(self) -> bool:
        """Скачати результати обробки"""
        print("📥 Скачування результатів...")
        
        # Перевірити чи існує папка з результатами
        success, _, _ = self.run_ssh_command(f"test -d {self.remote_session_dir} && echo 'OK'", show_output=False)
        
        if not success:
            print(f"❌ Папка з результатами не існує: {self.remote_session_dir}")
            return False
        
        # Скачати папку з результатами
        parent_dir = self.local_session_dir.parent
        return self.download_directory(self.remote_session_dir, str(parent_dir))

    def show_results_summary(self) -> None:
        """Показати підсумок результатів"""
        stats_file = self.local_session_dir / "processing_stats.json"
        report_file = self.local_session_dir / "processing_report.txt"
        
        print("\n🎯 ПІДСУМОК ОБРОБКИ")
        print("=" * 40)
        
        if stats_file.exists():
            try:
                with open(stats_file, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                
                print(f"📋 Session ID: {stats['session_id']}")
                print(f"📊 Всього зображень: {stats['total_images']}")
                print(f"✅ Успішно оброблено: {stats['processed_images']}")
                print(f"❌ Помилки: {stats['failed_images']}")
                print(f"🤖 Модель: {stats['model_name']}")
                print(f"⏰ Час обробки: {stats.get('processing_time', 0):.1f} секунд")
                print(f"📁 Локальні результати: {self.local_session_dir}")
                
            except Exception as e:
                print(f"❌ Помилка читання статистики: {e}")
        
        if report_file.exists():
            print(f"\n📋 Детальний звіт: {report_file}")

    def cleanup_remote(self) -> None:
        """Очистити тимчасові файли на RunPod"""
        print("🧹 Очищення тимчасових файлів...")
        
        cleanup_commands = [
            f"rm -f /workspace/*_{self.session_id}.tar.gz",
            f"rm -f /workspace/background_processing_{self.session_id}.log"
        ]
        
        for cmd in cleanup_commands:
            self.run_ssh_command(cmd, show_output=False)
        
        print("✅ Очищення завершено")

    def run_full_processing(self, model_name: str = "epicrealism_xl", monitor: bool = True) -> bool:
        """Запустити повний цикл обробки"""
        print("🚀 ЗАПУСК ФОНОВОЇ ОБРОБКИ ЗОБРАЖЕНЬ")
        print("=" * 50)
        
        # 1. Перевірити підключення
        if not self.check_connection():
            return False
        
        # 2. Завантажити зображення та скрипти
        if not self.upload_input_images():
            return False
        
        if not self.upload_scripts():
            return False
        
        # 3. Запустити обробку
        if not self.start_background_processing(model_name):
            return False
        
        # 4. Моніторити (опційно)
        if monitor:
            self.monitor_processing()
        
        # 5. Скачати результати
        if not self.download_results():
            print("❌ Не вдалося скачати результати автоматично")
            print("Ви можете скачати їх пізніше за допомогою:")
            print(f"scp -r -P {SSH_PORT} -i {SSH_KEY} {SSH_USER}@{SSH_HOST}:{self.remote_session_dir} {OUTPUTS_DIR}/")
            return False
        
        # 6. Показати підсумок
        self.show_results_summary()
        
        # 7. Очистити тимчасові файли
        self.cleanup_remote()
        
        print("\n🎉 Фонова обробка завершена успішно!")
        return True


def main():
    """Головна функція"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Background Processing Launcher")
    parser.add_argument("--model", default="epicrealism_xl", help="Model name to use")
    parser.add_argument("--no-monitor", action="store_true", help="Don't monitor processing")
    parser.add_argument("--download-only", help="Only download results for specific session ID")
    
    args = parser.parse_args()
    
    if args.download_only:
        # Скачати результати для конкретної сесії
        launcher = BackgroundProcessingLauncher()
        launcher.session_id = args.download_only
        launcher.local_session_dir = OUTPUTS_DIR / f"session_{args.download_only}"
        launcher.remote_session_dir = f"/workspace/data/outputs/session_{args.download_only}"
        launcher.local_session_dir.mkdir(parents=True, exist_ok=True)
        
        if launcher.download_results():
            launcher.show_results_summary()
            return 0
        else:
            return 1
    else:
        # Повний цикл обробки
        launcher = BackgroundProcessingLauncher()
        
        if launcher.run_full_processing(
            model_name=args.model,
            monitor=not args.no_monitor
        ):
            return 0
        else:
            return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Обробка перервана користувачем")
        print("Фонова обробка може все ще працювати на RunPod")
        sys.exit(1)