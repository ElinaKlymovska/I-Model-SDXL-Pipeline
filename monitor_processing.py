#!/usr/bin/env python3
"""
Processing Monitor
Швидкий скрипт для моніторингу фонової обробки на RunPod
"""

import sys
import json
import time
import subprocess
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))
from runpod.config import config

SSH_HOST = config.ssh_host
SSH_PORT = config.ssh_port
SSH_USER = "root"
SSH_KEY = "~/.ssh/id_ed25519"


def run_ssh_command(command: str) -> tuple[bool, str, str]:
    """Виконати команду через SSH"""
    ssh_cmd = f"ssh {SSH_USER}@{SSH_HOST} -p {SSH_PORT} -i {SSH_KEY} '{command}'"
    
    try:
        result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def list_active_sessions():
    """Показати активні сесії обробки"""
    print("🔍 Пошук активних сесій обробки...")
    
    success, output, _ = run_ssh_command("ls -la /workspace/data/outputs/ 2>/dev/null | grep session_ || echo 'No sessions found'")
    
    if success and "session_" in output:
        print("📋 Знайдені сесії:")
        for line in output.strip().split('\n'):
            if "session_" in line:
                parts = line.split()
                if len(parts) >= 9:
                    session_name = parts[-1]
                    print(f"  • {session_name}")
    else:
        print("❌ Активні сесії не знайдені")


def monitor_session(session_id: str):
    """Моніторити конкретну сесію"""
    print(f"👀 Моніторинг сесії: {session_id}")
    print("Натисніть Ctrl+C для виходу")
    print("-" * 50)
    
    remote_session_dir = f"/workspace/data/outputs/session_{session_id}"
    stats_file = f"{remote_session_dir}/processing_stats.json"
    log_file = f"/workspace/background_processing_{session_id}.log"
    
    try:
        while True:
            # Перевірити статистику
            success, stats_content, _ = run_ssh_command(f"cat {stats_file} 2>/dev/null || echo '{{}}'")
            
            if success and stats_content.strip():
                try:
                    stats = json.loads(stats_content)
                    
                    total = stats.get("total_images", 0)
                    processed = stats.get("processed_images", 0)
                    failed = stats.get("failed_images", 0)
                    model = stats.get("model_name", "Unknown")
                    
                    print(f"📊 Прогрес: {processed}/{total} зображень | Помилки: {failed} | Модель: {model}")
                    
                    if "end_time" in stats:
                        processing_time = stats.get("processing_time", 0)
                        print(f"✅ Обробка завершена за {processing_time:.1f} секунд")
                        
                        # Показати останні рядки логу
                        success, log_content, _ = run_ssh_command(f"tail -n 10 {log_file} 2>/dev/null || echo 'No log available'")
                        if success and log_content.strip():
                            print("\n📋 Останні записи логу:")
                            print(log_content)
                        
                        break
                        
                except json.JSONDecodeError:
                    print("❌ Помилка парсингу статистики")
            else:
                print("⏳ Очікування початку обробки...")
            
            time.sleep(10)  # Оновлювати кожні 10 секунд
            
    except KeyboardInterrupt:
        print("\n\n👋 Моніторинг зупинено")


def show_session_details(session_id: str):
    """Показати детальну інформацію про сесію"""
    print(f"📋 Деталі сесії: {session_id}")
    print("-" * 40)
    
    remote_session_dir = f"/workspace/data/outputs/session_{session_id}"
    stats_file = f"{remote_session_dir}/processing_stats.json"
    report_file = f"{remote_session_dir}/processing_report.txt"
    
    # Статистика
    success, stats_content, _ = run_ssh_command(f"cat {stats_file} 2>/dev/null")
    if success and stats_content.strip():
        try:
            stats = json.loads(stats_content)
            
            print(f"🎯 Session ID: {stats['session_id']}")
            print(f"🤖 Модель: {stats.get('model_name', 'Unknown')}")
            print(f"📊 Всього зображень: {stats.get('total_images', 0)}")
            print(f"✅ Оброблено: {stats.get('processed_images', 0)}")
            print(f"❌ Помилки: {stats.get('failed_images', 0)}")
            
            if "start_time" in stats:
                print(f"🕐 Початок: {stats['start_time']}")
            if "end_time" in stats:
                print(f"🕐 Завершення: {stats['end_time']}")
                print(f"⏰ Час обробки: {stats.get('processing_time', 0):.1f} секунд")
            else:
                print("⏳ Статус: В процесі")
                
        except json.JSONDecodeError:
            print("❌ Помилка читання статистики")
    else:
        print("❌ Статистика недоступна")
    
    # Звіт
    success, report_content, _ = run_ssh_command(f"cat {report_file} 2>/dev/null")
    if success and report_content.strip():
        print(f"\n📋 Звіт доступний ({len(report_content)} символів)")
        print("Перші 500 символів:")
        print("-" * 30)
        print(report_content[:500])
        if len(report_content) > 500:
            print("...")
    
    # Список файлів результатів
    success, files_content, _ = run_ssh_command(f"find {remote_session_dir} -name '*.jpg' -o -name '*.png' -o -name '*.webp' 2>/dev/null | wc -l")
    if success and files_content.strip():
        file_count = files_content.strip()
        print(f"\n🖼️ Згенеровано файлів зображень: {file_count}")


def show_logs(session_id: str):
    """Показати логи сесії"""
    log_file = f"/workspace/background_processing_{session_id}.log"
    
    print(f"📋 Логи сесії: {session_id}")
    print("-" * 40)
    
    success, log_content, _ = run_ssh_command(f"tail -n 50 {log_file} 2>/dev/null || echo 'Log file not found'")
    
    if success and log_content.strip():
        print(log_content)
    else:
        print("❌ Логи недоступні")


def main():
    """Головна функція"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Processing Monitor")
    parser.add_argument("--list", action="store_true", help="List active sessions")
    parser.add_argument("--monitor", help="Monitor specific session")
    parser.add_argument("--details", help="Show session details")
    parser.add_argument("--logs", help="Show session logs")
    parser.add_argument("--ssh", action="store_true", help="Open SSH connection")
    
    args = parser.parse_args()
    
    if args.ssh:
        print("🔗 Відкриття SSH підключення...")
        os.system(f"ssh {SSH_USER}@{SSH_HOST} -p {SSH_PORT} -i {SSH_KEY}")
    elif args.list:
        list_active_sessions()
    elif args.monitor:
        monitor_session(args.monitor)
    elif args.details:
        show_session_details(args.details)
    elif args.logs:
        show_logs(args.logs)
    else:
        print("🖥️ Processing Monitor")
        print("=" * 30)
        print("Доступні команди:")
        print("  --list                     Показати активні сесії")
        print("  --monitor <session_id>     Моніторити сесію")
        print("  --details <session_id>     Деталі сесії")
        print("  --logs <session_id>        Показати логи")
        print("  --ssh                      Відкрити SSH підключення")
        print()
        print("Приклад:")
        print("  python monitor_processing.py --list")
        print("  python monitor_processing.py --monitor 20241201_143022")


if __name__ == "__main__":
    import os
    main()