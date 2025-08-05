#!/usr/bin/env python3
"""
Test RunPod Connection
Швидка перевірка підключення до RunPod pod
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from runpod.manager import RunPodManager
from runpod.config import config

def test_connection():
    """Тест підключення до RunPod"""
    print("🔍 Тестування підключення до RunPod...")
    print(f"Pod ID: {config.pod_id}")
    
    if not config.api_key:
        print("❌ API ключ не знайдено! Оновіть .env файл.")
        return False
    
    print("✅ API ключ знайдено")
    
    # Initialize manager
    manager = RunPodManager()
    
    # Get pod info
    print("📊 Отримання інформації про pod...")
    pod_info = manager.get_pod_info()
    
    if not pod_info:
        print("❌ Не вдалося отримати інформацію про pod")
        return False
    
    print(f"✅ Pod знайдено: {pod_info.get('name')}")
    print(f"📈 Статус: {pod_info.get('desiredStatus')}")
    
    # Get connection details
    conn_info = manager.get_connection_info()
    print(f"🌐 IP: {conn_info.get('ip')}")
    print(f"🔌 SSH Port: {conn_info.get('ports', {}).get('ssh', 'N/A')}")
    print(f"📓 Jupyter: {conn_info.get('ports', {}).get('jupyter_url', 'N/A')}")
    
    if pod_info.get('desiredStatus') == 'RUNNING':
        print("🎉 Pod готовий до деплойменту!")
        return True
    else:
        print(f"⚠️ Pod не в стані RUNNING: {pod_info.get('desiredStatus')}")
        return False

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)