#!/usr/bin/env python3
"""
RunPod Pod Creator for SDXL Image Enhancement Pipeline
Automatically creates and configures RunPod GPU instance
"""

import os
import sys
import requests
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv(".env")

class RunPodCreator:
    def __init__(self):
        self.api_key = os.getenv("RUNPOD_API_KEY")
        self.endpoint = "https://api.runpod.io/graphql"
        
        # Updated configurations based on actual RunPod availability
        self.gpu_configs = {
            "A40": {
                "gpu_type": "NVIDIA A40",
                "recommended": True,
                "price_range": "$0.40/hr",
                "performance": "Excellent (48GB VRAM) 🌟"
            },
            "RTX6000Ada": {
                "gpu_type": "NVIDIA RTX 6000 Ada Generation",
                "recommended": True,
                "price_range": "$0.77/hr",
                "performance": "Excellent (48GB VRAM)"
            },
            "RTXA6000": {
                "gpu_type": "NVIDIA RTX A6000",
                "recommended": True,
                "price_range": "$0.49/hr",
                "performance": "Very Good (48GB VRAM)"
            },
            "RTX4090": {
                "gpu_type": "NVIDIA GeForce RTX 4090",
                "recommended": False,
                "price_range": "$0.69/hr",
                "performance": "Good (24GB VRAM limit)"
            }
        }
        
        # Docker images for different setups
        self.docker_images = {
            "pytorch": "runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04",
            "automatic1111": "runpod/stable-diffusion:web-ui-10.2.1",
            "custom": "your-custom-image"
        }

    def check_api_key(self):
        """Verify API key is configured"""
        if not self.api_key:
            print("❌ RunPod API key not found!")
            print("📝 Please set RUNPOD_API_KEY in your .env file")
            print("🔗 Get your API key from: https://runpod.io/console/user/settings")
            return False
        return True

    def get_headers(self):
        """Get request headers with API key"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def show_recommendations(self):
        """Show GPU and storage recommendations"""
        print("\n🎯 РЕКОМЕНДАЦІЇ ДЛЯ SDXL МОДЕЛЕЙ:")
        print("="*50)
        
        print("\n💻 GPU:")
        for name, config in self.gpu_configs.items():
            status = "✅ РЕКОМЕНДОВАНО" if config["recommended"] else "⚠️  ОБМЕЖЕНО"
            print(f"  {status} {name}")
            print(f"    💰 Ціна: {config['price_range']}")
            print(f"    ⚡ Продуктивність: {config['performance']}")
        
        print("\n💾 STORAGE:")
        print("  📁 Container Disk: 15-20GB (тільки для системи)")
        print("  🗄️  Network Volume: 40-60GB (РЕКОМЕНДОВАНО!)")
        print("     • Створіть Network Storage перед запуском")
        print("     • Постійне зберігання моделей")
        print("     • Менші вимоги до Container Disk")
        print("     • https://console.runpod.io/user/storage")
        print("\n🎯 ПОТОЧНА ДОСТУПНІСТЬ: HIGH для A40, RTX 6000 Ada!")
        print("💡 Network Storage вирішує проблеми з дисковим простором!")

    def get_user_config(self):
        """Interactive configuration"""
        print("\n⚙️  НАЛАШТУВАННЯ ПОДУ:")
        print("="*30)
        
        # GPU selection
        print("\n1. Оберіть GPU:")
        gpu_options = list(self.gpu_configs.keys())
        for i, gpu in enumerate(gpu_options, 1):
            status = "🌟" if self.gpu_configs[gpu]["recommended"] else "  "
            print(f"  {status} {i}. {gpu}")
        
        while True:
            try:
                gpu_choice = int(input(f"\nГPU (1-{len(gpu_options)}): ")) - 1
                selected_gpu = gpu_options[gpu_choice]
                break
            except (ValueError, IndexError):
                print("❌ Неправильний вибір, спробуйте ще раз")
        
        # Storage configuration
        print(f"\n2. Налаштування сховища:")
        print("   📁 Container Disk (мінімальний для системи)")
        container_disk = input("   Розмір (GB) або Enter для 15GB: ").strip()
        container_disk = int(container_disk) if container_disk else 15
        
        print("   🗄️  Network Volume ID (якщо створили)")
        network_volume = input("   Volume ID або Enter щоб пропустити: ").strip()
        network_volume = network_volume if network_volume else None
        
        # Cloud type selection
        print(f"\n3. Тип хмари:")
        print("   1. Secure Cloud (дорожче, стабільніше)")
        print("   2. Community Cloud (дешевше, може бути менш стабільно)")
        
        cloud_choice = input("   Оберіть (1-2) або Enter для Community: ").strip()
        cloud_type = "SECURE" if cloud_choice == "1" else "COMMUNITY"
        
        # Docker image
        print(f"\n4. Docker образ:")
        print("   1. PyTorch (рекомендовано для цього проекту)")
        print("   2. Automatic1111 (готовий WebUI)")
        
        docker_choice = input("   Оберіть (1-2) або Enter для PyTorch: ").strip()
        if docker_choice == "2":
            docker_image = self.docker_images["automatic1111"]
        else:
            docker_image = self.docker_images["pytorch"]
        
        return {
            "gpu_type": selected_gpu,
            "container_disk": container_disk,
            "network_volume": network_volume,
            "cloud_type": cloud_type,
            "docker_image": docker_image
        }

    def create_pod(self, config):
        """Create RunPod instance"""
        print(f"\n🚀 Створення поду...")
        print(f"   GPU: {config['gpu_type']}")
        print(f"   Container: {config['container_disk']}GB")
        print(f"   Cloud: {config.get('cloud_type', 'COMMUNITY')}")
        if config.get('network_volume'):
            print(f"   Network Volume: {config['network_volume']}")
        else:
            print(f"   Network Volume: None")
        
        # GraphQL mutation for pod creation
        query = """
        mutation PodFindAndDeployOnDemand($input: PodFindAndDeployOnDemandInput!) {
          podFindAndDeployOnDemand(input: $input) {
            id
            imageName
            env
            machineId
            machine {
              podHostId
              gpuDisplayName
            }
          }
        }
        """
        
        variables = {
            "input": {
                "cloudType": config.get('cloud_type', 'COMMUNITY'),
                "gpuCount": 1,
                "gpuTypeId": config['gpu_type'],
                "containerDiskInGb": config['container_disk'],
                "name": "sdxl-enhancement-pipeline",
                "imageName": config['docker_image'],
                "ports": "3000/http,8888/http",
                "env": [
                    {"key": "RUNPOD_PROJECT", "value": "I-Model"},
                    {"key": "JUPYTER_ENABLE", "value": "1"}
                ]
            }
        }
        
        # Add network volume if provided
        if config.get('network_volume'):
            variables["input"]["dataCenterId"] = "EU-RO-1"  # Europe region for i-model-storage
            variables["input"]["networkVolumeId"] = config['network_volume']
        
        try:
            response = requests.post(
                self.endpoint,
                headers=self.get_headers(),
                json={"query": query, "variables": variables},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if "errors" in result:
                    print("❌ Помилка створення поду:")
                    for error in result["errors"]:
                        print(f"   {error['message']}")
                    return None
                
                pod_data = result["data"]["podFindAndDeployOnDemand"]
                print("✅ Под успішно створено!")
                return pod_data
            else:
                print(f"❌ HTTP помилка: {response.status_code}")
                print(response.text)
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Помилка з'єднання: {e}")
            return None

    def show_next_steps(self, pod_data):
        """Show what to do next"""
        pod_id = pod_data["id"]
        
        print(f"\n🎉 УСПІХ! Под створено!")
        print("="*40)
        print(f"🆔 Pod ID: {pod_id}")
        print(f"💻 GPU: {pod_data.get('machine', {}).get('gpuDisplayName', 'N/A')}")
        
        print(f"\n📋 НАСТУПНІ КРОКИ:")
        print("1. Зайдіть в RunPod Console: https://runpod.io/console/pods")
        print("2. Зачекайте поки под запуститься (2-5 хвилин)")
        print("3. Відкрийте Terminal або Jupyter")
        print("4. Запустіть команди:")
        print()
        print("   # Клонувати проект")
        print("   git clone https://github.com/your-repo/I-Model.git")
        print("   cd I-Model")
        print()
        print("   # Встановити залежності") 
        print("   pip install -r requirements.txt")
        print()
        print("   # Завантажити моделі")
        print("   python utils/runpod_launcher.py --models epicrealism_xl")
        print()
        print("🌐 WebUI буде доступний на порту 3000")
        print("📱 URL: https://[pod-id]-3000.proxy.runpod.net")

def main():
    """Main function"""
    print("🌥️  RUNPOD CREATOR для I, Model")
    print("="*50)
    print("Автоматичне створення GPU поду для SDXL моделей")
    
    creator = RunPodCreator()
    
    # Check API key
    if not creator.check_api_key():
        sys.exit(1)
    
    # Show recommendations
    creator.show_recommendations()
    
    # Get user configuration
    config = creator.get_user_config()
    
    # Confirm creation
    print(f"\n❓ Створити под з цими налаштуваннями? (y/N)")
    confirm = input().strip().lower()
    
    if confirm in ['y', 'yes', 'так', 'да']:
        pod_data = creator.create_pod(config)
        if pod_data:
            creator.show_next_steps(pod_data)
    else:
        print("❌ Створення скасовано")

if __name__ == "__main__":
    main()
