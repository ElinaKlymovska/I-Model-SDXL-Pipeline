#!/usr/bin/env python3
"""
Simple SSH Deployment Script for MyNeuralKingdom
Використовує прямий SSH доступ для деплойменту
"""

import os
import sys
import subprocess
import tarfile
from pathlib import Path

# SSH connection details
SSH_HOST = "194.68.245.104"
SSH_PORT = "22160"
SSH_USER = "root"
SSH_KEY = "~/.ssh/id_ed25519"

def run_ssh_command(command, show_output=True):
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

def upload_file(local_path, remote_path):
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

def create_project_archive():
    """Створити архів проєкту"""
    print("📦 Creating project archive...")
    
    archive_path = "myneuralkingdom.tar.gz"
    
    # Files and directories to include
    include_items = [
        "core/",
        "services/", 
        "pipelines/",
        "configs/",
        "runpod/",
        "*.py",
        "*.txt",
        "*.json",
        "*.md"
    ]
    
    # Create tar archive
    with tarfile.open(archive_path, "w:gz") as tar:
        for item in include_items:
            if "*" in item:
                # Handle wildcards
                import glob
                for file_path in glob.glob(item):
                    if os.path.isfile(file_path):
                        tar.add(file_path)
            else:
                if os.path.exists(item):
                    tar.add(item)
    
    print(f"✅ Archive created: {archive_path}")
    return archive_path

def main():
    """Головна функція деплойменту"""
    print("🚀 MyNeuralKingdom SSH Deployment")
    print("=" * 40)
    
    # Test SSH connection
    print("🔍 Testing SSH connection...")
    success, output, error = run_ssh_command("echo 'SSH connection successful!'")
    
    if not success:
        print("❌ SSH connection failed!")
        print(f"Error: {error}")
        print("\nПеревірте:")
        print("1. SSH ключ доступний: ~/.ssh/id_ed25519")
        print("2. RunPod все ще запущений")
        print("3. IP адреса актуальна")
        return False
    
    print("✅ SSH connection successful!")
    
    # Check GPU
    print("\n🔍 Checking GPU...")
    run_ssh_command("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    
    # Create project archive
    archive_path = create_project_archive()
    
    # Upload project
    print("\n📤 Uploading project...")
    if not upload_file(archive_path, "/workspace/myneuralkingdom.tar.gz"):
        return False
    
    # Extract project
    print("\n📂 Extracting project on RunPod...")
    run_ssh_command("cd /workspace && tar -xzf myneuralkingdom.tar.gz")
    run_ssh_command("cd /workspace && ls -la")
    
    # Setup environment
    print("\n🔧 Setting up environment...")
    setup_commands = [
        "cd /workspace",
        "export TRANSFORMERS_CACHE=/runpod-volume/cache",
        "export HF_HOME=/runpod-volume/cache",
        "export TORCH_HOME=/runpod-volume/cache",
        "mkdir -p /runpod-volume/{models,cache,outputs,data,temp}",
        "pip install --upgrade pip",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "pip install diffusers[torch] transformers accelerate xformers",
        "pip install mediapipe opencv-python pillow safetensors",
        "pip install insightface onnxruntime-gpu",
        "pip install tqdm requests numpy psutil jupyter matplotlib"
    ]
    
    for cmd in setup_commands:
        success, _, _ = run_ssh_command(cmd)
        if not success and "pip install" in cmd:
            print(f"⚠️ Command may have warnings: {cmd}")
    
    # Test installation
    print("\n🧪 Testing installation...")
    run_ssh_command("cd /workspace && python -c \"import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'\"")
    
    # Download models (in background)
    print("\n🤖 Starting model download (this will take 15-30 minutes)...")
    run_ssh_command("cd /workspace && nohup python download_models.py > model_download.log 2>&1 &")
    
    print("\n🎉 Deployment completed!")
    print("\n📋 Next steps:")
    print("1. Monitor model download: ssh root@194.68.245.104 -p 22160 -i ~/.ssh/id_ed25519")
    print("   then: tail -f /workspace/model_download.log")
    print("2. Test system: python runpod_test.py")
    print("3. Start Jupyter: jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root")
    print("4. Access Jupyter: https://60981-7rrr0s51geqwi3.proxy.runpod.net")
    
    # Cleanup
    os.remove(archive_path)
    print(f"\n🧹 Cleaned up local archive: {archive_path}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)