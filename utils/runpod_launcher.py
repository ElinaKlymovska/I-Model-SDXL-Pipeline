import os
import subprocess
import requests
import yaml
import logging
import argparse

def check_gpu_compatibility():
    """Check GPU compatibility and provide diagnostics"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_capability = torch.cuda.get_device_capability(0)
            print(f"🔍 GPU detected: {gpu_name}")
            print(f"🔍 CUDA capability: sm_{gpu_capability[0]}{gpu_capability[1]}")
            print(f"🔍 PyTorch version: {torch.__version__}")
            
            # Check for RTX 5090 - try GPU mode if forced, otherwise intelligent detection
            if "RTX 5090" in gpu_name or gpu_capability >= (12, 0):
                if os.environ.get("RTX5090_FORCE_GPU") == "1":
                    print("🔥 RTX 5090 detected - FORCED GPU mode enabled!")
                    return "rtx5090_gpu"
                else:
                    print("🔥 RTX 5090 detected - attempting GPU mode with smart fallback")
                    return "rtx5090_gpu"
            elif gpu_capability >= (9, 0):
                print("⚠️  High-end GPU detected - using compatibility mode")
                return "high_end"
        else:
            print("❌ No CUDA GPU detected")
            return False
    except Exception as e:
        print(f"⚠️  GPU check failed: {e}")
        return False
    return "normal"

def upgrade_pytorch_for_rtx5090():
    """Try to upgrade PyTorch to latest version for better RTX 5090 support"""
    print("🔄 Upgrading PyTorch for RTX 5090 support...")
    try:
        # Try PyTorch 2.2+ which should have better RTX 5090 support
        subprocess.run([
            "pip", "install", "--upgrade", "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ], check=True)
        print("✅ PyTorch upgraded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  PyTorch upgrade failed: {e}")
        try:
            # Fallback to nightly
            subprocess.run([
                "pip", "install", "--pre", "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/nightly/cu121"
            ], check=True)
            print("✅ PyTorch nightly installed as fallback")
            return True
        except subprocess.CalledProcessError as e2:
            print(f"❌ All PyTorch upgrades failed: {e2}")
            return False

def launch_webui_with_fallback():
    """Launch WebUI with automatic fallback to CPU mode if CUDA fails"""
    # Pre-check GPU compatibility
    gpu_compatibility = check_gpu_compatibility()
    
    if gpu_compatibility == "rtx5090_gpu":
        print("🔥 RTX 5090 detected - attempting GPU mode with aggressive compatibility")
        try:
            launch_webui()
        except Exception as e:
            if "CUDA" in str(e) or "kernel image" in str(e):
                print("❌ RTX 5090 CUDA failed - falling back to CPU mode")
                print(f"Error: {e}")
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                os.environ["FORCE_CPU"] = "1"
                print("💻 CPU fallback activated - generation will be slower but stable")
                launch_webui()
            else:
                raise
    else:
        # Try launching normally first
        print("🚀 Launching WebUI with GPU support...")
        launch_webui()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clone_webui():
    # Перевіряємо чи ми в Network Volume
    if os.path.exists("/workspace") and os.getcwd() != "/workspace/I-Model":
        if os.path.exists("/workspace/I-Model"):
            os.chdir("/workspace/I-Model")
            print("📁 Переключились на Network Volume: /workspace/I-Model")
        else:
            print("⚠️  Проект не знайдено в Network Volume")
    
    if not os.path.exists("sd-webui"):
        print("🔄 Cloning AUTOMATIC1111 WebUI...")
        subprocess.run(["git", "clone", "https://github.com/AUTOMATIC1111/stable-diffusion-webui", "sd-webui"])

def install_adetailer():
    ext_path = "sd-webui/extensions/adetailer"
    if not os.path.exists(ext_path):
        print("🔄 Installing ADetailer extension...")
        subprocess.run(["git", "clone", "https://github.com/Bing-su/adetailer", ext_path])

def load_models_config():
    """Load models configuration from models.yaml"""
    try:
        with open("config/models.yaml", "r") as f:
            config = yaml.safe_load(f)
        return config.get("models", {})
    except FileNotFoundError:
        logging.error("config/models.yaml not found!")
        return {}
    except Exception as e:
        logging.error(f"Error loading models config: {e}")
        return {}

def download_models(specific_models=None):
    """
    Download models from configuration.
    
    Args:
        specific_models (list): List of model keys to download. If None, downloads all models.
    """
    model_dir = "sd-webui/models/Stable-diffusion"
    os.makedirs(model_dir, exist_ok=True)

    # Load models from configuration
    models_config = load_models_config()
    
    models = {}
    
    # If config is loaded, use URLs from there, otherwise use hardcoded fallback
    if models_config:
        logging.info("Using models from config/models.yaml")
        for model_key, model_info in models_config.items():
            # Skip model if specific_models is provided and this model is not in the list
            if specific_models and model_key not in specific_models:
                continue
                
            if "download_url" in model_info and "path" in model_info:
                filename = os.path.basename(model_info["path"])
                # Store primary URL and alternatives
                urls = [model_info["download_url"]]
                if "alternative_urls" in model_info:
                    urls.extend(model_info["alternative_urls"])
                models[filename] = urls
    else:
        logging.warning("Config not found, using hardcoded URLs")
        # Fallback hardcoded models (only if config fails to load)
        models = {
            "epiCRealismXL_VXVII_CrystalClear.safetensors": "https://civitai.com/api/download/models/1920523?type=Model&format=SafeTensor",
            "RealVisXL_V5_Lightning.safetensors": "https://huggingface.co/SG161222/RealVisXL_V5.0_Lightning/resolve/main/RealVisXL_V5.0_Lightning.safetensors",
            "Juggernaut_XL_v9.safetensors": "https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/resolve/main/Juggernaut_XL_v9.safetensors"
        }

    for filename, url_or_urls in models.items():
        dest_path = os.path.join(model_dir, filename)
        if not os.path.exists(dest_path):
            logging.info(f"📥 Downloading {filename}...")
            
            # Handle both single URLs and lists of URLs
            urls = url_or_urls if isinstance(url_or_urls, list) else [url_or_urls]
            
            downloaded_successfully = False
            for attempt, url in enumerate(urls, 1):
                if len(urls) > 1:
                    logging.info(f"Attempt {attempt}/{len(urls)}: {url}")
                
                try:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (compatible; RunPod-Launcher/1.0)"
                    }
                    
                    with requests.get(url, stream=True, headers=headers, timeout=30) as r:
                        r.raise_for_status()
                        
                        # Get file size if available
                        total_size = int(r.headers.get('content-length', 0))
                        if total_size:
                            logging.info(f"File size: {total_size / (1024**3):.2f} GB")
                        
                        with open(dest_path, "wb") as f:
                            downloaded = 0
                            for chunk in r.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    downloaded += len(chunk)
                                    if total_size and downloaded % (50 * 1024 * 1024) == 0:  # Progress every 50MB
                                        progress = (downloaded / total_size) * 100
                                        logging.info(f"Progress: {progress:.1f}%")
                    
                    logging.info(f"✅ {filename} downloaded successfully.")
                    downloaded_successfully = True
                    break
                    
                except requests.exceptions.RequestException as e:
                    logging.warning(f"⚠️  Attempt {attempt} failed: {e}")
                    # Clean up partial download
                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                    
                    # If this was the last attempt, log as error
                    if attempt == len(urls):
                        logging.error(f"❌ All download attempts failed for {filename}")
                    else:
                        logging.info(f"Trying alternative URL...")
                        
                except Exception as e:
                    logging.error(f"❌ Unexpected error downloading {filename}: {e}")
                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                    break
            
            if not downloaded_successfully:
                logging.error(f"❌ Failed to download {filename} from any source")

def launch_webui():
    os.chdir("sd-webui")
    print("🚀 Launching WebUI...")
    
    # Check GPU compatibility
    gpu_compatibility = check_gpu_compatibility()
    
    # Handle different GPU compatibility modes
    if gpu_compatibility == "rtx5090_gpu":
        print("🔥 Applying aggressive RTX 5090 GPU compatibility settings...")
        # Most aggressive CUDA compatibility settings for RTX 5090
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        os.environ["TORCH_CUDA_ARCH_LIST"] = "5.0;6.0;7.0;7.5;8.0;8.6;9.0;10.0;11.0;12.0"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True,backend:cudaMallocAsync"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["FORCE_CUDA"] = "1"
        # Try to force compatibility mode
        os.environ["CUDA_MODULE_LOADING"] = "LAZY"
        print("✅ Applied RTX 5090 aggressive GPU settings - attempting CUDA mode!")
    elif gpu_compatibility == "high_end":
        print("🔧 Applying high-end GPU compatibility settings...")
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        os.environ["TORCH_CUDA_ARCH_LIST"] = "5.0;6.0;7.0;7.5;8.0;8.6;9.0"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,expandable_segments:True"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["FORCE_CUDA"] = "1"
        print("✅ Applied high-end GPU compatibility settings")
    elif gpu_compatibility == "normal":
        print("✅ Normal GPU detected - using standard settings")
    else:
        print("⚠️  GPU compatibility unknown - using safe mode")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["FORCE_CPU"] = "1"
    
    # RunPod-specific flags for public access 
    base_flags = "--listen --port 3000 --enable-insecure-extension-access --theme dark --opt-split-attention --medvram --precision=full --no-half"
    
    # Add special flags for RTX 5090 GPU mode
    if gpu_compatibility == "rtx5090_gpu":
        base_flags += " --disable-nan-check --no-half-vae --autolaunch"
        print("🔥 RTX 5090 GPU mode enabled with aggressive CUDA flags")
    # Add CPU mode if CUDA is disabled
    elif os.environ.get("FORCE_CPU") == "1":
        base_flags += " --use-cpu all --skip-torch-cuda-test"
        print("💻 CPU fallback mode enabled")
    
    # Check if running on RunPod (common environment variables)
    if os.environ.get("RUNPOD_POD_ID") or os.environ.get("RUNPOD_PUBLIC_IP"):
        print("🌐 Detected RunPod environment - enabling public access")
        os.environ["COMMANDLINE_ARGS"] = base_flags
    else:
        print("💻 Local environment detected")
        local_flags = "--opt-split-attention --enable-insecure-extension-access --theme dark --precision=full --no-half"
        if gpu_compatibility == "rtx5090_gpu":
            local_flags += " --disable-nan-check --no-half-vae"
        elif os.environ.get("FORCE_CPU") == "1":
            local_flags += " --use-cpu all --skip-torch-cuda-test"
        os.environ["COMMANDLINE_ARGS"] = local_flags
    
    subprocess.run(["python3", "launch.py"])

def main():
    parser = argparse.ArgumentParser(description='RunPod Launcher для I, Model SDXL Pipeline')
    parser.add_argument('--models', nargs='*', help='Завантажити конкретні моделі (наприклад: epicrealism_xl realvisxl_v5_lightning)')
    parser.add_argument('--setup-only', action='store_true', help='Тільки налаштування (без завантаження моделей і запуску)')
    parser.add_argument('--launch', action='store_true', help='Тільки запуск WebUI (без setup)')
    parser.add_argument('--download-only', action='store_true', help='Тільки завантаження моделей')
    parser.add_argument('--force-cpu', action='store_true', help='Примусово використовувати CPU замість CUDA')
    parser.add_argument('--force-gpu-rtx5090', action='store_true', help='Примусово спробувати GPU режим для RTX 5090 (експериментально)')
    parser.add_argument('--upgrade-pytorch', action='store_true', help='Оновити PyTorch для кращої підтримки RTX 5090')
    
    args = parser.parse_args()
    
    # Handle PyTorch upgrade
    if args.upgrade_pytorch:
        print("🔄 Upgrading PyTorch for RTX 5090...")
        upgrade_pytorch_for_rtx5090()
        return
        
    # Handle GPU/CPU mode overrides
    if args.force_cpu:
        print("💻 Enabling CPU fallback mode...")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["FORCE_CPU"] = "1"
    elif args.force_gpu_rtx5090:
        print("🔥 Forcing RTX 5090 GPU mode - experimental!")
        os.environ["RTX5090_FORCE_GPU"] = "1"
    
    if args.launch:
        # Тільки запуск WebUI з автоматичним fallback
        print("🚀 Запуск Stable Diffusion WebUI...")
        launch_webui_with_fallback()
        return
    
    # Setup фаза
    print("🛠️  Налаштування середовища...")
    clone_webui()
    install_adetailer()
    
    if args.setup_only:
        print("✅ Налаштування завершено!")
        return
    
    # Завантаження моделей
    if not args.download_only and not args.models:
        print("ℹ️  Запуск без завантаження моделей. Використайте --models для завантаження.")
    else:
        download_models(args.models)
    
    if args.download_only:
        print("✅ Завантаження моделей завершено!")
        return
    
    # Повний запуск з автоматичним fallback
    print("🌐 Запуск Stable Diffusion WebUI...")
    launch_webui_with_fallback()

if __name__ == "__main__":
    main()
