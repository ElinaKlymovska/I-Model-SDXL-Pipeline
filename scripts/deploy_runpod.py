#!/usr/bin/env python3
"""
RunPod Deployment Script for Image Enhancement Pipeline
Sets up the cloud environment with models and dependencies
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from runpod.manager import RunPodManager
from runpod.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RunPodDeployment:
    """Manages deployment of image enhancement pipeline to RunPod"""
    
    def __init__(self):
        self.manager = RunPodManager()
        self.setup_commands = []
        
    def prepare_deployment(self) -> Dict[str, Any]:
        """Prepare deployment configuration"""
        deployment_config = {
            "environment_setup": [
                "apt-get update",
                "apt-get install -y git wget curl",
                "pip install --upgrade pip",
            ],
            "python_packages": [
                "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
                "pip install diffusers[torch] transformers accelerate",
                "pip install xformers --index-url https://download.pytorch.org/whl/cu121",
                "pip install mediapipe opencv-python pillow",
                "pip install safetensors",
                "pip install insightface onnxruntime-gpu",
                "pip install tqdm requests numpy",
            ],
            "volume_setup": [
                "mkdir -p /runpod-volume/models",
                "mkdir -p /runpod-volume/cache", 
                "mkdir -p /runpod-volume/outputs",
                "mkdir -p /runpod-volume/temp",
                "mkdir -p /runpod-volume/data",
            ],
            "environment_variables": {
                "TRANSFORMERS_CACHE": "/runpod-volume/cache",
                "HF_HOME": "/runpod-volume/cache",
                "TORCH_HOME": "/runpod-volume/cache",
                "CUDA_VISIBLE_DEVICES": "0",
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"
            }
        }
        
        return deployment_config
    
    def deploy_pipeline(self) -> bool:
        """Deploy the complete pipeline to RunPod"""
        try:
            logger.info("Starting RunPod deployment...")
            
            # Check pod status
            if not self.manager.wait_for_pod_ready():
                logger.error("Pod not ready for deployment")
                return False
            
            # Get connection info
            conn_info = self.manager.get_connection_info()
            logger.info(f"Connected to pod: {conn_info}")
            
            # Upload project files
            logger.info("Uploading project files...")
            if not self._upload_project_files():
                return False
            
            # Setup environment
            logger.info("Setting up environment...")
            if not self._setup_environment():
                return False
            
            # Download and cache models
            logger.info("Downloading and caching models...")
            if not self._setup_models():
                return False
            
            # Run tests
            logger.info("Running deployment tests...")
            if not self._test_deployment():
                return False
            
            logger.info("✅ Deployment completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
    
    def _upload_project_files(self) -> bool:
        """Upload project files to RunPod"""
        try:
            # Upload minimal requirements for image enhancement
            remote_req_path = "/workspace/requirements_image_enhancement.txt"
            if not self.manager.upload_file("requirements_image_enhancement.txt", remote_req_path):
                return False
            
            # Upload pipeline code
            remote_pipeline_path = "/workspace/pipelines"
            if not self.manager.upload_directory("pipelines", remote_pipeline_path):
                return False
            
            # Upload configs
            remote_config_path = "/workspace/configs"
            if not self.manager.upload_directory("configs", remote_config_path):
                return False
            
            # Upload deployment script
            remote_script_path = "/workspace/scripts"
            if not self.manager.upload_directory("scripts", remote_script_path):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"File upload failed: {e}")
            return False
    
    def _setup_environment(self) -> bool:
        """Setup the Python environment and dependencies"""
        try:
            deployment_config = self.prepare_deployment()
            
            # Update system packages
            for cmd in deployment_config["environment_setup"]:
                result = self.manager.execute_command(cmd)
                if result.get("exit_code") != 0:
                    logger.error(f"Command failed: {cmd}")
                    logger.error(result.get("stderr"))
                    return False
            
            # Set environment variables
            env_vars = deployment_config["environment_variables"]
            for key, value in env_vars.items():
                cmd = f'echo "export {key}={value}" >> ~/.bashrc'
                self.manager.execute_command(cmd)
            
            # Create volume directories
            for cmd in deployment_config["volume_setup"]:
                result = self.manager.execute_command(cmd)
                if result.get("exit_code") != 0:
                    logger.warning(f"Volume setup command failed: {cmd}")
            
            # Install Python packages
            for cmd in deployment_config["python_packages"]:
                logger.info(f"Installing: {cmd}")
                result = self.manager.execute_command(cmd, working_dir="/workspace")
                if result.get("exit_code") != 0:
                    logger.error(f"Package installation failed: {cmd}")
                    logger.error(result.get("stderr"))
                    return False
            
            # Install project requirements
            result = self.manager.execute_command(
                "pip install -r requirements_image_enhancement.txt",
                working_dir="/workspace"
            )
            if result.get("exit_code") != 0:
                logger.error("Requirements installation failed")
                logger.error(result.get("stderr"))
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Environment setup failed: {e}")
            return False
    
    def _setup_models(self) -> bool:
        """Download and setup models on persistent volume"""
        try:
            # Create model setup script
            setup_script = '''
import os
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline

# Set environment variables
os.environ["TRANSFORMERS_CACHE"] = "/runpod-volume/cache"
os.environ["HF_HOME"] = "/runpod-volume/cache"

# Models to download
models = {
    "epicrealism_xl": "stablediffusionapi/epic-realism-xl",
    "realvis_xl_lightning": "SG161222/RealVisXL_V5.0_Lightning", 
    "juggernaut_xl": "RunDiffusion/Juggernaut-XL-v9"
}

print("Downloading SDXL models to persistent volume...")

for model_name, model_id in models.items():
    print(f"\\nDownloading {model_name}...")
    try:
        model_path = f"/runpod-volume/models/{model_name}"
        
        # Download and save model
        pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            cache_dir="/runpod-volume/cache"
        )
        
        # Save to persistent volume
        pipeline.save_pretrained(model_path)
        print(f"✅ {model_name} downloaded and saved to {model_path}")
        
        # Free memory
        del pipeline
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")

print("\\n✅ Model setup completed!")
'''
            
            # Write and execute setup script
            script_path = "/workspace/setup_models.py"
            with open("temp_setup_models.py", "w") as f:
                f.write(setup_script)
            
            if not self.manager.upload_file("temp_setup_models.py", script_path):
                return False
            
            # Execute model download
            result = self.manager.execute_command(
                f"cd /workspace && python {script_path}",
                working_dir="/workspace"
            )
            
            # Clean up temp file
            os.remove("temp_setup_models.py")
            
            if result.get("exit_code") != 0:
                logger.error("Model setup failed")
                logger.error(result.get("stderr"))
                return False
            
            logger.info("Models downloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model setup failed: {e}")
            return False
    
    def _test_deployment(self) -> bool:
        """Test the deployed pipeline"""
        try:
            # Create test script
            test_script = '''
import sys
sys.path.append("/workspace")

from pipelines.inference.image_enhancement import ImageEnhancementPipeline
from PIL import Image
import torch

print("Testing image enhancement pipeline...")

# Create test pipeline
pipeline = ImageEnhancementPipeline()

# Test setup
if not pipeline.setup():
    print("❌ Pipeline setup failed")
    exit(1)

print("✅ Pipeline setup successful")

# Test model loading
models_to_test = ["epicrealism_xl"]  # Test one model

for model_name in models_to_test:
    print(f"Testing model: {model_name}")
    
    if pipeline.load_model(model_name):
        print(f"✅ {model_name} loaded successfully")
    else:
        print(f"❌ Failed to load {model_name}")
        continue

# Test with dummy image
print("Testing with dummy image...")
test_image = Image.new("RGB", (512, 512), color="red")

try:
    result = pipeline.run(
        input_data=test_image,
        model_name="epicrealism_xl",
        enhancement_config={"strength": 0.1, "num_inference_steps": 5}
    )
    
    if result.get("status") == "success":
        print("✅ Pipeline test successful")
    else:
        print(f"❌ Pipeline test failed: {result.get('error')}")
        
except Exception as e:
    print(f"❌ Pipeline test failed with exception: {e}")

print("\\n🎉 Deployment test completed!")
'''
            
            # Write and execute test script
            test_path = "/workspace/test_deployment.py"
            with open("temp_test_deployment.py", "w") as f:
                f.write(test_script)
            
            if not self.manager.upload_file("temp_test_deployment.py", test_path):
                return False
            
            # Execute test
            result = self.manager.execute_command(
                f"cd /workspace && python {test_path}",
                working_dir="/workspace"
            )
            
            # Clean up temp file
            os.remove("temp_test_deployment.py")
            
            print(result.get("stdout"))
            
            if result.get("exit_code") != 0:
                logger.error("Deployment test failed")
                logger.error(result.get("stderr"))
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Deployment test failed: {e}")
            return False
    
    def create_jupyter_notebook(self) -> str:
        """Create a Jupyter notebook for easy pipeline usage"""
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# Image Enhancement Pipeline - RunPod Usage\\n",
                        "\\n",
                        "This notebook demonstrates how to use the SDXL + ADetailer image enhancement pipeline on RunPod."
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Import required libraries\\n",
                        "import sys\\n",
                        "sys.path.append('/workspace')\\n",
                        "\\n",
                        "from pipelines.inference.image_enhancement import ImageEnhancementPipeline\\n",
                        "from PIL import Image\\n",
                        "import matplotlib.pyplot as plt\\n",
                        "import json"
                    ],
                    "outputs": []
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Initialize pipeline\\n",
                        "pipeline = ImageEnhancementPipeline()\\n",
                        "pipeline.setup()\\n",
                        "\\n",
                        "print('Available models:', list(pipeline.available_models.keys()))"
                    ],
                    "outputs": []
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Load your preferred model\\n",
                        "model_name = 'epicrealism_xl'  # Change to your preferred model\\n",
                        "pipeline.load_model(model_name)\\n",
                        "print(f'Loaded model: {model_name}')"
                    ],
                    "outputs": []
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Load and enhance your images\\n",
                        "# Upload your images to /runpod-volume/data/ first\\n",
                        "\\n",
                        "input_image_path = '/runpod-volume/data/your_image.jpg'  # Change this path\\n",
                        "output_dir = '/runpod-volume/outputs/enhanced'\\n",
                        "\\n",
                        "# Enhancement configuration\\n",
                        "config = {\\n",
                        "    'strength': 0.3,\\n",
                        "    'guidance_scale': 7.5,\\n",
                        "    'num_inference_steps': 25,\\n",
                        "    'face_enhancement_strength': 0.4\\n",
                        "}\\n",
                        "\\n",
                        "# Run enhancement\\n",
                        "results = pipeline.run(\\n",
                        "    input_data=input_image_path,\\n",
                        "    model_name=model_name,\\n",
                        "    enhancement_config=config,\\n",
                        "    output_dir=output_dir\\n",
                        ")\\n",
                        "\\n",
                        "print('Enhancement completed!')"
                    ],
                    "outputs": []
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Display results\\n",
                        "if results['status'] == 'success':\\n",
                        "    for i, result in enumerate(results['results']):\\n",
                        "        if 'enhanced_image' in result:\\n",
                        "            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))\\n",
                        "            \\n",
                        "            ax1.imshow(result['original_image'])\\n",
                        "            ax1.set_title('Original')\\n",
                        "            ax1.axis('off')\\n",
                        "            \\n",
                        "            ax2.imshow(result['enhanced_image'])\\n",
                        "            ax2.set_title(f'Enhanced (Faces: {result[\\\"faces_detected\\\"]})')\\n",
                        "            ax2.axis('off')\\n",
                        "            \\n",
                        "            plt.tight_layout()\\n",
                        "            plt.show()\\n",
                        "else:\\n",
                        "    print('Enhancement failed:', results.get('error'))"
                    ],
                    "outputs": []
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.8.10"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        # Save notebook
        with open("temp_enhancement_notebook.ipynb", "w") as f:
            json.dump(notebook_content, f, indent=2)
        
        # Upload to RunPod
        remote_notebook = "/workspace/image_enhancement_demo.ipynb"
        if self.manager.upload_file("temp_enhancement_notebook.ipynb", remote_notebook):
            logger.info(f"Jupyter notebook created: {remote_notebook}")
            os.remove("temp_enhancement_notebook.ipynb")
            return remote_notebook
        
        return None


def main():
    """Main deployment function"""
    print("🚀 Starting RunPod Image Enhancement Pipeline Deployment")
    print("=" * 60)
    
    deployment = RunPodDeployment()
    
    try:
        # Deploy pipeline
        if deployment.deploy_pipeline():
            print("\\n✅ Deployment successful!")
            
            # Create Jupyter notebook
            notebook_path = deployment.create_jupyter_notebook()
            if notebook_path:
                print(f"📓 Jupyter notebook created: {notebook_path}")
            
            # Show connection info
            conn_info = deployment.manager.get_connection_info()
            print("\\n🔗 Connection Information:")
            print(f"Pod ID: {conn_info.get('pod_id')}")
            print(f"Status: {conn_info.get('status')}")
            
            if 'jupyter_url' in conn_info.get('ports', {}):
                print(f"Jupyter URL: {conn_info['ports']['jupyter_url']}")
            
            print("\\n🎉 Your image enhancement pipeline is ready to use!")
            print("You can now:")
            print("1. Access Jupyter notebook for interactive usage")
            print("2. Upload images to /runpod-volume/data/")
            print("3. Run enhancements and download results from /runpod-volume/outputs/")
            
        else:
            print("❌ Deployment failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\\n⚠️ Deployment interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Deployment failed with error: {e}")
        return 1
    finally:
        deployment.manager.close_connection()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())