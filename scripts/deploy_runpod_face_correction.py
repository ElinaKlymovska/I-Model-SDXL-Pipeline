#!/usr/bin/env python3
"""
RunPod Face Correction Deployment Script
Автоматичне налаштування та запуск повної системи корекції лиця на RunPod.
"""

import os
import sys
import subprocess
import time
import logging
import argparse
import requests
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/runpod_deployment.log')
    ]
)
logger = logging.getLogger(__name__)

class RunPodFaceCorrectionDeployer:
    """Enhanced deployment for face correction pipeline on RunPod"""
    
    def __init__(self):
        self.workspace_path = "/workspace"
        self.project_path = os.path.join(self.workspace_path, "I-Model")
        self.is_runpod = self._detect_runpod_environment()
        
    def _detect_runpod_environment(self) -> bool:
        """Detect if running on RunPod"""
        runpod_indicators = [
            os.environ.get("RUNPOD_POD_ID"),
            os.environ.get("RUNPOD_PUBLIC_IP"),
            os.path.exists("/workspace")
        ]
        return any(runpod_indicators)
    
    def setup_workspace(self):
        """Setup workspace structure"""
        logger.info("🏗️ Setting up workspace...")
        
        # Change to workspace if in RunPod
        if self.is_runpod and os.path.exists(self.workspace_path):
            os.chdir(self.workspace_path)
            logger.info(f"📁 Changed to workspace: {self.workspace_path}")
        
        # Create project directory
        if not os.path.exists(self.project_path):
            os.makedirs(self.project_path, exist_ok=True)
            logger.info(f"📁 Created project directory: {self.project_path}")
    
    def clone_or_update_project(self, repo_url: str = None):
        """Clone or update the project repository"""
        logger.info("📦 Setting up project repository...")
        
        if not repo_url:
            repo_url = "https://github.com/ElinaKlymovska/I-Model-SDXL-Pipeline.git"
        
        if os.path.exists(self.project_path) and os.path.exists(os.path.join(self.project_path, ".git")):
            logger.info("🔄 Updating existing repository...")
            os.chdir(self.project_path)
            subprocess.run(["git", "pull", "origin", "develop"], check=True)
        else:
            logger.info("📥 Cloning repository...")
            subprocess.run([
                "git", "clone", "-b", "develop", repo_url, self.project_path
            ], check=True)
        
        os.chdir(self.project_path)
        logger.info(f"✅ Project ready at: {self.project_path}")
    
    def install_dependencies(self):
        """Install Python dependencies"""
        logger.info("📦 Installing dependencies...")
        
        # Upgrade pip first
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        requirements_files = ["requirements_runpod.txt", "requirements.txt"]
        for req_file in requirements_files:
            if os.path.exists(req_file):
                logger.info(f"Installing from {req_file}...")
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", req_file
                ], check=True)
                break
        
        logger.info("✅ Dependencies installed")
    
    def setup_stable_diffusion_webui(self, preset: str = "professional"):
        """Setup Stable Diffusion WebUI with ADetailer"""
        logger.info("🎨 Setting up Stable Diffusion WebUI...")
        
        # Run runpod_launcher with face models
        cmd = [
            sys.executable, "utils/runpod_launcher.py",
            "--setup-only",
            "--preset", preset,
            "--face-models"
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        logger.info("✅ WebUI setup completed")
    
    def download_models(self, models: list = None, preset: str = "professional"):
        """Download SDXL and face detection models"""
        logger.info("📥 Downloading models...")
        
        cmd = [
            sys.executable, "utils/runpod_launcher.py",
            "--download-only",
            "--preset", preset,
            "--face-models"
        ]
        
        if models:
            cmd.extend(["--models"] + models)
        
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        logger.info("✅ Models downloaded")
    
    def start_webui_server(self, background: bool = True):
        """Start WebUI server"""
        logger.info("🚀 Starting WebUI server...")
        
        cmd = [sys.executable, "utils/runpod_launcher.py", "--launch"]
        
        if background:
            logger.info("Starting WebUI in background...")
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            
            # Wait a bit and check if it started
            time.sleep(10)
            if process.poll() is None:
                logger.info("✅ WebUI started in background")
                logger.info("🌐 WebUI will be available on port 3000")
                if self.is_runpod:
                    pod_id = os.environ.get("RUNPOD_POD_ID", "YOUR_POD_ID")
                    logger.info(f"🔗 Access URL: https://{pod_id}-3000.proxy.runpod.net")
                return process
            else:
                logger.error("❌ WebUI failed to start")
                return None
        else:
            subprocess.run(cmd, check=True)
    
    def run_face_correction_demo(self):
        """Run face correction demo"""
        logger.info("🎭 Running face correction demo...")
        
        # Wait for WebUI to be ready
        if self._wait_for_webui():
            try:
                subprocess.run([
                    sys.executable, "scripts/demo_face_correction.py",
                    "--demo", "single", "--verbose"
                ], check=True)
                logger.info("✅ Demo completed successfully")
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️ Demo failed: {e}")
        else:
            logger.error("❌ WebUI not ready for demo")
    
    def _wait_for_webui(self, timeout: int = 120) -> bool:
        """Wait for WebUI to be ready"""
        logger.info("⏳ Waiting for WebUI to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get("http://127.0.0.1:3000", timeout=5)
                if response.status_code == 200:
                    logger.info("✅ WebUI is ready!")
                    return True
            except:
                pass
            time.sleep(5)
        
        logger.warning("⚠️ WebUI readiness timeout")
        return False
    
    def create_startup_script(self):
        """Create startup script for easy relaunch"""
        startup_script = """#!/bin/bash
# Auto-generated RunPod Face Correction Startup Script

echo "🎭 Starting I-Model Face Correction Pipeline..."
cd /workspace/I-Model

# Start WebUI
python utils/runpod_launcher.py --launch &

# Wait for WebUI to start
sleep 15

echo "✅ WebUI started!"
echo "🌐 Access your WebUI at port 3000"
echo "🎯 Run face correction demo: python scripts/demo_face_correction.py"
echo "🔧 Run face correction pipeline: python scripts/demo_pipeline.py --help"

# Keep terminal open
bash
"""
        
        script_path = "/workspace/start_face_correction.sh"
        with open(script_path, "w") as f:
            f.write(startup_script)
        
        os.chmod(script_path, 0o755)
        logger.info(f"📄 Startup script created: {script_path}")
    
    def deploy_full_system(self, repo_url: str = None, models: list = None, preset: str = "professional", run_demo: bool = True):
        """Deploy the complete face correction system"""
        logger.info("🚀 Starting full RunPod face correction deployment...")
        
        try:
            # Step 1: Setup workspace
            self.setup_workspace()
            
            # Step 2: Clone/update project
            self.clone_or_update_project(repo_url)
            
            # Step 3: Install dependencies
            self.install_dependencies()
            
            # Step 4: Setup WebUI
            self.setup_stable_diffusion_webui(preset)
            
            # Step 5: Download models
            self.download_models(models, preset)
            
            # Step 6: Start WebUI
            webui_process = self.start_webui_server(background=True)
            
            # Step 7: Create startup script
            self.create_startup_script()
            
            # Step 8: Run demo if requested
            if run_demo and webui_process:
                self.run_face_correction_demo()
            
            logger.info("🎉 Full deployment completed successfully!")
            
            # Show access information
            self._show_access_info()
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            raise
    
    def _show_access_info(self):
        """Show access information"""
        logger.info("\n" + "="*60)
        logger.info("🎉 FACE CORRECTION PIPELINE DEPLOYED!")
        logger.info("="*60)
        
        if self.is_runpod:
            pod_id = os.environ.get("RUNPOD_POD_ID", "YOUR_POD_ID")
            logger.info(f"🌐 WebUI Access: https://{pod_id}-3000.proxy.runpod.net")
        else:
            logger.info(f"🌐 WebUI Access: http://localhost:3000")
        
        logger.info("\n📋 Available Commands:")
        logger.info("• Face correction demo:")
        logger.info("  python scripts/demo_face_correction.py")
        logger.info("\n• Single image processing:")
        logger.info("  python scripts/demo_pipeline.py --input photo.jpg --output ./results/")
        logger.info("\n• Batch processing:")
        logger.info("  python scripts/demo_pipeline.py --batch ./photos/ --output ./results/")
        logger.info("\n• Enhanced ADetailer:")
        logger.info("  python scripts/enhanced_adetailer.py --input photo.jpg --output enhanced.jpg")
        
        logger.info("\n🎯 Available Models:")
        logger.info("• epicrealism_xl - Epic realism and detail")
        logger.info("• copax_realistic_xl - Portrait photography specialist")  
        logger.info("• proteus_xl - Advanced photorealism")
        logger.info("• newreality_xl - Facial details expert")
        
        logger.info("\n🎭 Face Detection Models:")
        logger.info("• face_yolov8s.pt - Balanced (recommended)")
        logger.info("• face_yolov8m.pt - High accuracy")
        logger.info("• face_yolov8l.pt - Maximum quality")
        
        logger.info("\n🔄 To restart WebUI:")
        logger.info("bash /workspace/start_face_correction.sh")
        logger.info("="*60)


def main():
    """Main deployment interface"""
    parser = argparse.ArgumentParser(
        description="Deploy I-Model Face Correction Pipeline on RunPod",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full deployment with professional preset
  python deploy_runpod_face_correction.py --preset professional --demo
  
  # Quick deployment with basic models
  python deploy_runpod_face_correction.py --preset basic --no-demo
  
  # Custom models deployment
  python deploy_runpod_face_correction.py --models epicrealism_xl copax_realistic_xl
        """
    )
    
    parser.add_argument("--repo-url", 
                       help="Custom repository URL (default: GitHub repo)")
    parser.add_argument("--models", nargs="*",
                       help="Specific models to download")
    parser.add_argument("--preset", choices=["basic", "advanced", "professional"], 
                       default="professional",
                       help="Model preset (default: professional)")
    parser.add_argument("--no-demo", action="store_true",
                       help="Skip running demo after deployment")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create deployer and run
    deployer = RunPodFaceCorrectionDeployer()
    
    try:
        deployer.deploy_full_system(
            repo_url=args.repo_url,
            models=args.models,
            preset=args.preset,
            run_demo=not args.no_demo
        )
    except KeyboardInterrupt:
        logger.info("❌ Deployment cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()