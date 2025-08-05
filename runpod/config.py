"""
RunPod Configuration Module
"""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env')

@dataclass
class RunPodConfig:
    """Configuration for RunPod connection and management"""
    
    # Pod details from your RunPod dashboard
    pod_id: str = "7rrr0s51geqwi3"
    
    # API settings
    api_key: Optional[str] = None  # Will be loaded from environment variable
    
    # Connection details (will be populated when pod is running)
    pod_url: Optional[str] = None
    ssh_host: str = "194.68.245.104"
    ssh_port: int = 22160
    ssh_user: str = "root"
    ssh_key_path: str = "~/.ssh/id_ed25519"
    jupyter_port: Optional[int] = 8888
    
    # Resource specifications
    gpu_type: str = "A40"
    gpu_count: int = 1
    cpu_count: int = 9
    memory_gb: int = 50
    
    # Container details
    container_image: str = "runpod/pytorch:2.8.0-py3.11-cuda12.1-cudnn-devel-ubuntu22.04"
    workspace_path: str = "/workspace"
    volume_name: str = "i-model-storage"
    
    # Local paths
    local_scripts_path: str = "./pipelines"
    local_data_path: str = "./data"
    local_models_path: str = "./models"
    local_outputs_path: str = "./outputs"
    
    def __post_init__(self):
        """Load configuration from environment variables"""
        self.api_key = os.getenv("RUNPOD_API_KEY")
        
        # Create local directories if they don't exist
        os.makedirs(self.local_scripts_path, exist_ok=True)
        os.makedirs(self.local_data_path, exist_ok=True)
        os.makedirs(self.local_models_path, exist_ok=True)
        os.makedirs(self.local_outputs_path, exist_ok=True)

# Global configuration instance
config = RunPodConfig()