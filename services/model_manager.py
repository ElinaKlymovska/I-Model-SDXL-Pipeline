"""
Model Management Service
Handles loading, caching, and management of SDXL models.
"""

import os
import torch
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from diffusers import StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler

from core.interfaces import BaseModelManager
from core.config import ConfigManager, ModelConfig
from core.exceptions import ModelLoadError, ResourceError

logger = logging.getLogger(__name__)


class SDXLModelManager(BaseModelManager):
    """
    Manages SDXL model loading, caching, and optimization.
    Handles model storage on both local and RunPod environments.
    """
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        
        # Setup paths
        runpod_config = config_manager.get_runpod_config()
        self.models_path = runpod_config.models_path
        self.cache_path = runpod_config.cache_path
        
        # Create directories
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.cache_path, exist_ok=True)
        
        # Set environment variables for caching
        self._setup_environment()
        
        # Track current device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"SDXLModelManager initialized with device: {self.device}")
        logger.info(f"Models path: {self.models_path}")
        logger.info(f"Cache path: {self.cache_path}")
    
    def _setup_environment(self):
        """Setup environment variables for model caching"""
        runpod_config = self.config_manager.get_runpod_config()
        
        for key, value in runpod_config.environment_variables.items():
            os.environ[key] = value
            logger.debug(f"Set environment variable: {key}={value}")
    
    def _load_model_implementation(self, model_name: str, model_config: ModelConfig) -> StableDiffusionXLImg2ImgPipeline:
        """Load SDXL model implementation"""
        model_path = Path(self.models_path) / model_name
        
        try:
            # Determine source (local or remote)
            if model_path.exists():
                logger.info(f"Loading model from local path: {model_path}")
                model_id = str(model_path)
                local_files_only = True
            else:
                logger.info(f"Loading model from HuggingFace: {model_config.model_id}")
                model_id = model_config.model_id
                local_files_only = False
            
            # Load pipeline
            pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
                cache_dir=self.cache_path,
                local_files_only=local_files_only
            )
            
            logger.info(f"Successfully loaded model: {model_name}")
            
            # Save model locally if not already saved
            if not model_path.exists():
                logger.info(f"Saving model to {model_path} for future use...")
                pipeline.save_pretrained(str(model_path))
            
            return pipeline
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise ModelLoadError(f"Failed to load model {model_name}", model_name=model_name) from e
    
    def _prepare_model_for_inference(self, model: StableDiffusionXLImg2ImgPipeline, model_config: ModelConfig) -> StableDiffusionXLImg2ImgPipeline:
        """Prepare model for inference with optimizations"""
        try:
            # Move to device
            model = model.to(self.device)
            
            # Apply memory optimizations
            runpod_config = self.config_manager.get_runpod_config()
            memory_opts = runpod_config.memory_optimization
            
            if memory_opts.get("enable_attention_slicing", True):
                model.enable_attention_slicing()
                logger.debug("Enabled attention slicing")
            
            if memory_opts.get("enable_xformers", True):
                try:
                    model.enable_xformers_memory_efficient_attention()
                    logger.debug("Enabled xFormers memory efficient attention")
                except Exception as e:
                    logger.warning(f"Could not enable xFormers: {e}")
            
            # Set efficient scheduler
            model.scheduler = DPMSolverMultistepScheduler.from_config(
                model.scheduler.config
            )
            
            logger.info(f"Model prepared for inference with optimizations")
            return model
            
        except Exception as e:
            logger.error(f"Failed to prepare model for inference: {e}")
            raise ModelLoadError(f"Failed to prepare model for inference") from e
    
    def _handle_load_error(self, model_name: str, error: Exception):
        """Handle model loading errors with helpful suggestions"""
        logger.error(f"Model loading failed for {model_name}: {error}")
        
        # Check available space
        if "No space left on device" in str(error):
            raise ResourceError(
                f"Insufficient disk space to load model {model_name}",
                resource_type="disk_space"
            ) from error
        
        # Check memory
        if "CUDA out of memory" in str(error):
            raise ResourceError(
                f"Insufficient GPU memory to load model {model_name}",
                resource_type="gpu_memory"
            ) from error
        
        # Re-raise as ModelLoadError
        raise ModelLoadError(f"Failed to load model {model_name}", model_name=model_name) from error
    
    def get_model_memory_usage(self, model_name: str) -> Dict[str, Any]:
        """Get memory usage information for a loaded model"""
        if not self.is_model_loaded(model_name):
            return {"error": f"Model {model_name} not loaded"}
        
        if self.device == "cuda":
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
            
            return {
                "model_name": model_name,
                "gpu_memory_allocated": f"{memory_allocated:.2f} GB",
                "gpu_memory_cached": f"{memory_cached:.2f} GB",
                "device": self.device
            }
        else:
            return {
                "model_name": model_name,
                "device": self.device,
                "note": "Memory tracking not available for CPU"
            }
    
    def optimize_memory(self):
        """Optimize memory usage"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            logger.info("GPU memory cache cleared")
    
    def get_available_models_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available models"""
        models_info = {}
        
        for model_name in self.config_manager.list_available_models():
            model_config = self.config_manager.get_model_config(model_name)
            model_path = Path(self.models_path) / model_name
            
            models_info[model_name] = {
                "description": model_config.description,
                "model_id": model_config.model_id,
                "optimal_steps": model_config.optimal_steps,
                "optimal_guidance": model_config.optimal_guidance,
                "vram_requirement": model_config.vram_requirement,
                "download_size": model_config.download_size,
                "locally_available": model_path.exists(),
                "currently_loaded": self.is_model_loaded(model_name),
                "special_features": model_config.special_features
            }
        
        return models_info
    
    def preload_model(self, model_name: str) -> bool:
        """Preload a model for faster inference"""
        return self.load_model(model_name)
    
    def switch_model(self, new_model_name: str, unload_current: bool = True) -> bool:
        """Switch to a different model, optionally unloading current ones"""
        if unload_current:
            # Unload all current models to free memory
            for loaded_model in list(self.get_loaded_models()):
                self.unload_model(loaded_model)
                logger.info(f"Unloaded model: {loaded_model}")
        
        return self.load_model(new_model_name)
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about model management"""
        return {
            "loaded_models": self.get_loaded_models(),
            "total_models_available": len(self.config_manager.list_available_models()),
            "models_path": self.models_path,
            "cache_path": self.cache_path,
            "device": self.device,
            "memory_info": self.get_memory_info() if self.device == "cuda" else None
        }
    
    def get_memory_info(self) -> Dict[str, str]:
        """Get current GPU memory information"""
        if self.device != "cuda":
            return {"error": "GPU not available"}
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        return {
            "allocated": f"{allocated:.2f} GB",
            "cached": f"{cached:.2f} GB", 
            "total": f"{total:.2f} GB",
            "free": f"{total - allocated:.2f} GB"
        }