"""
Configuration Management System
Centralized configuration for the image enhancement pipeline.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a specific SDXL model"""
    model_id: str
    description: str
    style_prompt: str
    optimal_steps: int
    optimal_guidance: float
    recommended_strength: Dict[str, float]
    vram_requirement: str
    download_size: str
    special_features: List[str] = field(default_factory=list)


@dataclass
class EnhancementConfig:
    """Configuration for enhancement parameters"""
    strength: float = 0.3
    face_enhancement_strength: float = 0.4
    guidance_scale: float = 7.5
    num_inference_steps: int = 20
    face_padding: float = 0.1
    min_face_size: int = 64
    negative_prompt: str = ""
    use_case: str = "general_improvement"
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for pipeline consumption"""
        return asdict(self)


@dataclass
class FaceDetectionConfig:
    """Configuration for face detection methods"""
    mediapipe_confidence: float = 0.5
    mediapipe_model_selection: int = 1
    opencv_scale_factor: float = 1.1
    opencv_min_neighbors: int = 5
    opencv_min_size: tuple = (64, 64)


@dataclass
class RunPodConfig:
    """Configuration for RunPod deployment"""
    volume_path: str = "/runpod-volume"
    models_path: str = "/runpod-volume/models"
    cache_path: str = "/runpod-volume/cache"
    temp_path: str = "/runpod-volume/temp"
    outputs_path: str = "/runpod-volume/outputs"
    environment_variables: Dict[str, str] = field(default_factory=lambda: {
        "TRANSFORMERS_CACHE": "/runpod-volume/cache",
        "HF_HOME": "/runpod-volume/cache",
        "TORCH_HOME": "/runpod-volume/cache",
        "CUDA_VISIBLE_DEVICES": "0"
    })
    memory_optimization: Dict[str, Any] = field(default_factory=lambda: {
        "enable_attention_slicing": True,
        "enable_xformers": True,
        "torch_dtype": "float16",
        "device_map": "auto"
    })


class ConfigManager:
    """
    Centralized configuration manager for the image enhancement pipeline.
    Loads from JSON files and provides typed configuration objects.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_file: Path to JSON configuration file
        """
        self.config_file = config_file or self._find_config_file()
        self._models: Dict[str, ModelConfig] = {}
        self._presets: Dict[str, EnhancementConfig] = {}
        self._negative_prompts: Dict[str, str] = {}
        self._face_detection = FaceDetectionConfig()
        self._runpod = RunPodConfig()
        
        self._load_configuration()
    
    def _find_config_file(self) -> str:
        """Find the configuration file in standard locations"""
        possible_paths = [
            "configs/models/image_enhancement_models.json",
            "../configs/models/image_enhancement_models.json",
            Path(__file__).parent.parent / "configs/models/image_enhancement_models.json"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return str(path)
        
        raise FileNotFoundError("Configuration file not found in standard locations")
    
    def _load_configuration(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            
            # Load models
            models_data = config_data.get("sdxl_models", {})
            for model_name, model_data in models_data.items():
                self._models[model_name] = ModelConfig(**model_data)
            
            # Load enhancement presets
            presets_data = config_data.get("enhancement_presets", {})
            for preset_name, preset_data in presets_data.items():
                self._presets[preset_name] = EnhancementConfig(**preset_data)
            
            # Load negative prompts
            self._negative_prompts = config_data.get("negative_prompts", {})
            
            # Load face detection config
            face_config = config_data.get("face_detection", {})
            if "mediapipe" in face_config:
                mp_config = face_config["mediapipe"]
                self._face_detection.mediapipe_confidence = mp_config.get("min_detection_confidence", 0.5)
                self._face_detection.mediapipe_model_selection = mp_config.get("model_selection", 1)
            
            if "opencv_fallback" in face_config:
                cv_config = face_config["opencv_fallback"]
                self._face_detection.opencv_scale_factor = cv_config.get("scale_factor", 1.1)
                self._face_detection.opencv_min_neighbors = cv_config.get("min_neighbors", 5)
                self._face_detection.opencv_min_size = tuple(cv_config.get("min_size", [64, 64]))
            
            # Load RunPod configuration
            runpod_config = config_data.get("runpod_optimization", {})
            if "volume_paths" in runpod_config:
                paths = runpod_config["volume_paths"]
                self._runpod.volume_path = paths.get("models", "/runpod-volume").replace("/models", "")
                self._runpod.models_path = paths.get("models", "/runpod-volume/models")
                self._runpod.cache_path = paths.get("cache", "/runpod-volume/cache")
                self._runpod.temp_path = paths.get("temp", "/runpod-volume/temp")
                self._runpod.outputs_path = paths.get("outputs", "/runpod-volume/outputs")
            
            if "environment_variables" in runpod_config:
                self._runpod.environment_variables.update(runpod_config["environment_variables"])
            
            if "memory_optimization" in runpod_config:
                self._runpod.memory_optimization.update(runpod_config["memory_optimization"])
            
            logger.info(f"Configuration loaded from {self.config_file}")
            logger.info(f"Loaded {len(self._models)} models and {len(self._presets)} presets")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for a specific model"""
        if model_name not in self._models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self._models.keys())}")
        return self._models[model_name]
    
    def get_enhancement_preset(self, preset_name: str) -> EnhancementConfig:
        """Get enhancement preset configuration"""
        if preset_name not in self._presets:
            raise ValueError(f"Preset '{preset_name}' not found. Available presets: {list(self._presets.keys())}")
        return self._presets[preset_name]
    
    def get_negative_prompt(self, prompt_type: str = "default") -> str:
        """Get negative prompt by type"""
        return self._negative_prompts.get(prompt_type, self._negative_prompts.get("default", ""))
    
    def list_available_models(self) -> List[str]:
        """Get list of available model names"""
        return list(self._models.keys())
    
    def list_available_presets(self) -> List[str]:
        """Get list of available preset names"""
        return list(self._presets.keys())
    
    def get_face_detection_config(self) -> FaceDetectionConfig:
        """Get face detection configuration"""
        return self._face_detection
    
    def get_runpod_config(self) -> RunPodConfig:
        """Get RunPod configuration"""
        return self._runpod
    
    def create_custom_enhancement_config(
        self,
        strength: float = 0.3,
        face_enhancement_strength: float = 0.4,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20,
        **kwargs
    ) -> EnhancementConfig:
        """Create a custom enhancement configuration"""
        return EnhancementConfig(
            strength=strength,
            face_enhancement_strength=face_enhancement_strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            **kwargs
        )
    
    def save_custom_preset(self, name: str, config: EnhancementConfig):
        """Save a custom preset to memory (not persisted to file)"""
        self._presets[name] = config
        logger.info(f"Custom preset '{name}' saved to memory")
    
    def get_model_info_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary information about all available models"""
        summary = {}
        for name, config in self._models.items():
            summary[name] = {
                "description": config.description,
                "optimal_steps": config.optimal_steps,
                "optimal_guidance": config.optimal_guidance,
                "vram_requirement": config.vram_requirement,
                "download_size": config.download_size,
                "special_features": config.special_features
            }
        return summary


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def reset_config_manager():
    """Reset the global configuration manager (useful for testing)"""
    global _config_manager
    _config_manager = None