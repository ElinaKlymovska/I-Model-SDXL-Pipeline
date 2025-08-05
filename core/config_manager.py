"""
Configuration Manager
Управляє всіма конфігураціями проекту: models, pipeline settings, prompts.
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Pipeline configuration settings"""
    sdxl_model: str
    detail_model: Optional[str]
    face_detection_model: str
    character_type: str
    enhancement_level: str
    quality_preset: str
    save_intermediate: bool
    skip_preprocessing: bool
    skip_postprocessing: bool

@dataclass
class WebUIConfig:
    """WebUI configuration settings"""
    port: int
    api_timeout: int
    max_retries: int
    retry_delay: int
    sampler: str
    cfg_scale: int
    steps: int
    width: int
    height: int

class ConfigManager:
    """
    Centralized configuration manager for the face correction pipeline.
    Читає налаштування з YAML файлів та environment variables.
    """
    
    def __init__(self, config_dir: str = None):
        """
        Initialize configuration manager
        
        Args:
            config_dir: Path to configuration directory (default: project_root/config)
        """
        if config_dir is None:
            # Auto-detect project root and config directory
            self.project_root = self._find_project_root()
            self.config_dir = self.project_root / "config"
        else:
            self.config_dir = Path(config_dir)
            self.project_root = self.config_dir.parent
        
        # Configuration cache
        self._models_config = None
        self._pipeline_config = None
        self._prompts_config = None
        
        logger.info(f"ConfigManager initialized: {self.config_dir}")
    
    def _find_project_root(self) -> Path:
        """Find project root directory"""
        current = Path(__file__).parent
        
        # Look for project indicators
        indicators = [".cursorrules", "config", "scripts", "utils"]
        
        while current.parent != current:  # Not root directory
            if all((current / indicator).exists() for indicator in indicators[:2]):
                return current
            current = current.parent
        
        # Fallback to parent directory of this file
        return Path(__file__).parent.parent
    
    def load_models_config(self) -> Dict[str, Any]:
        """Load models configuration"""
        if self._models_config is None:
            config_path = self.config_dir / "models.yaml"
            self._models_config = self._load_yaml_file(config_path, "models configuration")
        return self._models_config
    
    def load_pipeline_config(self) -> Dict[str, Any]:
        """Load pipeline settings configuration"""
        if self._pipeline_config is None:
            config_path = self.config_dir / "pipeline_settings.yaml"
            self._pipeline_config = self._load_yaml_file(config_path, "pipeline configuration")
        return self._pipeline_config
    
    def load_prompts_config(self) -> Dict[str, Any]:
        """Load prompts configuration"""
        if self._prompts_config is None:
            config_path = self.config_dir / "prompt_settings.yaml"
            self._prompts_config = self._load_yaml_file(config_path, "prompts configuration")
        return self._prompts_config
    
    def _load_yaml_file(self, file_path: Path, description: str) -> Dict[str, Any]:
        """Load YAML file with error handling"""
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    logger.debug(f"Loaded {description} from {file_path}")
                    return config or {}
            else:
                logger.warning(f"{description} file not found: {file_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading {description} from {file_path}: {e}")
            return {}
    
    def get_default_pipeline_config(self) -> PipelineConfig:
        """Get default pipeline configuration"""
        pipeline_config = self.load_pipeline_config()
        defaults = pipeline_config.get('default_models', {})
        processing = pipeline_config.get('default_processing', {})
        options = pipeline_config.get('pipeline_options', {})
        
        return PipelineConfig(
            sdxl_model=defaults.get('sdxl_model', 'copax_realistic_xl'),
            detail_model=defaults.get('detail_model'),
            face_detection_model=defaults.get('face_detection_model', 'face_yolov8m.pt'),
            character_type=processing.get('character_type', 'female_portrait'),
            enhancement_level=processing.get('enhancement_level', 'medium'),
            quality_preset=processing.get('quality_preset', 'balanced'),
            save_intermediate=options.get('save_intermediate_results', True),
            skip_preprocessing=options.get('skip_preprocessing', False),
            skip_postprocessing=options.get('skip_postprocessing', False)
        )
    
    def get_webui_config(self) -> WebUIConfig:
        """Get WebUI configuration"""
        pipeline_config = self.load_pipeline_config()
        webui_settings = pipeline_config.get('webui_integration', {})
        default_settings = webui_settings.get('default_settings', {})
        
        return WebUIConfig(
            port=webui_settings.get('default_port', 3000),
            api_timeout=webui_settings.get('api_timeout', 600),
            max_retries=webui_settings.get('max_retries', 3),
            retry_delay=webui_settings.get('retry_delay', 5),
            sampler=default_settings.get('sampler', 'DPM++ 2M Karras'),
            cfg_scale=default_settings.get('cfg_scale', 7),
            steps=default_settings.get('steps', 30),
            width=default_settings.get('width', 768),
            height=default_settings.get('height', 1024)
        )
    
    def get_model_preset_models(self, preset: str) -> List[str]:
        """Get list of models for a preset"""
        pipeline_config = self.load_pipeline_config()
        presets = pipeline_config.get('model_presets', {})
        
        if preset in presets:
            return presets[preset].get('models', [])
        
        # Fallback presets
        fallback_presets = {
            'basic': ['epicrealism_xl'],
            'advanced': ['epicrealism_xl', 'copax_realistic_xl'],
            'professional': ['epicrealism_xl', 'copax_realistic_xl', 'proteus_xl', 'newreality_xl']
        }
        
        return fallback_presets.get(preset, fallback_presets['advanced'])
    
    def get_recommended_face_model(self, use_case: str = "production") -> str:
        """Get recommended face detection model for use case"""
        pipeline_config = self.load_pipeline_config()
        recommendations = pipeline_config.get('face_detection_recommendations', {})
        
        # Find model by use case
        for model_key, model_info in recommendations.items():
            if model_info.get('use_case') == use_case:
                return model_info.get('model', 'face_yolov8s.pt')
            if model_info.get('recommended', False):
                return model_info.get('model', 'face_yolov8s.pt')
        
        # Fallback
        return 'face_yolov8s.pt'
    
    def get_available_models(self) -> List[str]:
        """Get list of available SDXL models"""
        models_config = self.load_models_config()
        return list(models_config.get('models', {}).keys())
    
    def get_available_face_models(self) -> List[str]:
        """Get list of available face detection models"""
        models_config = self.load_models_config()
        adetailer_models = models_config.get('adetailer_models', {})
        return [info.get('model_file', '') for info in adetailer_models.values() if 'model_file' in info]
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        models_config = self.load_models_config()
        models = models_config.get('models', {})
        return models.get(model_name, {})
    
    def get_prompt_preset(self, preset_name: str) -> Dict[str, Any]:
        """Get prompt preset configuration"""
        prompts_config = self.load_prompts_config()
        presets = prompts_config.get('presets', {})
        return presets.get(preset_name, {})
    
    def get_enhancement_level_settings(self, level: str) -> Dict[str, Any]:
        """Get enhancement level settings"""
        prompts_config = self.load_prompts_config()
        levels = prompts_config.get('enhancement_levels', {})
        return levels.get(level, levels.get('medium', {}))
    
    def get_adetailer_settings(self, preset: str) -> Dict[str, Any]:
        """Get ADetailer settings for quality preset"""
        prompts_config = self.load_prompts_config()
        settings = prompts_config.get('adetailer_settings', {})
        return settings.get(preset, settings.get('balanced', {}))
    
    def get_character_prompts(self, character_type: str) -> Dict[str, Any]:
        """Get character-specific prompts"""
        prompts_config = self.load_prompts_config()
        character_prompts = prompts_config.get('character_prompts', {})
        return character_prompts.get(character_type, character_prompts.get('female_portrait', {}))
    
    def reload_config(self):
        """Reload all configuration from files"""
        self._models_config = None
        self._pipeline_config = None
        self._prompts_config = None
        logger.info("Configuration reloaded")
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check if config files exist
        required_files = ['models.yaml', 'pipeline_settings.yaml', 'prompt_settings.yaml']
        for filename in required_files:
            config_path = self.config_dir / filename
            if not config_path.exists():
                issues.append(f"Missing configuration file: {filename}")
        
        # Validate models config
        models_config = self.load_models_config()
        if not models_config.get('models'):
            issues.append("No models defined in models.yaml")
        
        # Validate pipeline config
        pipeline_config = self.load_pipeline_config()
        if not pipeline_config.get('default_models'):
            issues.append("No default models defined in pipeline_settings.yaml")
        
        # Validate prompts config
        prompts_config = self.load_prompts_config()
        if not prompts_config.get('enhancement_levels'):
            issues.append("No enhancement levels defined in prompt_settings.yaml")
        
        return issues
    
    def get_env_setting(self, key: str, default: Any = None) -> Any:
        """Get setting from environment variables with fallback"""
        return os.getenv(key, default)
    
    def get_webui_url(self) -> str:
        """Get WebUI URL from environment or config"""
        env_url = self.get_env_setting('WEBUI_API_URL')
        if env_url:
            return env_url
        
        webui_config = self.get_webui_config()
        return f"http://127.0.0.1:{webui_config.port}"


# Global config manager instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def reload_config():
    """Reload global configuration"""
    global _config_manager
    if _config_manager:
        _config_manager.reload_config()
    else:
        _config_manager = ConfigManager()