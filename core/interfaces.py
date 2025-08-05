"""
Interface Definitions
Defines protocols and abstract base classes for key pipeline components.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Any, Dict, List, Optional, Union
from PIL import Image
import torch
from dataclasses import dataclass

from .config import EnhancementConfig, ModelConfig


@dataclass
class FaceDetectionResult:
    """Result of face detection operation"""
    bbox: tuple  # (x, y, width, height)
    confidence: float
    area: int
    landmarks: Optional[Dict[str, Any]] = None


@dataclass
class EnhancementResult:
    """Result of image enhancement operation"""
    original_image: Image.Image
    enhanced_image: Image.Image
    faces_detected: int
    faces_data: List[FaceDetectionResult]
    model_used: str
    config_used: Dict[str, Any]
    processing_time: float
    metadata: Dict[str, Any]
    error: Optional[str] = None


class ModelLoader(Protocol):
    """Protocol for model loading functionality"""
    
    def load_model(self, model_name: str, model_config: ModelConfig) -> Any:
        """Load a model by name and configuration"""
        ...
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a model to free memory"""
        ...
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is currently loaded"""
        ...
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded model names"""
        ...


class FaceDetector(Protocol):
    """Protocol for face detection functionality"""
    
    def detect_faces(self, image: Image.Image) -> List[FaceDetectionResult]:
        """Detect faces in an image"""
        ...
    
    def setup(self) -> bool:
        """Setup the face detector"""
        ...
    
    def cleanup(self) -> None:
        """Cleanup face detector resources"""
        ...


class ImageEnhancer(Protocol):
    """Protocol for image enhancement functionality"""
    
    def enhance_image(
        self, 
        image: Image.Image, 
        config: EnhancementConfig,
        model_name: str
    ) -> Image.Image:
        """Enhance a single image"""
        ...
    
    def enhance_face_region(
        self, 
        image: Image.Image, 
        face_bbox: tuple, 
        config: EnhancementConfig,
        model_name: str
    ) -> tuple[Image.Image, tuple]:
        """Enhance a specific face region"""
        ...


class ImageProcessor(Protocol):
    """Protocol for image processing utilities"""
    
    def preprocess_input(self, input_data: Any) -> List[Image.Image]:
        """Preprocess input data into list of images"""
        ...
    
    def postprocess_output(self, results: List[EnhancementResult]) -> List[EnhancementResult]:
        """Postprocess enhancement results"""
        ...
    
    def save_results(self, results: List[EnhancementResult], output_dir: str) -> bool:
        """Save enhancement results to disk"""
        ...


class ResourceManager(Protocol):
    """Protocol for resource management (memory, storage)"""
    
    def check_available_memory(self) -> Dict[str, Any]:
        """Check available GPU and system memory"""
        ...
    
    def cleanup_memory(self) -> None:
        """Free up memory resources"""
        ...
    
    def ensure_storage_space(self, required_gb: float) -> bool:
        """Ensure sufficient storage space is available"""
        ...


# Abstract base classes for implementation


class BaseModelManager(ABC):
    """Base class for model management"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self._loaded_models: Dict[str, Any] = {}
    
    @abstractmethod
    def _load_model_implementation(self, model_name: str, model_config: ModelConfig) -> Any:
        """Load model implementation - to be overridden by subclasses"""
        pass
    
    @abstractmethod
    def _prepare_model_for_inference(self, model: Any, model_config: ModelConfig) -> Any:
        """Prepare model for inference (move to device, optimize, etc.)"""
        pass
    
    def load_model(self, model_name: str) -> bool:
        """Load a model by name"""
        if model_name in self._loaded_models:
            return True
        
        try:
            model_config = self.config_manager.get_model_config(model_name)
            model = self._load_model_implementation(model_name, model_config)
            model = self._prepare_model_for_inference(model, model_config)
            self._loaded_models[model_name] = model
            return True
        except Exception as e:
            self._handle_load_error(model_name, e)
            return False
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a model"""
        if model_name in self._loaded_models:
            del self._loaded_models[model_name]
            torch.cuda.empty_cache()
            return True
        return False
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get a loaded model"""
        return self._loaded_models.get(model_name)
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if model is loaded"""
        return model_name in self._loaded_models
    
    def get_loaded_models(self) -> List[str]:
        """Get list of loaded model names"""
        return list(self._loaded_models.keys())
    
    def cleanup(self):
        """Cleanup all loaded models"""
        for model_name in list(self._loaded_models.keys()):
            self.unload_model(model_name)
    
    @abstractmethod
    def _handle_load_error(self, model_name: str, error: Exception):
        """Handle model loading errors"""
        pass


class BaseFaceDetector(ABC):
    """Base class for face detection implementations"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.face_config = config_manager.get_face_detection_config()
        self.is_setup = False
    
    @abstractmethod
    def _setup_detector(self) -> bool:
        """Setup the specific detector implementation"""
        pass
    
    @abstractmethod
    def _detect_faces_implementation(self, image: Image.Image) -> List[FaceDetectionResult]:
        """Detect faces implementation"""
        pass
    
    def setup(self) -> bool:
        """Setup the face detector"""
        if not self.is_setup:
            self.is_setup = self._setup_detector()
        return self.is_setup
    
    def detect_faces(self, image: Image.Image) -> List[FaceDetectionResult]:
        """Detect faces in image"""
        if not self.is_setup:
            raise RuntimeError("Face detector not setup. Call setup() first.")
        
        results = self._detect_faces_implementation(image)
        # Sort by area (largest first)
        results.sort(key=lambda x: x.area, reverse=True)
        return results
    
    @abstractmethod
    def cleanup(self):
        """Cleanup detector resources"""
        pass


class BaseImageEnhancer(ABC):
    """Base class for image enhancement implementations"""
    
    def __init__(self, config_manager, model_manager):
        self.config_manager = config_manager
        self.model_manager = model_manager
    
    @abstractmethod
    def enhance_image(
        self, 
        image: Image.Image, 
        config: EnhancementConfig,
        model_name: str
    ) -> Image.Image:
        """Enhance a single image"""
        pass
    
    @abstractmethod
    def enhance_face_region(
        self, 
        image: Image.Image, 
        face_bbox: tuple, 
        config: EnhancementConfig,
        model_name: str
    ) -> tuple[Image.Image, tuple]:
        """Enhance a specific face region"""
        pass