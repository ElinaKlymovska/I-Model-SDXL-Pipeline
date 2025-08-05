"""
Custom Exception Classes
Provides structured error handling for the image enhancement pipeline.
"""


class PipelineError(Exception):
    """Base exception for pipeline-related errors"""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "PIPELINE_ERROR"
        self.details = details or {}
    
    def __str__(self):
        return f"[{self.error_code}] {self.message}"


class ConfigurationError(PipelineError):
    """Raised when there are configuration-related issues"""
    
    def __init__(self, message: str, config_key: str = None, **kwargs):
        super().__init__(message, error_code="CONFIG_ERROR", **kwargs)
        self.config_key = config_key


class ModelLoadError(PipelineError):
    """Raised when model loading fails"""
    
    def __init__(self, message: str, model_name: str = None, **kwargs):
        super().__init__(message, error_code="MODEL_LOAD_ERROR", **kwargs)
        self.model_name = model_name


class EnhancementError(PipelineError):
    """Raised when image enhancement fails"""
    
    def __init__(self, message: str, image_path: str = None, **kwargs):
        super().__init__(message, error_code="ENHANCEMENT_ERROR", **kwargs)
        self.image_path = image_path


class FaceDetectionError(PipelineError):
    """Raised when face detection fails"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="FACE_DETECTION_ERROR", **kwargs)


class ResourceError(PipelineError):
    """Raised when there are resource-related issues (memory, disk space, etc.)"""
    
    def __init__(self, message: str, resource_type: str = None, **kwargs):
        super().__init__(message, error_code="RESOURCE_ERROR", **kwargs)
        self.resource_type = resource_type


class RunPodError(PipelineError):
    """Raised when RunPod operations fail"""
    
    def __init__(self, message: str, pod_id: str = None, **kwargs):
        super().__init__(message, error_code="RUNPOD_ERROR", **kwargs)
        self.pod_id = pod_id