"""
Refactored Image Enhancement Pipeline
Clean, modular implementation using dependency injection and service-oriented architecture.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Union
from PIL import Image

from core.config import ConfigManager, EnhancementConfig
from core.interfaces import EnhancementResult, FaceDetectionResult
from core.exceptions import PipelineError, EnhancementError, ModelLoadError, ResourceError
from core.container import Container, ServiceRegistry
from services.model_manager import SDXLModelManager
from services.face_detection import FaceDetectionService
from services.image_enhancer import SDXLImageEnhancer
from services.image_processor import ImageProcessingService
from services.resource_manager import ResourceManager
from pipelines.base import InferencePipeline

logger = logging.getLogger(__name__)


class EnhancedImagePipeline(InferencePipeline):
    """
    Refactored image enhancement pipeline using service-oriented architecture.
    
    Features:
    - Dependency injection for better testability
    - Modular service-based design
    - Comprehensive error handling
    - Resource management and monitoring
    - Pluggable face detection strategies
    """
    
    def __init__(self, config_manager: ConfigManager = None, container: Container = None):
        """
        Initialize the enhanced image pipeline
        
        Args:
            config_manager: Configuration manager instance
            container: Dependency injection container
        """
        super().__init__()
        
        # Setup dependency injection
        self.container = container or Container()
        self.service_registry = ServiceRegistry(self.container)
        
        # Initialize configuration
        if config_manager:
            self.container.register_config_manager(config_manager)
        else:
            from core.config import get_config_manager
            self.container.register_config_manager(get_config_manager())
        
        self.config_manager = self.container.get_config_manager()
        
        # Services (lazy-loaded)
        self._model_manager: Optional[SDXLModelManager] = None
        self._face_detection: Optional[FaceDetectionService] = None
        self._image_enhancer: Optional[SDXLImageEnhancer] = None
        self._image_processor: Optional[ImageProcessingService] = None
        self._resource_manager: Optional[ResourceManager] = None
        
        # Pipeline state
        self.current_model: Optional[str] = None
        self.performance_metrics: Dict[str, Any] = {}
    
    @property
    def model_manager(self) -> SDXLModelManager:
        """Lazy-loaded model manager"""
        if self._model_manager is None:
            self._model_manager = self.service_registry.get_or_create_service(
                SDXLModelManager
            )
        return self._model_manager
    
    @property
    def face_detection(self) -> FaceDetectionService:
        """Lazy-loaded face detection service"""
        if self._face_detection is None:
            self._face_detection = self.service_registry.get_or_create_service(
                FaceDetectionService
            )
        return self._face_detection
    
    @property
    def image_enhancer(self) -> SDXLImageEnhancer:
        """Lazy-loaded image enhancer service"""
        if self._image_enhancer is None:
            self._image_enhancer = self.service_registry.get_or_create_service(
                SDXLImageEnhancer, self.model_manager
            )
        return self._image_enhancer
    
    @property
    def image_processor(self) -> ImageProcessingService:
        """Lazy-loaded image processor service"""
        if self._image_processor is None:
            self._image_processor = self.service_registry.get_or_create_service(
                ImageProcessingService
            )
        return self._image_processor
    
    @property
    def resource_manager(self) -> ResourceManager:
        """Lazy-loaded resource manager service"""
        if self._resource_manager is None:
            self._resource_manager = self.service_registry.get_or_create_service(
                ResourceManager
            )
        return self._resource_manager
    
    def setup(self) -> bool:
        """Setup the enhancement pipeline and all services"""
        try:
            logger.info("Setting up Enhanced Image Pipeline...")
            
            # Check system health
            health = self.resource_manager.get_system_health()
            if health["status"] == "critical":
                logger.error("System health is critical. Cannot proceed.")
                for error in health["errors"]:
                    logger.error(f"  - {error}")
                return False
            elif health["status"] == "warning":
                logger.warning("System health warnings detected:")
                for warning in health["warnings"]:
                    logger.warning(f"  - {warning}")
            
            # Setup face detection
            if not self.face_detection.setup():
                logger.error("Failed to setup face detection service")
                return False
            
            # Setup is successful
            self.is_initialized = True
            logger.info("Enhanced Image Pipeline setup completed successfully")
            
            # Log service information
            detector_info = self.face_detection.get_detector_info()
            logger.info(f"Face detection: {detector_info['current_detector']}")
            
            return super().setup()
            
        except Exception as e:
            logger.error(f"Pipeline setup failed: {e}")
            return False
    
    def load_model(self, model_name: str) -> bool:
        """
        Load a specific SDXL model
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check GPU requirements
            gpu_check = self.resource_manager.check_gpu_requirements(model_name)
            if not gpu_check["meets_requirements"]:
                logger.error(f"GPU requirements not met for {model_name}: {gpu_check['reason']}")
                for rec in gpu_check.get("recommendations", []):
                    logger.info(f"  Recommendation: {rec}")
                return False
            
            # Check storage space
            model_config = self.config_manager.get_model_config(model_name)
            required_space = float(model_config.download_size.replace('GB', '').strip())
            
            if not self.resource_manager.ensure_storage_space(required_space):
                raise ResourceError(f"Insufficient storage space for model {model_name}")
            
            # Load the model
            if self.model_manager.load_model(model_name):
                self.current_model = model_name
                logger.info(f"Successfully loaded model: {model_name}")
                
                # Log memory usage
                memory_info = self.model_manager.get_model_memory_usage(model_name)
                if "gpu_memory_allocated" in memory_info:
                    logger.info(f"GPU memory allocated: {memory_info['gpu_memory_allocated']}")
                
                return True
            else:
                logger.error(f"Failed to load model: {model_name}")
                return False
                
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return False
    
    def preprocess_input(self, input_data: Any) -> List[Image.Image]:
        """Preprocess input data using image processor service"""
        return self.image_processor.preprocess_input(input_data)
    
    def predict(self, input_data: Any, enhancement_config: Dict = None) -> List[EnhancementResult]:
        """
        Run image enhancement prediction
        
        Args:
            input_data: Input images
            enhancement_config: Enhancement configuration dict
            
        Returns:
            List of enhancement results
        """
        if not self.is_initialized:
            raise PipelineError("Pipeline not initialized. Call setup() first.")
        
        if not self.current_model:
            raise ModelLoadError("No model loaded. Call load_model() first.")
        
        try:
            # Convert config dict to EnhancementConfig object
            if isinstance(enhancement_config, dict):
                config = EnhancementConfig(**enhancement_config)
            elif isinstance(enhancement_config, EnhancementConfig):
                config = enhancement_config
            else:
                # Use default config
                config = self.config_manager.get_enhancement_preset("moderate_enhancement")
            
            # Preprocess input
            images = self.preprocess_input(input_data)
            results = []
            
            # Process each image
            for i, image in enumerate(images):
                start_time = time.time()
                
                try:
                    result = self._process_single_image(image, config, i)
                    result.processing_time = time.time() - start_time
                    results.append(result)
                    
                    logger.info(f"Processed image {i+1}/{len(images)} in {result.processing_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Failed to process image {i}: {e}")
                    results.append(EnhancementResult(
                        original_image=image,
                        enhanced_image=image,  # Return original on failure
                        faces_detected=0,
                        faces_data=[],
                        model_used=self.current_model,
                        config_used=config.to_dict(),
                        processing_time=time.time() - start_time,
                        metadata={},
                        error=str(e)
                    ))
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise EnhancementError(f"Prediction failed: {e}") from e
    
    def _process_single_image(self, image: Image.Image, config: EnhancementConfig, index: int) -> EnhancementResult:
        """Process a single image with face detection and enhancement"""
        
        # Detect faces
        faces = self.face_detection.detect_faces(image)
        logger.debug(f"Detected {len(faces)} faces in image {index}")
        
        # Get enhancement suggestions
        suggestions = self.image_enhancer.get_enhancement_suggestions(image, len(faces))
        
        if faces:
            # Process with face enhancement
            enhanced_image = image.copy()
            
            # Enhance each detected face
            for face_idx, face in enumerate(faces):
                logger.debug(f"Enhancing face {face_idx+1} in image {index}")
                
                try:
                    enhanced_face, face_coords = self.image_enhancer.enhance_face_region(
                        enhanced_image, face.bbox, config, self.current_model
                    )
                    
                    if enhanced_face and face_coords:
                        # Blend enhanced face back into image
                        enhanced_image = self.image_enhancer.blend_face_into_image(
                            enhanced_image, enhanced_face, face_coords, blend_strength=0.9
                        )
                        
                except Exception as e:
                    logger.warning(f"Face enhancement failed for face {face_idx}: {e}")
            
            # Apply overall enhancement with lower strength to preserve face work
            overall_config = EnhancementConfig(
                strength=min(config.strength * 0.7, 0.3),  # Reduced strength
                guidance_scale=config.guidance_scale,
                num_inference_steps=config.num_inference_steps,
                negative_prompt=config.negative_prompt
            )
            
            final_enhanced = self.image_enhancer.enhance_image(
                enhanced_image, overall_config, self.current_model
            )
            
        else:
            # No faces detected, apply general enhancement
            logger.debug(f"No faces detected in image {index}, applying general enhancement")
            final_enhanced = self.image_enhancer.enhance_image(
                image, config, self.current_model
            )
        
        # Create result
        return EnhancementResult(
            original_image=image,
            enhanced_image=final_enhanced,
            faces_detected=len(faces),
            faces_data=faces,
            model_used=self.current_model,
            config_used=config.to_dict(),
            processing_time=0.0,  # Will be set by caller
            metadata={
                "enhancement_suggestions": suggestions,
                "image_index": index
            }
        )
    
    def postprocess_output(self, predictions: List[EnhancementResult]) -> List[EnhancementResult]:
        """Postprocess enhancement results using image processor service"""
        return self.image_processor.postprocess_output(predictions)
    
    def run(
        self,
        input_data: Any,
        model_name: str = "epicrealism_xl",
        enhancement_config: Union[Dict, EnhancementConfig, str] = None,
        output_dir: Optional[str] = None,
        create_comparison: bool = False
    ) -> Dict[str, Any]:
        """
        Run the complete enhancement pipeline
        
        Args:
            input_data: Input images (path, PIL Image, or list)
            model_name: Model to use for enhancement
            enhancement_config: Enhancement settings (dict, EnhancementConfig, or preset name)
            output_dir: Directory to save results
            create_comparison: Whether to create before/after comparison grid
            
        Returns:
            Dictionary with enhancement results and metadata
        """
        start_time = time.time()
        
        try:
            # Setup if not already done
            if not self.is_initialized:
                if not self.setup():
                    return {"error": "Pipeline setup failed"}
            
            # Load model if needed
            if not self.load_model(model_name):
                return {"error": f"Failed to load model: {model_name}"}
            
            # Prepare enhancement configuration
            if isinstance(enhancement_config, str):
                # Preset name
                config = self.config_manager.get_enhancement_preset(enhancement_config)
            elif isinstance(enhancement_config, dict):
                # Dictionary config
                config = EnhancementConfig(**enhancement_config)
            elif isinstance(enhancement_config, EnhancementConfig):
                # Already an EnhancementConfig
                config = enhancement_config
            else:
                # Use default
                config = self.config_manager.get_enhancement_preset("moderate_enhancement")
            
            # Run prediction
            logger.info("Starting image enhancement pipeline...")
            predictions = self.predict(input_data, config)
            results = self.postprocess_output(predictions)
            
            # Calculate metrics
            total_time = time.time() - start_time
            successful_enhancements = len([r for r in results if not r.error])
            total_faces_detected = sum(r.faces_detected for r in results)
            
            # Save results if output directory specified
            if output_dir:
                save_success = self.image_processor.save_results(results, output_dir)
                if not save_success:
                    logger.warning(f"Failed to save some results to {output_dir}")
                
                # Create comparison grid if requested
                if create_comparison:
                    comparison_path = f"{output_dir}/comparison_grid.jpg"
                    self.image_processor.create_comparison_grid(results, comparison_path)
            
            # Update performance metrics
            self.performance_metrics = {
                "last_run_time": total_time,
                "last_run_images": len(results),
                "last_run_success_rate": successful_enhancements / len(results) if results else 0,
                "total_faces_detected": total_faces_detected
            }
            
            return {
                "status": "success",
                "results": results,
                "metadata": {
                    "model_used": model_name,
                    "enhancement_config": config.to_dict(),
                    "total_images": len(results),
                    "successful_enhancements": successful_enhancements,
                    "total_faces_detected": total_faces_detected,
                    "processing_time": total_time,
                    "average_time_per_image": total_time / len(results) if results else 0,
                    "output_directory": output_dir,
                    "performance_metrics": self.performance_metrics
                }
            }
            
        except Exception as e:
            logger.error(f"Pipeline run failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "processing_time": time.time() - start_time
            }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status"""
        return {
            "pipeline_initialized": self.is_initialized,
            "current_model": self.current_model,
            "available_models": self.config_manager.list_available_models(),
            "available_presets": self.config_manager.list_available_presets(),
            "model_manager": self.model_manager.get_model_stats() if self._model_manager else None,
            "face_detection": self.face_detection.get_detector_info() if self._face_detection else None,
            "resource_status": self.resource_manager.monitor_resources() if self._resource_manager else None,
            "performance_metrics": self.performance_metrics
        }
    
    def optimize_for_batch_processing(self, batch_size: int = None) -> bool:
        """Optimize pipeline settings for batch processing"""
        try:
            # Clear memory before optimization
            self.resource_manager.cleanup_memory()
            
            # Enable memory optimizations
            if self.current_model and self._model_manager:
                model = self.model_manager.get_model(self.current_model)
                if model:
                    # Ensure memory optimizations are enabled
                    model.enable_attention_slicing()
                    try:
                        model.enable_xformers_memory_efficient_attention()
                    except Exception:
                        logger.warning("xFormers optimization not available")
            
            logger.info("Pipeline optimized for batch processing")
            return True
            
        except Exception as e:
            logger.error(f"Batch optimization failed: {e}")
            return False
    
    def switch_model(self, new_model_name: str, unload_current: bool = True) -> bool:
        """Switch to a different model"""
        try:
            if unload_current and self.current_model:
                self.model_manager.unload_model(self.current_model)
                logger.info(f"Unloaded current model: {self.current_model}")
            
            if self.load_model(new_model_name):
                logger.info(f"Switched to model: {new_model_name}")
                return True
            else:
                logger.error(f"Failed to switch to model: {new_model_name}")
                return False
                
        except Exception as e:
            logger.error(f"Model switching failed: {e}")
            return False
    
    def cleanup(self):
        """Cleanup pipeline resources"""
        try:
            # Cleanup services
            self.service_registry.cleanup_services()
            
            # Clear performance metrics
            self.performance_metrics.clear()
            
            # Reset state
            self.current_model = None
            self.is_initialized = False
            
            logger.info("Enhanced Image Pipeline cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
        
        finally:
            super().cleanup()