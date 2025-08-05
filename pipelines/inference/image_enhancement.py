"""
Photorealistic Image Enhancement Pipeline for Character Datasets
Uses Stable Diffusion XL with ADetailer for facial correction
Optimized for RunPod deployment with persistent storage
"""

import os
import torch
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from PIL import Image
import numpy as np
from diffusers import StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler
from transformers import pipeline as hf_pipeline
import cv2

from ..base import InferencePipeline

logger = logging.getLogger(__name__)

class ImageEnhancementPipeline(InferencePipeline):
    """
    Photorealistic image enhancement pipeline using SDXL + ADetailer
    
    Features:
    - Multiple SDXL models for different enhancement styles
    - ADetailer integration for facial correction
    - Identity preservation techniques
    - Batch processing capabilities
    - RunPod volume optimization
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        # Initialize without wandb or runpod dependencies for image enhancement
        super().__init__(runpod_manager=None, wandb_tracker=None, config=config)
        
        # Model configuration
        self.current_model_name = None
        self.sdxl_pipeline = None
        self.face_detector = None
        self.face_enhancer = None
        
        # Enhancement settings
        self.default_config = {
            "strength": 0.3,  # Lower strength preserves identity better
            "guidance_scale": 7.5,
            "num_inference_steps": 20,
            "negative_prompt": "ugly, deformed, mutilated, disfigured, text, extra limbs, face cut, head cut, extra fingers, extra arms, poorly drawn face, mutation, bad proportions, cropped head, malformed limbs, mutated hands, fused fingers, long neck",
            "face_enhancement_strength": 0.4,
            "face_padding": 0.1,  # Padding around detected face
            "min_face_size": 64,  # Minimum face size to process
        }
        
        # Model paths on persistent volume
        self.volume_path = "/runpod-volume"
        self.models_path = f"{self.volume_path}/models"
        self.cache_path = f"{self.volume_path}/cache"
        
        # Available models
        self.available_models = {
            "epicrealism_xl": {
                "model_id": "stablediffusionapi/epic-realism-xl",
                "style_prompt": "professional photography, highly detailed, photorealistic",
                "optimal_steps": 25,
                "optimal_guidance": 8.0
            },
            "realvis_xl_lightning": {
                "model_id": "SG161222/RealVisXL_V5.0_Lightning",
                "style_prompt": "ultra realistic, high quality, professional portrait",
                "optimal_steps": 8,  # Lightning model needs fewer steps
                "optimal_guidance": 2.0
            },
            "juggernaut_xl": {
                "model_id": "RunDiffusion/Juggernaut-XL-v9",
                "style_prompt": "hyperrealistic, detailed skin texture, natural lighting",
                "optimal_steps": 30,
                "optimal_guidance": 7.5
            }
        }

    def setup(self) -> bool:
        """Setup the enhancement pipeline"""
        try:
            # Create directories
            os.makedirs(self.models_path, exist_ok=True)
            os.makedirs(self.cache_path, exist_ok=True)
            
            # Set cache directory for transformers
            os.environ['TRANSFORMERS_CACHE'] = self.cache_path
            os.environ['HF_HOME'] = self.cache_path
            
            # Initialize face detection
            self._setup_face_detection()
            
            logger.info("Image enhancement pipeline setup completed")
            return super().setup()
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            return False

    def _setup_face_detection(self):
        """Setup face detection and analysis tools"""
        try:
            # Use MediaPipe for face detection (lightweight and accurate)
            import mediapipe as mp
            
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_detector = self.mp_face_detection.FaceDetection(
                model_selection=1,  # Full range model (better for varied distances)
                min_detection_confidence=0.5
            )
            
            logger.info("Face detection setup completed")
            
        except ImportError:
            logger.warning("MediaPipe not available, falling back to OpenCV")
            # Fallback to OpenCV Haar cascades
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_detector = cv2.CascadeClassifier(cascade_path)

    def load_model(self, model_name: str = "epicrealism_xl") -> bool:
        """
        Load SDXL model for enhancement
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            True if successful, False otherwise
        """
        if model_name not in self.available_models:
            logger.error(f"Model {model_name} not available. Choose from: {list(self.available_models.keys())}")
            return False
            
        if self.current_model_name == model_name and self.sdxl_pipeline is not None:
            logger.info(f"Model {model_name} already loaded")
            return True
            
        try:
            model_config = self.available_models[model_name]
            model_path = f"{self.models_path}/{model_name}"
            
            logger.info(f"Loading model {model_name}...")
            
            # Check if model exists locally
            if os.path.exists(model_path):
                logger.info(f"Loading from local path: {model_path}")
                model_id = model_path
            else:
                logger.info(f"Loading from HuggingFace: {model_config['model_id']}")
                model_id = model_config['model_id']
            
            # Load pipeline with optimizations
            self.sdxl_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
                cache_dir=self.cache_path,
                local_files_only=os.path.exists(model_path)
            )
            
            # Move to GPU
            self.sdxl_pipeline = self.sdxl_pipeline.to("cuda")
            
            # Enable memory efficient attention
            self.sdxl_pipeline.enable_attention_slicing()
            self.sdxl_pipeline.enable_xformers_memory_efficient_attention()
            
            # Use efficient scheduler
            self.sdxl_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.sdxl_pipeline.scheduler.config
            )
            
            self.current_model_name = model_name
            logger.info(f"Successfully loaded model {model_name}")
            
            # Save model locally if not already saved
            if not os.path.exists(model_path):
                logger.info(f"Saving model to {model_path} for future use...")
                self.sdxl_pipeline.save_pretrained(model_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False

    def detect_faces(self, image: Image.Image) -> List[Dict]:
        """
        Detect faces in the image
        
        Args:
            image: PIL Image
            
        Returns:
            List of face bounding boxes with confidence scores
        """
        faces = []
        
        try:
            # Convert PIL to numpy array
            img_array = np.array(image)
            
            if hasattr(self, 'mp_face_detection'):
                # Use MediaPipe
                img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                results = self.face_detector.process(img_rgb)
                
                if results.detections:
                    height, width = img_array.shape[:2]
                    
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        
                        # Convert relative coordinates to absolute
                        x = int(bbox.xmin * width)
                        y = int(bbox.ymin * height)
                        w = int(bbox.width * width)
                        h = int(bbox.height * height)
                        
                        # Check minimum face size
                        if w >= self.default_config["min_face_size"] and h >= self.default_config["min_face_size"]:
                            faces.append({
                                "bbox": (x, y, w, h),
                                "confidence": detection.score[0],
                                "area": w * h
                            })
            else:
                # Use OpenCV as fallback
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                detected_faces = self.face_detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64)
                )
                
                for (x, y, w, h) in detected_faces:
                    faces.append({
                        "bbox": (x, y, w, h),
                        "confidence": 1.0,  # OpenCV doesn't provide confidence
                        "area": w * h
                    })
        
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
        
        # Sort by face area (largest first)
        faces.sort(key=lambda x: x["area"], reverse=True)
        return faces

    def enhance_face_region(self, image: Image.Image, face_bbox: tuple, 
                           enhancement_config: Dict) -> Image.Image:
        """
        Enhance specific face region using SDXL
        
        Args:
            image: Original image
            face_bbox: Face bounding box (x, y, w, h)
            enhancement_config: Enhancement settings
            
        Returns:
            Enhanced face region
        """
        try:
            x, y, w, h = face_bbox
            padding = enhancement_config.get("face_padding", 0.1)
            
            # Add padding around face
            pad_x = int(w * padding)
            pad_y = int(h * padding)
            
            # Expand bounding box with padding
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(image.width, x + w + pad_x)
            y2 = min(image.height, y + h + pad_y)
            
            # Extract face region
            face_crop = image.crop((x1, y1, x2, y2))
            
            # Resize to optimal size for SDXL (multiple of 8)
            target_size = 512
            if face_crop.width > face_crop.height:
                new_width = target_size
                new_height = int((target_size * face_crop.height) / face_crop.width)
            else:
                new_height = target_size
                new_width = int((target_size * face_crop.width) / face_crop.height)
            
            # Ensure dimensions are multiples of 8
            new_width = (new_width // 8) * 8
            new_height = (new_height // 8) * 8
            
            face_resized = face_crop.resize((new_width, new_height), Image.LANCZOS)
            
            # Enhance with SDXL
            model_config = self.available_models[self.current_model_name]
            
            prompt = f"{model_config['style_prompt']}, detailed facial features, sharp focus, natural skin texture"
            negative_prompt = enhancement_config.get("negative_prompt", self.default_config["negative_prompt"])
            
            enhanced_face = self.sdxl_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=face_resized,
                strength=enhancement_config.get("face_enhancement_strength", 0.4),
                guidance_scale=model_config.get("optimal_guidance", 7.5),
                num_inference_steps=model_config.get("optimal_steps", 20),
                generator=torch.Generator(device="cuda").manual_seed(42)  # Consistent results
            ).images[0]
            
            # Resize back to original crop size
            enhanced_face = enhanced_face.resize((x2-x1, y2-y1), Image.LANCZOS)
            
            return enhanced_face, (x1, y1, x2, y2)
            
        except Exception as e:
            logger.error(f"Face enhancement failed: {e}")
            return None, None

    def preprocess_input(self, input_data: Union[str, Image.Image, List]) -> List[Image.Image]:
        """
        Preprocess input images
        
        Args:
            input_data: Image path, PIL Image, or list of images
            
        Returns:
            List of preprocessed PIL Images
        """
        images = []
        
        if isinstance(input_data, str):
            # Single image path
            if os.path.exists(input_data):
                images.append(Image.open(input_data).convert("RGB"))
            else:
                logger.error(f"Image file not found: {input_data}")
                
        elif isinstance(input_data, Image.Image):
            # Single PIL Image
            images.append(input_data.convert("RGB"))
            
        elif isinstance(input_data, list):
            # List of images or paths
            for item in input_data:
                if isinstance(item, str) and os.path.exists(item):
                    images.append(Image.open(item).convert("RGB"))
                elif isinstance(item, Image.Image):
                    images.append(item.convert("RGB"))
                    
        return images

    def predict(self, input_data: Any, enhancement_config: Dict = None) -> List[Dict]:
        """
        Run image enhancement prediction
        
        Args:
            input_data: Input images
            enhancement_config: Enhancement configuration
            
        Returns:
            List of enhancement results
        """
        if not self.sdxl_pipeline:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        # Merge config with defaults
        config = {**self.default_config}
        if enhancement_config:
            config.update(enhancement_config)
            
        images = self.preprocess_input(input_data)
        results = []
        
        for i, image in enumerate(images):
            logger.info(f"Processing image {i+1}/{len(images)}")
            
            try:
                # Detect faces
                faces = self.detect_faces(image)
                logger.info(f"Detected {len(faces)} faces")
                
                if faces:
                    # Process with face enhancement
                    enhanced_image = image.copy()
                    
                    for face_idx, face in enumerate(faces):
                        logger.info(f"Enhancing face {face_idx+1}")
                        
                        enhanced_face, face_coords = self.enhance_face_region(
                            enhanced_image, face["bbox"], config
                        )
                        
                        if enhanced_face and face_coords:
                            # Blend enhanced face back into image
                            x1, y1, x2, y2 = face_coords
                            enhanced_image.paste(enhanced_face, (x1, y1))
                    
                    # Apply overall enhancement with lower strength
                    model_config = self.available_models[self.current_model_name]
                    
                    final_enhanced = self.sdxl_pipeline(
                        prompt=f"{model_config['style_prompt']}, high quality portrait",
                        negative_prompt=config["negative_prompt"],
                        image=enhanced_image,
                        strength=config["strength"],
                        guidance_scale=config["guidance_scale"],
                        num_inference_steps=config["num_inference_steps"],
                        generator=torch.Generator(device="cuda").manual_seed(42)
                    ).images[0]
                    
                else:
                    # No faces detected, apply general enhancement
                    logger.info("No faces detected, applying general enhancement")
                    model_config = self.available_models[self.current_model_name]
                    
                    final_enhanced = self.sdxl_pipeline(
                        prompt=f"{model_config['style_prompt']}, high quality",
                        negative_prompt=config["negative_prompt"],
                        image=image,
                        strength=config["strength"],
                        guidance_scale=config["guidance_scale"],
                        num_inference_steps=config["num_inference_steps"],
                        generator=torch.Generator(device="cuda").manual_seed(42)
                    ).images[0]
                
                results.append({
                    "original_image": image,
                    "enhanced_image": final_enhanced,
                    "faces_detected": len(faces),
                    "faces_data": faces,
                    "model_used": self.current_model_name,
                    "config_used": config
                })
                
            except Exception as e:
                logger.error(f"Enhancement failed for image {i}: {e}")
                results.append({
                    "original_image": image,
                    "enhanced_image": image,  # Return original on failure
                    "error": str(e),
                    "faces_detected": 0
                })
        
        return results

    def postprocess_output(self, predictions: List[Dict]) -> List[Dict]:
        """
        Postprocess enhancement results
        
        Args:
            predictions: Raw prediction results
            
        Returns:
            Processed results with additional metadata
        """
        processed_results = []
        
        for result in predictions:
            processed_result = result.copy()
            
            if "enhanced_image" in result and "original_image" in result:
                # Calculate enhancement metrics
                original = result["original_image"]
                enhanced = result["enhanced_image"]
                
                # Add image dimensions
                processed_result["original_size"] = original.size
                processed_result["enhanced_size"] = enhanced.size
                
                # Add processing metadata
                processed_result["processing_timestamp"] = __import__('datetime').datetime.now().isoformat()
                
            processed_results.append(processed_result)
        
        return processed_results

    def run(self, 
            input_data: Any,
            model_name: str = "epicrealism_xl",
            enhancement_config: Dict = None,
            output_dir: str = None) -> Dict[str, Any]:
        """
        Run the complete enhancement pipeline
        
        Args:
            input_data: Input images (path, PIL Image, or list)
            model_name: Model to use for enhancement
            enhancement_config: Enhancement settings
            output_dir: Directory to save results
            
        Returns:
            Dictionary with enhancement results
        """
        if not self.is_initialized:
            self.setup()
            
        # Load model if needed
        if not self.load_model(model_name):
            return {"error": "Failed to load model"}
        
        try:
            # Run prediction
            predictions = self.predict(input_data, enhancement_config)
            results = self.postprocess_output(predictions)
            
            # Save results if output directory specified
            if output_dir:
                self._save_results(results, output_dir)
            
            return {
                "status": "success",
                "results": results,
                "model_used": model_name,
                "total_images": len(results),
                "successful_enhancements": len([r for r in results if "error" not in r])
            }
            
        except Exception as e:
            logger.error(f"Pipeline run failed: {e}")
            return {"error": str(e)}

    def _save_results(self, results: List[Dict], output_dir: str):
        """Save enhancement results to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        for i, result in enumerate(results):
            if "enhanced_image" in result:
                # Save enhanced image
                enhanced_path = os.path.join(output_dir, f"enhanced_{i:03d}.jpg")
                result["enhanced_image"].save(enhanced_path, quality=95)
                
                # Save original for comparison
                original_path = os.path.join(output_dir, f"original_{i:03d}.jpg")
                result["original_image"].save(original_path, quality=95)
                
                logger.info(f"Saved results for image {i} to {output_dir}")

    def cleanup(self):
        """Cleanup pipeline resources"""
        if self.sdxl_pipeline:
            del self.sdxl_pipeline
            torch.cuda.empty_cache()
        
        if hasattr(self, 'mp_face_detection') and self.face_detector:
            self.face_detector.close()
            
        super().cleanup()