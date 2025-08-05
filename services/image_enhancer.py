"""
Image Enhancement Service
Handles the core SDXL image enhancement functionality.
"""

import torch
import logging
from typing import Optional, Tuple
from PIL import Image
from pathlib import Path

from core.interfaces import BaseImageEnhancer
from core.config import ConfigManager, EnhancementConfig
from core.exceptions import EnhancementError, ModelLoadError
from services.model_manager import SDXLModelManager

logger = logging.getLogger(__name__)


class SDXLImageEnhancer(BaseImageEnhancer):
    """
    SDXL-based image enhancement service.
    Handles both full image and face region enhancement.
    """
    
    def __init__(self, config_manager: ConfigManager, model_manager: SDXLModelManager):
        super().__init__(config_manager, model_manager)
        self.current_model_name: Optional[str] = None
    
    def enhance_image(
        self, 
        image: Image.Image, 
        config: EnhancementConfig,
        model_name: str
    ) -> Image.Image:
        """
        Enhance a complete image using SDXL
        
        Args:
            image: Input image
            config: Enhancement configuration
            model_name: Name of the model to use
            
        Returns:
            Enhanced image
        """
        try:
            # Ensure model is loaded
            if not self._ensure_model_loaded(model_name):
                raise ModelLoadError(f"Failed to load model: {model_name}", model_name=model_name)
            
            # Get model and configuration
            model = self.model_manager.get_model(model_name)
            model_config = self.config_manager.get_model_config(model_name)
            
            # Prepare prompts
            prompt = f"{model_config.style_prompt}, high quality"
            negative_prompt = config.negative_prompt or self.config_manager.get_negative_prompt("default")
            
            # Generate enhanced image
            enhanced = model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                strength=config.strength,
                guidance_scale=config.guidance_scale,
                num_inference_steps=config.num_inference_steps,
                generator=torch.Generator(device=self.model_manager.device).manual_seed(42)
            ).images[0]
            
            logger.debug(f"Enhanced image using model {model_name}")
            return enhanced
            
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            raise EnhancementError(f"Image enhancement failed: {e}") from e
    
    def enhance_face_region(
        self, 
        image: Image.Image, 
        face_bbox: tuple, 
        config: EnhancementConfig,
        model_name: str
    ) -> Tuple[Image.Image, tuple]:
        """
        Enhance a specific face region in an image
        
        Args:
            image: Original image
            face_bbox: Face bounding box (x, y, w, h)
            config: Enhancement configuration
            model_name: Name of the model to use
            
        Returns:
            Tuple of (enhanced_face_image, face_coordinates)
        """
        try:
            # Ensure model is loaded
            if not self._ensure_model_loaded(model_name):
                raise ModelLoadError(f"Failed to load model: {model_name}", model_name=model_name)
            
            # Extract and prepare face region
            face_crop, face_coords = self._extract_face_region(image, face_bbox, config)
            
            # Get model and configuration
            model = self.model_manager.get_model(model_name)
            model_config = self.config_manager.get_model_config(model_name)
            
            # Prepare face-specific prompts
            prompt = f"{model_config.style_prompt}, detailed facial features, sharp focus, natural skin texture"
            negative_prompt = config.negative_prompt or self.config_manager.get_negative_prompt("portrait_focused")
            
            # Enhance face region
            enhanced_face = model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=face_crop,
                strength=config.face_enhancement_strength,
                guidance_scale=model_config.optimal_guidance,
                num_inference_steps=model_config.optimal_steps,
                generator=torch.Generator(device=self.model_manager.device).manual_seed(42)
            ).images[0]
            
            # Resize back to original crop size
            enhanced_face = enhanced_face.resize(face_crop.size, Image.LANCZOS)
            
            logger.debug(f"Enhanced face region using model {model_name}")
            return enhanced_face, face_coords
            
        except Exception as e:
            logger.error(f"Face enhancement failed: {e}")
            raise EnhancementError(f"Face enhancement failed: {e}") from e
    
    def _ensure_model_loaded(self, model_name: str) -> bool:
        """Ensure the specified model is loaded"""
        if not self.model_manager.is_model_loaded(model_name):
            return self.model_manager.load_model(model_name)
        return True
    
    def _extract_face_region(
        self, 
        image: Image.Image, 
        face_bbox: tuple, 
        config: EnhancementConfig
    ) -> Tuple[Image.Image, tuple]:
        """
        Extract face region with padding for enhancement
        
        Args:
            image: Original image
            face_bbox: Face bounding box (x, y, w, h)
            config: Enhancement configuration
            
        Returns:
            Tuple of (face_crop, face_coordinates)
        """
        x, y, w, h = face_bbox
        padding = config.face_padding
        
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
        face_crop = self._resize_for_sdxl(face_crop)
        
        return face_crop, (x1, y1, x2, y2)
    
    def _resize_for_sdxl(self, image: Image.Image, target_size: int = 512) -> Image.Image:
        """
        Resize image to optimal dimensions for SDXL (multiples of 8)
        
        Args:
            image: Input image
            target_size: Target size for the larger dimension
            
        Returns:
            Resized image
        """
        # Calculate new dimensions maintaining aspect ratio
        if image.width > image.height:
            new_width = target_size
            new_height = int((target_size * image.height) / image.width)
        else:
            new_height = target_size
            new_width = int((target_size * image.width) / image.height)
        
        # Ensure dimensions are multiples of 8
        new_width = (new_width // 8) * 8
        new_height = (new_height // 8) * 8
        
        # Ensure minimum size
        new_width = max(new_width, 64)
        new_height = max(new_height, 64)
        
        return image.resize((new_width, new_height), Image.LANCZOS)
    
    def blend_face_into_image(
        self, 
        original_image: Image.Image, 
        enhanced_face: Image.Image, 
        face_coords: tuple,
        blend_strength: float = 1.0
    ) -> Image.Image:
        """
        Blend enhanced face back into the original image
        
        Args:
            original_image: Original full image
            enhanced_face: Enhanced face region
            face_coords: Coordinates where to place the face (x1, y1, x2, y2)
            blend_strength: Strength of blending (0.0 to 1.0)
            
        Returns:
            Image with enhanced face blended in
        """
        try:
            result_image = original_image.copy()
            x1, y1, x2, y2 = face_coords
            
            # Resize enhanced face to match coordinates
            enhanced_face_resized = enhanced_face.resize((x2 - x1, y2 - y1), Image.LANCZOS)
            
            if blend_strength >= 1.0:
                # Direct replacement
                result_image.paste(enhanced_face_resized, (x1, y1))
            else:
                # Alpha blending
                original_region = original_image.crop((x1, y1, x2, y2))
                blended = Image.blend(original_region, enhanced_face_resized, blend_strength)
                result_image.paste(blended, (x1, y1))
            
            return result_image
            
        except Exception as e:
            logger.error(f"Face blending failed: {e}")
            # Return original image on failure
            return original_image
    
    def get_enhancement_suggestions(self, image: Image.Image, faces_count: int) -> dict:
        """
        Get enhancement suggestions based on image characteristics
        
        Args:
            image: Input image
            faces_count: Number of faces detected
            
        Returns:
            Dictionary with enhancement suggestions
        """
        suggestions = {
            "recommended_model": "epicrealism_xl",  # Default
            "recommended_preset": "moderate_enhancement",
            "custom_settings": {}
        }
        
        # Model suggestions based on use case
        if faces_count > 0:
            suggestions["recommended_model"] = "epicrealism_xl"  # Best for portraits
            suggestions["recommended_preset"] = "portrait_focus"
        else:
            suggestions["recommended_model"] = "realvis_xl_lightning"  # Fast for general images
        
        # Settings based on image size
        width, height = image.size
        total_pixels = width * height
        
        if total_pixels > 1024 * 1024:  # Large image
            suggestions["custom_settings"]["num_inference_steps"] = 15  # Fewer steps for speed
        elif total_pixels < 512 * 512:  # Small image
            suggestions["custom_settings"]["strength"] = 0.4  # Higher strength for enhancement
        
        # Settings based on face count
        if faces_count > 3:
            suggestions["custom_settings"]["face_enhancement_strength"] = 0.3  # Lower for multiple faces
        elif faces_count == 1:
            suggestions["custom_settings"]["face_enhancement_strength"] = 0.5  # Higher for single portrait
        
        return suggestions