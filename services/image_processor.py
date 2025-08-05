"""
Image Processing Service
Handles image I/O, preprocessing, and postprocessing operations.
"""

import os
import json
import datetime
import logging
from typing import List, Any, Union, Dict
from pathlib import Path
from PIL import Image

from core.interfaces import ImageProcessor, EnhancementResult
from core.config import ConfigManager
from core.exceptions import EnhancementError

logger = logging.getLogger(__name__)


class ImageProcessingService:
    """
    Service for image processing operations including I/O and preprocessing.
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
    
    def preprocess_input(self, input_data: Any) -> List[Image.Image]:
        """
        Preprocess input data into list of PIL Images
        
        Args:
            input_data: Can be:
                - String: path to single image
                - List[str]: paths to multiple images
                - PIL.Image: single image
                - List[PIL.Image]: multiple images
                
        Returns:
            List of preprocessed PIL Images
        """
        images = []
        
        try:
            if isinstance(input_data, str):
                # Single image path
                images.extend(self._load_images_from_path(input_data))
                
            elif isinstance(input_data, Image.Image):
                # Single PIL Image
                images.append(self._validate_and_convert_image(input_data))
                
            elif isinstance(input_data, list):
                # List of images or paths
                for item in input_data:
                    if isinstance(item, str):
                        images.extend(self._load_images_from_path(item))
                    elif isinstance(item, Image.Image):
                        images.append(self._validate_and_convert_image(item))
                    else:
                        logger.warning(f"Unsupported input type: {type(item)}")
            else:
                raise ValueError(f"Unsupported input data type: {type(input_data)}")
            
            if not images:
                raise ValueError("No valid images found in input data")
            
            logger.info(f"Preprocessed {len(images)} images")
            return images
            
        except Exception as e:
            logger.error(f"Input preprocessing failed: {e}")
            raise EnhancementError(f"Input preprocessing failed: {e}") from e
    
    def _load_images_from_path(self, path: str) -> List[Image.Image]:
        """Load images from a file path or directory"""
        images = []
        path_obj = Path(path)
        
        if path_obj.is_file():
            # Single file
            if path_obj.suffix.lower() in self.supported_formats:
                try:
                    image = Image.open(path)
                    images.append(self._validate_and_convert_image(image))
                except Exception as e:
                    logger.error(f"Failed to load image {path}: {e}")
            else:
                logger.warning(f"Unsupported image format: {path_obj.suffix}")
                
        elif path_obj.is_dir():
            # Directory - load all supported images
            for file_path in path_obj.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                    try:
                        image = Image.open(file_path)
                        images.append(self._validate_and_convert_image(image))
                    except Exception as e:
                        logger.error(f"Failed to load image {file_path}: {e}")
        else:
            logger.error(f"Path does not exist: {path}")
        
        return images
    
    def _validate_and_convert_image(self, image: Image.Image) -> Image.Image:
        """Validate and convert image to RGB format"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Validate dimensions
        width, height = image.size
        if width < 64 or height < 64:
            logger.warning(f"Image is very small: {width}x{height}")
        
        if width > 4096 or height > 4096:
            logger.warning(f"Image is very large: {width}x{height}, consider resizing")
        
        return image
    
    def postprocess_output(self, results: List[EnhancementResult]) -> List[EnhancementResult]:
        """
        Postprocess enhancement results with additional metadata
        
        Args:
            results: List of enhancement results
            
        Returns:
            Enhanced results with additional metadata
        """
        try:
            processed_results = []
            
            for result in results:
                # Create a copy to avoid modifying original
                processed_result = EnhancementResult(
                    original_image=result.original_image,
                    enhanced_image=result.enhanced_image,
                    faces_detected=result.faces_detected,
                    faces_data=result.faces_data,
                    model_used=result.model_used,
                    config_used=result.config_used,
                    processing_time=result.processing_time,
                    metadata=result.metadata.copy(),
                    error=result.error
                )
                
                # Add additional metadata
                if result.enhanced_image and result.original_image:
                    processed_result.metadata.update({
                        "original_size": result.original_image.size,
                        "enhanced_size": result.enhanced_image.size,
                        "processing_timestamp": datetime.datetime.now().isoformat(),
                        "aspect_ratio": result.original_image.size[0] / result.original_image.size[1],
                        "total_pixels": result.original_image.size[0] * result.original_image.size[1]
                    })
                
                # Add quality metrics if possible
                try:
                    quality_metrics = self._calculate_quality_metrics(
                        result.original_image, 
                        result.enhanced_image
                    )
                    processed_result.metadata.update(quality_metrics)
                except Exception as e:
                    logger.debug(f"Could not calculate quality metrics: {e}")
                
                processed_results.append(processed_result)
            
            logger.info(f"Postprocessed {len(processed_results)} results")
            return processed_results
            
        except Exception as e:
            logger.error(f"Output postprocessing failed: {e}")
            # Return original results on failure
            return results
    
    def _calculate_quality_metrics(self, original: Image.Image, enhanced: Image.Image) -> Dict[str, Any]:
        """Calculate basic quality metrics between original and enhanced images"""
        import numpy as np
        
        try:
            # Convert to numpy arrays
            orig_array = np.array(original)
            enh_array = np.array(enhanced)
            
            # Ensure same size for comparison
            if orig_array.shape != enh_array.shape:
                enhanced_resized = enhanced.resize(original.size, Image.LANCZOS)
                enh_array = np.array(enhanced_resized)
            
            # Calculate basic metrics
            mse = np.mean((orig_array - enh_array) ** 2)
            
            # Peak Signal-to-Noise Ratio
            if mse == 0:
                psnr = float('inf')
            else:
                psnr = 20 * np.log10(255.0 / np.sqrt(mse))
            
            # Structural similarity (simplified)
            mean_orig = np.mean(orig_array)
            mean_enh = np.mean(enh_array)
            std_orig = np.std(orig_array)
            std_enh = np.std(enh_array)
            
            return {
                "quality_metrics": {
                    "mse": float(mse),
                    "psnr": float(psnr),
                    "mean_brightness_change": float(mean_enh - mean_orig),
                    "contrast_change": float(std_enh - std_orig)
                }
            }
            
        except Exception as e:
            logger.debug(f"Quality metrics calculation failed: {e}")
            return {}
    
    def save_results(self, results: List[EnhancementResult], output_dir: str) -> bool:
        """
        Save enhancement results to disk
        
        Args:
            results: Enhancement results to save
            output_dir: Output directory path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save metadata
            metadata_file = output_path / "enhancement_metadata.json"
            self._save_metadata(results, metadata_file)
            
            # Save images
            for i, result in enumerate(results):
                if result.error:
                    logger.warning(f"Skipping result {i} due to error: {result.error}")
                    continue
                
                # Save enhanced image
                enhanced_path = output_path / f"enhanced_{i:03d}.jpg"
                result.enhanced_image.save(enhanced_path, quality=95)
                
                # Save original for comparison
                original_path = output_path / f"original_{i:03d}.jpg"
                result.original_image.save(original_path, quality=95)
                
                # Save individual metadata
                individual_metadata = output_path / f"metadata_{i:03d}.json"
                self._save_individual_metadata(result, individual_metadata)
                
                logger.debug(f"Saved result {i} to {output_dir}")
            
            logger.info(f"Successfully saved {len(results)} results to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return False
    
    def _save_metadata(self, results: List[EnhancementResult], metadata_file: Path):
        """Save overall metadata for all results"""
        metadata = {
            "enhancement_session": {
                "timestamp": datetime.datetime.now().isoformat(),
                "total_images": len(results),
                "successful_enhancements": len([r for r in results if not r.error]),
                "failed_enhancements": len([r for r in results if r.error])
            },
            "results_summary": []
        }
        
        for i, result in enumerate(results):
            summary = {
                "index": i,
                "faces_detected": result.faces_detected,
                "model_used": result.model_used,
                "processing_time": result.processing_time,
                "error": result.error,
                "has_enhancement": result.enhanced_image is not None
            }
            metadata["results_summary"].append(summary)
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def _save_individual_metadata(self, result: EnhancementResult, metadata_file: Path):
        """Save metadata for individual result"""
        metadata = {
            "faces_detected": result.faces_detected,
            "faces_data": [
                {
                    "bbox": face.bbox,
                    "confidence": face.confidence,
                    "area": face.area
                } for face in result.faces_data
            ],
            "model_used": result.model_used,
            "config_used": result.config_used,
            "processing_time": result.processing_time,
            "metadata": result.metadata,
            "error": result.error
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def create_comparison_grid(self, results: List[EnhancementResult], output_path: str) -> bool:
        """
        Create a comparison grid showing before/after images
        
        Args:
            results: Enhancement results
            output_path: Path to save the comparison grid
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not results:
                return False
            
            # Filter out results with errors
            valid_results = [r for r in results if not r.error and r.enhanced_image]
            
            if not valid_results:
                logger.warning("No valid results to create comparison grid")
                return False
            
            # Limit to first 9 images for grid
            valid_results = valid_results[:9]
            
            # Calculate grid dimensions
            num_images = len(valid_results)
            grid_cols = min(3, num_images)
            grid_rows = (num_images + grid_cols - 1) // grid_cols
            
            # Calculate individual image size
            max_img_width = max(r.original_image.width for r in valid_results)
            max_img_height = max(r.original_image.height for r in valid_results)
            
            # Scale down if too large
            scale_factor = min(300 / max_img_width, 300 / max_img_height, 1.0)
            img_width = int(max_img_width * scale_factor)
            img_height = int(max_img_height * scale_factor)
            
            # Create grid image (2 columns per result: original + enhanced)
            grid_width = grid_cols * img_width * 2
            grid_height = grid_rows * img_height
            
            grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
            
            # Place images in grid
            for i, result in enumerate(valid_results):
                row = i // grid_cols
                col = i % grid_cols
                
                # Resize images
                orig_resized = result.original_image.resize((img_width, img_height), Image.LANCZOS)
                enh_resized = result.enhanced_image.resize((img_width, img_height), Image.LANCZOS)
                
                # Calculate positions
                orig_x = col * img_width * 2
                orig_y = row * img_height
                enh_x = orig_x + img_width
                enh_y = orig_y
                
                # Paste images
                grid_image.paste(orig_resized, (orig_x, orig_y))
                grid_image.paste(enh_resized, (enh_x, enh_y))
            
            # Save grid
            grid_image.save(output_path, quality=95)
            logger.info(f"Comparison grid saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create comparison grid: {e}")
            return False