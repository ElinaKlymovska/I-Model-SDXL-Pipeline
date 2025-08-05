"""
Face Correction Pipeline
Integrated ADetailer + Img2Img pipeline for comprehensive character face enhancement.
"""

import os
import sys
import yaml
import argparse
import logging
import requests
import base64
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image
import time

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logging import setup_logging
from scripts.enhanced_adetailer import EnhancedADetailer

logger = logging.getLogger(__name__)

class FaceCorrectionPipeline:
    """
    Comprehensive face correction pipeline that combines:
    1. Initial image enhancement with Img2Img
    2. Advanced face detection and correction with ADetailer
    3. Optional final refinement pass
    """
    
    def __init__(self, webui_url: str = "http://127.0.0.1:7860"):
        self.webui_url = webui_url
        self.adetailer = EnhancedADetailer(webui_url)
        self.models_config = self._load_models_config()
        self.prompts_config = self._load_prompts_config()
        
    def _load_models_config(self) -> Dict[str, Any]:
        """Load models configuration"""
        config_path = Path(__file__).parent.parent / "config" / "models.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}
    
    def _load_prompts_config(self) -> Dict[str, Any]:
        """Load prompts configuration"""
        config_path = Path(__file__).parent.parent / "config" / "prompt_settings.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def _save_image_from_base64(self, b64_data: str, output_path: str):
        """Save base64 image data to file"""
        image_data = base64.b64decode(b64_data.split(",", 1)[-1])
        with open(output_path, "wb") as f:
            f.write(image_data)
    
    def _img2img_pass(self, 
                     image_path: str, 
                     output_path: str,
                     model_name: str,
                     positive_prompt: str,
                     negative_prompt: str,
                     denoising_strength: float = 0.3,
                     step_name: str = "img2img") -> bool:
        """
        Perform Img2Img processing
        
        Args:
            image_path: Input image path
            output_path: Output image path
            model_name: SDXL model to use
            positive_prompt: Positive prompt
            negative_prompt: Negative prompt
            denoising_strength: Denoising strength
            step_name: Name of the processing step for logging
            
        Returns:
            True if successful, False otherwise
        """
        
        logger.info(f"🎨 Running {step_name} pass: {image_path}")
        
        # Get model settings
        model_settings = self.adetailer.get_model_settings(model_name)
        if not model_settings:
            logger.error(f"Model {model_name} not found in configuration")
            return False
        
        payload = {
            "init_images": [self._encode_image(image_path)],
            "prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "denoising_strength": denoising_strength,
            "sampler_name": model_settings.get('sampler', 'DPM++ 2M Karras'),
            "cfg_scale": model_settings.get('recommended_cfg', 7),
            "steps": model_settings.get('recommended_steps', 30),
            "width": 768,
            "height": 1024,
            "override_settings": {
                "sd_model_checkpoint": model_settings.get('path', model_name)
            }
        }
        
        # Add hi-res fix if enabled for final passes
        if model_settings.get('hires_fix', False) and denoising_strength <= 0.4:
            payload.update({
                "enable_hr": True,
                "hr_upscaler": model_settings.get('hires_upscaler', '4x-UltraSharp'),
                "hr_second_pass_steps": model_settings.get('hires_steps', 20),
                "hr_scale": 1.5
            })
        
        try:
            response = requests.post(
                f"{self.webui_url}/sdapi/v1/img2img", 
                json=payload, 
                timeout=300
            )
            response.raise_for_status()
            result = response.json()
            
            if "images" in result and result["images"]:
                self._save_image_from_base64(result["images"][0], output_path)
                logger.info(f"✅ {step_name} completed: {output_path}")
                return True
            else:
                logger.error(f"❌ {step_name} failed: {result}")
                return False
                
        except Exception as e:
            logger.error(f"❌ {step_name} error: {e}")
            return False
    
    def process_character_face(self,
                              input_path: str,
                              output_dir: str,
                              character_type: str = "female_portrait",
                              model_primary: str = "copax_realistic_xl",
                              model_detail: str = "proteus_xl",
                              face_model: str = "face_yolov8m.pt",
                              enhancement_level: str = "medium",
                              quality_preset: str = "balanced",
                              skip_preprocessing: bool = False,
                              skip_postprocessing: bool = False,
                              save_intermediate: bool = True) -> Dict[str, Any]:
        """
        Complete character face correction pipeline
        
        Args:
            input_path: Input image path
            output_dir: Output directory for results
            character_type: Character type for prompts (female_portrait, male_portrait, etc.)
            model_primary: Primary SDXL model for initial enhancement
            model_detail: Detail model for face refinement
            face_model: Face detection model
            enhancement_level: Enhancement strength level
            quality_preset: ADetailer quality preset
            skip_preprocessing: Skip initial Img2Img pass
            skip_postprocessing: Skip final refinement pass
            save_intermediate: Save intermediate results
            
        Returns:
            Dictionary with processing results and file paths
        """
        
        logger.info("🚀 Starting comprehensive face correction pipeline")
        logger.info(f"📸 Input: {input_path}")
        logger.info(f"👤 Character Type: {character_type}")
        logger.info(f"🎯 Primary Model: {model_primary}")
        logger.info(f"🔍 Detail Model: {model_detail}")
        logger.info(f"👁️ Face Detection: {face_model}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Base filename for outputs
        input_name = Path(input_path).stem
        
        # File paths for pipeline stages
        stage1_path = os.path.join(output_dir, f"{input_name}_01_preprocessed.jpg")
        stage2_path = os.path.join(output_dir, f"{input_name}_02_face_corrected.jpg") 
        stage3_path = os.path.join(output_dir, f"{input_name}_03_final_refined.jpg")
        final_path = os.path.join(output_dir, f"{input_name}_enhanced_final.jpg")
        
        results = {
            "success": False,
            "input_path": input_path,
            "output_dir": output_dir,
            "stages_completed": [],
            "final_output": None,
            "processing_time": 0,
            "settings_used": {
                "character_type": character_type,
                "model_primary": model_primary,
                "model_detail": model_detail,
                "face_model": face_model,
                "enhancement_level": enhancement_level,
                "quality_preset": quality_preset
            }
        }
        
        start_time = time.time()
        
        try:
            # Get character-specific prompts
            character_prompts = self.prompts_config.get('character_prompts', {}).get(character_type, {})
            base_positive = character_prompts.get('positive', '')
            base_negative = character_prompts.get('negative', '')
            
            # Enhancement level settings
            enhancement_config = self.prompts_config.get('enhancement_levels', {}).get(enhancement_level, {})
            
            current_image = input_path
            
            # Stage 1: Preprocessing with Img2Img (optional)
            if not skip_preprocessing:
                logger.info("📝 Stage 1: Image preprocessing with Img2Img")
                
                preprocess_prompt = base_positive + ", " + self.prompts_config.get('default_prompts', {}).get('base_positive', '')
                preprocess_negative = base_negative
                
                # Add model-specific adjustments
                model_specific = self.prompts_config.get('model_specific', {}).get(model_primary, {})
                if model_specific.get('positive_suffix'):
                    preprocess_prompt += f", {model_specific['positive_suffix']}"
                if model_specific.get('negative_suffix'):
                    preprocess_negative += f", {model_specific['negative_suffix']}"
                
                success = self._img2img_pass(
                    image_path=current_image,
                    output_path=stage1_path,
                    model_name=model_primary,
                    positive_prompt=preprocess_prompt,
                    negative_prompt=preprocess_negative,
                    denoising_strength=0.25,  # Light preprocessing
                    step_name="preprocessing"
                )
                
                if success:
                    results["stages_completed"].append("preprocessing")
                    current_image = stage1_path if save_intermediate else current_image
                    logger.info("✅ Stage 1 completed")
                else:
                    logger.warning("⚠️ Stage 1 failed, continuing with original image")
            
            # Stage 2: Face correction with ADetailer (main stage)
            logger.info("🎭 Stage 2: Advanced face correction with ADetailer")
            
            success = self.adetailer.enhance_face(
                image_path=current_image,
                output_path=stage2_path,
                model_name=model_detail,
                face_detection_model=face_model,
                prompt_preset=character_type if character_type in self.prompts_config.get('presets', {}) else 'professional_headshot',
                enhancement_level=enhancement_level,
                quality_preset=quality_preset,
                custom_prompt="",
                custom_negative=""
            )
            
            if success:
                results["stages_completed"].append("face_correction")
                current_image = stage2_path
                logger.info("✅ Stage 2 completed")
            else:
                logger.error("❌ Stage 2 failed - face correction unsuccessful")
                return results
            
            # Stage 3: Final refinement (optional)
            if not skip_postprocessing:
                logger.info("✨ Stage 3: Final refinement and polish")
                
                # Use lighter denoising for final polish
                refine_prompt = base_positive + ", refined details, professional photography, high quality"
                refine_negative = base_negative + ", artifacts, imperfections"
                
                # Add quality enhancement terms
                quality_config = self.prompts_config.get('quality_settings', {}).get('high_quality', {})
                if quality_config.get('positive_suffix'):
                    refine_prompt += f", {quality_config['positive_suffix']}"
                
                success = self._img2img_pass(
                    image_path=current_image,
                    output_path=stage3_path,
                    model_name=model_primary,
                    positive_prompt=refine_prompt,
                    negative_prompt=refine_negative,
                    denoising_strength=0.15,  # Very light final polish
                    step_name="final_refinement"
                )
                
                if success:
                    results["stages_completed"].append("final_refinement")
                    current_image = stage3_path
                    logger.info("✅ Stage 3 completed")
                else:
                    logger.warning("⚠️ Stage 3 failed, using Stage 2 result")
            
            # Copy final result
            if current_image != final_path:
                import shutil
                shutil.copy2(current_image, final_path)
            
            results["success"] = True
            results["final_output"] = final_path
            results["processing_time"] = time.time() - start_time
            
            # Save processing report
            report_path = os.path.join(output_dir, f"{input_name}_processing_report.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"🎉 Pipeline completed successfully in {results['processing_time']:.1f}s")
            logger.info(f"📂 Final result: {final_path}")
            
            # Clean up intermediate files if not saving them
            if not save_intermediate:
                for stage_path in [stage1_path, stage2_path, stage3_path]:
                    if os.path.exists(stage_path) and stage_path != final_path:
                        os.remove(stage_path)
                        logger.debug(f"🗑️ Cleaned up: {stage_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed with error: {e}")
            results["error"] = str(e)
            results["processing_time"] = time.time() - start_time
            return results
    
    def batch_process_faces(self,
                           input_dir: str,
                           output_dir: str,
                           **kwargs) -> List[Dict[str, Any]]:
        """
        Batch process multiple images
        
        Args:
            input_dir: Directory containing input images
            output_dir: Base output directory
            **kwargs: Arguments passed to process_character_face
            
        Returns:
            List of processing results for each image
        """
        
        logger.info(f"📁 Starting batch face correction: {input_dir}")
        
        # Find image files
        input_path = Path(input_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions and f.is_file()]
        
        if not image_files:
            logger.error(f"No image files found in {input_dir}")
            return []
        
        logger.info(f"🔍 Found {len(image_files)} images to process")
        
        results = []
        
        for i, image_file in enumerate(image_files, 1):
            logger.info(f"\n📷 Processing {i}/{len(image_files)}: {image_file.name}")
            
            # Create individual output directory
            image_output_dir = os.path.join(output_dir, f"image_{i:02d}_{image_file.stem}")
            
            # Process single image
            result = self.process_character_face(
                input_path=str(image_file),
                output_dir=image_output_dir,
                **kwargs
            )
            
            result["batch_index"] = i
            result["batch_total"] = len(image_files)
            results.append(result)
            
            if result["success"]:
                logger.info(f"✅ {i}/{len(image_files)} completed")
            else:
                logger.error(f"❌ {i}/{len(image_files)} failed")
        
        # Save batch summary
        batch_summary = {
            "total_images": len(image_files),
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
            "total_time": sum(r.get("processing_time", 0) for r in results),
            "results": results
        }
        
        summary_path = os.path.join(output_dir, "batch_processing_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(batch_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n🎯 Batch processing completed:")
        logger.info(f"   ✅ Successful: {batch_summary['successful']}")
        logger.info(f"   ❌ Failed: {batch_summary['failed']}")
        logger.info(f"   ⏱️ Total time: {batch_summary['total_time']:.1f}s")
        logger.info(f"   📊 Summary: {summary_path}")
        
        return results


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Comprehensive face correction pipeline with ADetailer + Img2Img",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image with default settings
  python face_correction_pipeline.py --input photo.jpg --output ./results/
  
  # Female portrait with high quality settings
  python face_correction_pipeline.py --input photo.jpg --output ./results/ \\
    --character female_portrait --model-primary copax_realistic_xl \\
    --model-detail proteus_xl --quality aggressive
  
  # Batch processing directory
  python face_correction_pipeline.py --batch ./input_photos/ --output ./results/ \\
    --character male_portrait --enhancement strong
  
  # Quick processing (skip pre/post)
  python face_correction_pipeline.py --input photo.jpg --output ./results/ \\
    --skip-preprocessing --skip-postprocessing
        """
    )
    
    # Input/Output
    parser.add_argument("--input", help="Input image path (for single image)")
    parser.add_argument("--batch", help="Input directory path (for batch processing)")
    parser.add_argument("--output", required=True, help="Output directory")
    
    # Model settings
    parser.add_argument("--character", default="female_portrait",
                       choices=["female_portrait", "male_portrait", "child_portrait"],
                       help="Character type for prompts (default: female_portrait)")
    parser.add_argument("--model-primary", default="copax_realistic_xl",
                       help="Primary SDXL model (default: copax_realistic_xl)")
    parser.add_argument("--model-detail", default="proteus_xl",
                       help="Detail refinement model (default: proteus_xl)")
    parser.add_argument("--face-model", default="face_yolov8m.pt",
                       help="Face detection model (default: face_yolov8m.pt)")
    
    # Enhancement settings
    parser.add_argument("--enhancement", default="medium",
                       choices=["light", "medium", "strong", "extreme"],
                       help="Enhancement level (default: medium)")
    parser.add_argument("--quality", default="balanced",
                       choices=["conservative", "balanced", "aggressive"],
                       help="ADetailer quality preset (default: balanced)")
    
    # Pipeline options
    parser.add_argument("--skip-preprocessing", action="store_true",
                       help="Skip initial Img2Img preprocessing")
    parser.add_argument("--skip-postprocessing", action="store_true",
                       help="Skip final refinement pass")
    parser.add_argument("--no-intermediate", action="store_true",
                       help="Don't save intermediate results")
    
    # System settings
    parser.add_argument("--webui-url", default="http://127.0.0.1:7860",
                       help="WebUI API URL (default: http://127.0.0.1:7860)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.input and not args.batch:
        parser.error("Either --input or --batch must be specified")
    
    if args.input and args.batch:
        parser.error("Cannot specify both --input and --batch")
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    # Initialize pipeline
    pipeline = FaceCorrectionPipeline(args.webui_url)
    
    # Process images
    if args.input:
        # Single image processing
        if not os.path.exists(args.input):
            logger.error(f"Input file not found: {args.input}")
            sys.exit(1)
        
        result = pipeline.process_character_face(
            input_path=args.input,
            output_dir=args.output,
            character_type=args.character,
            model_primary=args.model_primary,
            model_detail=args.model_detail,
            face_model=args.face_model,
            enhancement_level=args.enhancement,
            quality_preset=args.quality,
            skip_preprocessing=args.skip_preprocessing,
            skip_postprocessing=args.skip_postprocessing,
            save_intermediate=not args.no_intermediate
        )
        
        if result["success"]:
            logger.info("🎉 Single image processing completed successfully!")
        else:
            logger.error("❌ Single image processing failed")
            sys.exit(1)
    
    else:
        # Batch processing
        if not os.path.exists(args.batch):
            logger.error(f"Batch directory not found: {args.batch}")
            sys.exit(1)
        
        results = pipeline.batch_process_faces(
            input_dir=args.batch,
            output_dir=args.output,
            character_type=args.character,
            model_primary=args.model_primary,
            model_detail=args.model_detail,
            face_model=args.face_model,
            enhancement_level=args.enhancement,
            quality_preset=args.quality,
            skip_preprocessing=args.skip_preprocessing,
            skip_postprocessing=args.skip_postprocessing,
            save_intermediate=not args.no_intermediate
        )
        
        successful_count = sum(1 for r in results if r["success"])
        if successful_count == len(results):
            logger.info("🎉 Batch processing completed successfully!")
        elif successful_count > 0:
            logger.warning(f"⚠️ Batch processing completed with {len(results) - successful_count} failures")
        else:
            logger.error("❌ Batch processing failed completely")
            sys.exit(1)


if __name__ == "__main__":
    main()