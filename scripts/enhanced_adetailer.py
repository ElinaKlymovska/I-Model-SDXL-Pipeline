"""
Enhanced ADetailer for Face Correction
Advanced ADetailer integration with SDXL models and configurable face detection.
"""

import os
import sys
import yaml
import argparse
import logging
import requests
import base64
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logging import setup_logging
from core.config_manager import get_config_manager

logger = logging.getLogger(__name__)

class EnhancedADetailer:
    """Enhanced ADetailer with configurable models and settings"""
    
    def __init__(self, webui_url: str = None):
        self.config_manager = get_config_manager()
        self.webui_url = webui_url or self.config_manager.get_webui_url()
        
        # Load configurations through ConfigManager
        self.models_config = self.config_manager.load_models_config()
        self.prompts_config = self.config_manager.load_prompts_config()
        self.pipeline_config = self.config_manager.load_pipeline_config()
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def _save_image_from_base64(self, b64_data: str, output_path: str):
        """Save base64 image data to file"""
        image_data = base64.b64decode(b64_data.split(",", 1)[-1])
        with open(output_path, "wb") as f:
            f.write(image_data)
    
    def get_available_models(self) -> List[str]:
        """Get list of available SDXL models"""
        return self.config_manager.get_available_models()
    
    def get_face_detection_models(self) -> List[str]:
        """Get list of available face detection models"""
        return self.config_manager.get_available_face_models()
    
    def get_model_settings(self, model_name: str) -> Dict[str, Any]:
        """Get recommended settings for a specific model"""
        return self.config_manager.get_model_info(model_name)
    
    def get_prompt_preset(self, preset_name: str) -> Dict[str, Any]:
        """Get prompt preset configuration"""
        return self.config_manager.get_prompt_preset(preset_name)
    
    def get_adetailer_settings(self, preset: str = "balanced") -> Dict[str, Any]:
        """Get ADetailer settings for specified quality preset"""
        return self.config_manager.get_adetailer_settings(preset)
    
    def enhance_face(self, 
                    image_path: str, 
                    output_path: str,
                    model_name: str = None,
                    face_detection_model: str = None,
                    prompt_preset: str = None,
                    enhancement_level: str = None,
                    quality_preset: str = None,
                    custom_prompt: str = "",
                    custom_negative: str = "") -> bool:
        """
        Enhanced face correction with ADetailer
        
        Args:
            image_path: Path to input image
            output_path: Path to save enhanced image
            model_name: SDXL model to use
            face_detection_model: Face detection model
            prompt_preset: Prompt preset name
            enhancement_level: Enhancement strength level
            quality_preset: ADetailer quality preset
            custom_prompt: Custom positive prompt (overrides preset)
            custom_negative: Custom negative prompt (overrides preset)
            
        Returns:
            True if successful, False otherwise
        """
        
        # Get default values from config if not provided
        pipeline_defaults = self.config_manager.get_default_pipeline_config()
        
        model_name = model_name or pipeline_defaults.sdxl_model
        face_detection_model = face_detection_model or pipeline_defaults.face_detection_model
        prompt_preset = prompt_preset or "professional_headshot"
        enhancement_level = enhancement_level or pipeline_defaults.enhancement_level
        quality_preset = quality_preset or pipeline_defaults.quality_preset
        
        logger.info(f"🎭 Starting enhanced face correction: {image_path}")
        logger.info(f"📦 Model: {model_name}")
        logger.info(f"👁️ Face Detection: {face_detection_model}")
        logger.info(f"🎨 Prompt Preset: {prompt_preset}")
        
        # Get model settings
        model_settings = self.get_model_settings(model_name)
        if not model_settings:
            logger.error(f"Model {model_name} not found in configuration")
            return False
        
        # Get prompt configuration
        prompt_config = self.get_prompt_preset(prompt_preset)
        enhancement_config = self.config_manager.get_enhancement_level_settings(enhancement_level)
        adetailer_config = self.get_adetailer_settings(quality_preset)
        
        # Build prompts
        positive_prompt = custom_prompt if custom_prompt else prompt_config.get('positive', '')
        negative_prompt = custom_negative if custom_negative else prompt_config.get('negative', '')
        
        # Add model-specific prompt adjustments
        model_specific = self.prompts_config.get('model_specific', {}).get(model_name, {})
        if model_specific.get('positive_suffix'):
            positive_prompt += f", {model_specific['positive_suffix']}"
        if model_specific.get('negative_suffix'):
            negative_prompt += f", {model_specific['negative_suffix']}"
        
        # Build payload
        payload = {
            "init_images": [self._encode_image(image_path)],
            "prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "denoising_strength": enhancement_config.get('denoising_strength', adetailer_config.get('denoising_strength', 0.4)),
            "sampler_name": model_settings.get('sampler', 'DPM++ 2M Karras'),
            "cfg_scale": enhancement_config.get('cfg_scale', model_settings.get('recommended_cfg', 7)),
            "steps": enhancement_config.get('steps', model_settings.get('recommended_steps', 30)),
            "width": 768,
            "height": 1024,
            "override_settings": {
                "sd_model_checkpoint": model_settings.get('path', model_name)
            },
            "alwayson_scripts": {
                "ADetailer": {
                    "args": [
                        {
                            "ad_model": face_detection_model,
                            "ad_prompt": positive_prompt,
                            "ad_negative_prompt": negative_prompt,
                            "ad_confidence": adetailer_config.get('confidence', 0.3),
                            "ad_denoising_strength": adetailer_config.get('denoising_strength', 0.4),
                            "ad_mask_blur": adetailer_config.get('mask_blur', 8),
                            "ad_mask_k_largest": 1,
                            "ad_mask_min_ratio": 0.05,
                            "ad_mask_max_ratio": 1.0,
                            "ad_x_offset": 0,
                            "ad_y_offset": 0,
                            "ad_dilate_erode": 4,
                            "ad_mask_merge_invert": "None",
                            "ad_mask_padding": adetailer_config.get('mask_padding', 64),
                            "ad_use_inpaint_width_height": False,
                            "ad_inpaint_width": 768,
                            "ad_inpaint_height": 768,
                            "ad_use_steps": True,
                            "ad_steps": enhancement_config.get('steps', 30),
                            "ad_use_cfg_scale": True,
                            "ad_cfg_scale": enhancement_config.get('cfg_scale', 7),
                            "ad_use_sampler": True,
                            "ad_sampler": model_settings.get('sampler', 'DPM++ 2M Karras'),
                            "ad_restore_face": False,
                            "ad_controlnet_model": "None",
                            "ad_controlnet_weight": 1.0
                        }
                    ]
                }
            }
        }
        
        # Add hi-res fix if enabled
        if model_settings.get('hires_fix', False):
            payload.update({
                "enable_hr": True,
                "hr_upscaler": model_settings.get('hires_upscaler', '4x-UltraSharp'),
                "hr_second_pass_steps": model_settings.get('hires_steps', 20),
                "hr_scale": 1.5,
                "denoising_strength": model_settings.get('strength', 0.4)
            })
        
        try:
            logger.info("🚀 Sending request to WebUI API...")
            response = requests.post(
                f"{self.webui_url}/sdapi/v1/img2img", 
                json=payload, 
                timeout=600  # Increased timeout for complex processing
            )
            response.raise_for_status()
            result = response.json()
            
            if "images" in result and result["images"]:
                self._save_image_from_base64(result["images"][0], output_path)
                logger.info(f"✅ Enhanced face saved: {output_path}")
                
                # Save processing info
                info_path = output_path.replace('.jpg', '.json').replace('.png', '.json')
                processing_info = {
                    "model": model_name,
                    "face_detection": face_detection_model,
                    "prompt_preset": prompt_preset,
                    "enhancement_level": enhancement_level,
                    "quality_preset": quality_preset,
                    "positive_prompt": positive_prompt,
                    "negative_prompt": negative_prompt,
                    "settings": {
                        "cfg_scale": payload["cfg_scale"],
                        "steps": payload["steps"],
                        "denoising_strength": payload["denoising_strength"],
                        "sampler": payload["sampler_name"]
                    }
                }
                
                with open(info_path, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(processing_info, f, indent=2, ensure_ascii=False)
                
                return True
            else:
                logger.error(f"❌ No images in response: {result}")
                return False
                
        except requests.exceptions.ConnectionError:
            logger.error("❌ Cannot connect to WebUI API. Make sure Stable Diffusion WebUI is running")
            logger.info("💡 Start WebUI with: python utils/runpod_launcher.py")
            return False
        except requests.exceptions.Timeout:
            logger.error("❌ Request timed out. Complex face enhancement takes time")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            return False


def main():
    """Main CLI interface"""
    # Get default values from config
    config_manager = get_config_manager()
    defaults = config_manager.get_default_pipeline_config()
    
    parser = argparse.ArgumentParser(
        description="Enhanced ADetailer for professional face correction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic face enhancement
  python enhanced_adetailer.py --input photo.jpg --output enhanced.jpg
  
  # Professional headshot with high quality
  python enhanced_adetailer.py --input photo.jpg --output enhanced.jpg \\
    --model copax_realistic_xl --face-model face_yolov8m.pt \\
    --preset professional_headshot --quality aggressive
  
  # Custom artistic portrait
  python enhanced_adetailer.py --input photo.jpg --output enhanced.jpg \\
    --model proteus_xl --preset artistic_portrait \\
    --enhancement strong --custom-prompt "beautiful portrait, artistic lighting"
        """
    )
    
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--model", default=defaults.sdxl_model, 
                       help=f"SDXL model to use (default: {defaults.sdxl_model})")
    parser.add_argument("--face-model", default=defaults.face_detection_model,
                       help=f"Face detection model (default: {defaults.face_detection_model})")
    parser.add_argument("--preset", default="professional_headshot",
                       help="Prompt preset (default: professional_headshot)")
    parser.add_argument("--enhancement", default=defaults.enhancement_level,
                       choices=["light", "medium", "strong", "extreme"],
                       help=f"Enhancement level (default: {defaults.enhancement_level})")
    parser.add_argument("--quality", default=defaults.quality_preset,
                       choices=["conservative", "balanced", "aggressive"],
                       help=f"ADetailer quality preset (default: {defaults.quality_preset})")
    parser.add_argument("--custom-prompt", default="",
                       help="Custom positive prompt (overrides preset)")
    parser.add_argument("--custom-negative", default="",
                       help="Custom negative prompt (overrides preset)")
    webui_url = config_manager.get_webui_url()
    parser.add_argument("--webui-url", default=webui_url,
                       help=f"WebUI API URL (default: {webui_url})")
    parser.add_argument("--list-models", action="store_true",
                       help="List available models and exit")
    parser.add_argument("--list-face-models", action="store_true",
                       help="List available face detection models and exit")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    # Initialize enhancer
    enhancer = EnhancedADetailer(args.webui_url)
    
    # Handle list commands
    if args.list_models:
        models = enhancer.get_available_models()
        print("Available SDXL models:")
        for model in models:
            settings = enhancer.get_model_settings(model)
            speciality = settings.get('speciality', 'general')
            print(f"  {model} - {speciality}")
        return
    
    if args.list_face_models:
        face_models = enhancer.get_face_detection_models()
        print("Available face detection models:")
        for model in face_models:
            print(f"  {model}")
        return
    
    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Run enhancement
    success = enhancer.enhance_face(
        image_path=args.input,
        output_path=args.output,
        model_name=args.model,
        face_detection_model=args.face_model,
        prompt_preset=args.preset,
        enhancement_level=args.enhancement,
        quality_preset=args.quality,
        custom_prompt=args.custom_prompt,
        custom_negative=args.custom_negative
    )
    
    if success:
        logger.info("🎉 Face enhancement completed successfully!")
    else:
        logger.error("❌ Face enhancement failed")
        sys.exit(1)


if __name__ == "__main__":
    main()