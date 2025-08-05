"""
demo_pipeline.py
Enhanced CLI entrypoint that runs the full face correction pipeline end-to-end.
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logging import setup_logging
from scripts.face_correction_pipeline import FaceCorrectionPipeline
from scripts.enhanced_adetailer import EnhancedADetailer

logger = logging.getLogger(__name__)

def run_single_image_pipeline(args):
    """Run pipeline for single image"""
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return False
    
    pipeline = FaceCorrectionPipeline(args.webui_url)
    
    result = pipeline.process_character_face(
        input_path=args.input,
        output_dir=args.output,
        character_type=args.character,
        model_primary=args.model,
        model_detail=args.detail_model or args.model,
        face_model=args.face_model,
        enhancement_level=args.enhancement,
        quality_preset=args.quality,
        skip_preprocessing=args.skip_preprocessing,
        skip_postprocessing=args.skip_postprocessing,
        save_intermediate=not args.no_intermediate
    )
    
    if result["success"]:
        logger.info("🎉 Pipeline completed successfully!")
        logger.info(f"📂 Final result: {result['final_output']}")
        logger.info(f"⏱️ Processing time: {result['processing_time']:.1f}s")
        logger.info(f"🔧 Stages completed: {', '.join(result['stages_completed'])}")
        return True
    else:
        logger.error("❌ Pipeline failed")
        if "error" in result:
            logger.error(f"Error: {result['error']}")
        return False

def run_batch_pipeline(args):
    """Run pipeline for batch processing"""
    if not os.path.exists(args.input):
        logger.error(f"Input directory not found: {args.input}")
        return False
    
    pipeline = FaceCorrectionPipeline(args.webui_url)
    
    results = pipeline.batch_process_faces(
        input_dir=args.input,
        output_dir=args.output,
        character_type=args.character,
        model_primary=args.model,
        model_detail=args.detail_model or args.model,
        face_model=args.face_model,
        enhancement_level=args.enhancement,
        quality_preset=args.quality,
        skip_preprocessing=args.skip_preprocessing,
        skip_postprocessing=args.skip_postprocessing,
        save_intermediate=not args.no_intermediate
    )
    
    successful = sum(1 for r in results if r["success"])
    total = len(results)
    
    logger.info(f"📊 Batch processing completed: {successful}/{total} successful")
    
    if successful == total:
        logger.info("🎉 All images processed successfully!")
        return True
    elif successful > 0:
        logger.warning(f"⚠️ {total - successful} images failed processing")
        return True
    else:
        logger.error("❌ All images failed processing")
        return False

def run_adetailer_only(args):
    """Run ADetailer only (without full pipeline)"""
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return False
    
    enhancer = EnhancedADetailer(args.webui_url)
    
    output_path = os.path.join(args.output, f"adetailer_{Path(args.input).stem}.jpg")
    os.makedirs(args.output, exist_ok=True)
    
    success = enhancer.enhance_face(
        image_path=args.input,
        output_path=output_path,
        model_name=args.model,
        face_detection_model=args.face_model,
        prompt_preset=args.character if args.character in ['professional_headshot', 'artistic_portrait', 'natural_candid', 'glamour_portrait'] else 'professional_headshot',
        enhancement_level=args.enhancement,
        quality_preset=args.quality,
        custom_prompt=args.custom_prompt,
        custom_negative=args.custom_negative
    )
    
    if success:
        logger.info(f"✅ ADetailer enhancement completed: {output_path}")
        return True
    else:
        logger.error("❌ ADetailer enhancement failed")
        return False

def list_available_options():
    """List available models and options"""
    pipeline = FaceCorrectionPipeline()
    
    print("\n📦 Available SDXL Models:")
    models = pipeline.adetailer.get_available_models()
    for model in models:
        settings = pipeline.adetailer.get_model_settings(model)
        speciality = settings.get('speciality', 'general')
        print(f"  • {model} - {speciality}")
    
    print("\n👁️ Available Face Detection Models:")
    face_models = pipeline.adetailer.get_face_detection_models()
    for model in face_models:
        print(f"  • {model}")
    
    print("\n🎭 Character Types:")
    print("  • female_portrait")
    print("  • male_portrait") 
    print("  • child_portrait")
    
    print("\n⚡ Enhancement Levels:")
    print("  • light - Minimal enhancement")
    print("  • medium - Balanced enhancement")
    print("  • strong - Significant enhancement")
    print("  • extreme - Maximum enhancement")
    
    print("\n🎨 Quality Presets:")
    print("  • conservative - Safe, minimal changes")
    print("  • balanced - Good balance of quality/speed")
    print("  • aggressive - Maximum quality")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Enhanced face correction pipeline with ADetailer + SDXL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image with default settings
  python demo_pipeline.py --input photo.jpg --output ./results/
  
  # Professional female portrait
  python demo_pipeline.py --input photo.jpg --output ./results/ \\
    --character female_portrait --model copax_realistic_xl \\
    --enhancement medium --quality balanced
  
  # Batch processing with high quality
  python demo_pipeline.py --batch ./photos/ --output ./results/ \\
    --model proteus_xl --detail-model newreality_xl \\
    --enhancement strong --quality aggressive
  
  # ADetailer only (fast processing)
  python demo_pipeline.py --input photo.jpg --output ./results/ \\
    --adetailer-only --model epicrealism_xl
        """
    )
    
    # Input/Output
    parser.add_argument("--input", help="Input image path")
    parser.add_argument("--batch", help="Input directory for batch processing")
    parser.add_argument("--output", help="Output directory")
    
    # Processing modes
    parser.add_argument("--adetailer-only", action="store_true",
                       help="Run ADetailer only (skip pre/post processing)")
    
    # Model settings
    parser.add_argument("--model", default="copax_realistic_xl",
                       help="Primary SDXL model (default: copax_realistic_xl)")
    parser.add_argument("--detail-model", 
                       help="Detail refinement model (defaults to primary model)")
    parser.add_argument("--face-model", default="face_yolov8m.pt",
                       help="Face detection model (default: face_yolov8m.pt)")
    
    # Enhancement settings
    parser.add_argument("--character", default="female_portrait",
                       choices=["female_portrait", "male_portrait", "child_portrait"],
                       help="Character type (default: female_portrait)")
    parser.add_argument("--enhancement", default="medium",
                       choices=["light", "medium", "strong", "extreme"],
                       help="Enhancement level (default: medium)")
    parser.add_argument("--quality", default="balanced",
                       choices=["conservative", "balanced", "aggressive"],
                       help="Quality preset (default: balanced)")
    
    # Custom prompts
    parser.add_argument("--custom-prompt", default="",
                       help="Custom positive prompt")
    parser.add_argument("--custom-negative", default="",
                       help="Custom negative prompt")
    
    # Pipeline options
    parser.add_argument("--skip-preprocessing", action="store_true",
                       help="Skip initial image enhancement")
    parser.add_argument("--skip-postprocessing", action="store_true",
                       help="Skip final refinement")
    parser.add_argument("--no-intermediate", action="store_true",
                       help="Don't save intermediate results")
    
    # System settings
    parser.add_argument("--webui-url", default="http://127.0.0.1:7860",
                       help="WebUI API URL (default: http://127.0.0.1:7860)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    # Information commands
    parser.add_argument("--list-options", action="store_true",
                       help="List available models and options")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    # Handle information commands
    if args.list_options:
        list_available_options()
        return
    
    # Validate input arguments
    if not args.input and not args.batch:
        parser.error("Either --input or --batch must be specified")
    
    if args.input and args.batch:
        parser.error("Cannot specify both --input and --batch")
    
    if not args.output:
        parser.error("--output directory must be specified")
    
    # Check WebUI availability
    logger.info("🔗 Checking WebUI connection...")
    try:
        import requests
        response = requests.get(args.webui_url, timeout=5)
        logger.info("✅ WebUI is running")
    except Exception as e:
        logger.error(f"❌ WebUI not available at {args.webui_url}")
        logger.info("💡 Start WebUI with: python utils/runpod_launcher.py")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Run appropriate processing mode
    success = False
    
    if args.adetailer_only:
        if args.batch:
            logger.error("ADetailer-only mode doesn't support batch processing")
            sys.exit(1)
        success = run_adetailer_only(args)
    elif args.batch:
        success = run_batch_pipeline(args)
    else:
        success = run_single_image_pipeline(args)
    
    if success:
        logger.info("🎉 Processing completed successfully!")
    else:
        logger.error("❌ Processing failed")
        sys.exit(1)

if __name__ == "__main__":
    main()