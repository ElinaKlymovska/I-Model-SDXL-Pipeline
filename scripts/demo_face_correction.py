"""
Demo Face Correction Pipeline
Demonstrates the new enhanced face correction capabilities.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logging import setup_logging
from scripts.face_correction_pipeline import FaceCorrectionPipeline
from scripts.enhanced_adetailer import EnhancedADetailer

logger = logging.getLogger(__name__)

def demo_single_portrait():
    """Demo: Single portrait enhancement"""
    logger.info("🎭 Demo 1: Single Portrait Enhancement")
    logger.info("="*50)
    
    # Check if input images exist
    input_dir = Path(__file__).parent.parent / "data" / "input"
    
    if not input_dir.exists():
        logger.error("Input directory not found. Please add images to data/input/")
        return False
    
    # Find first image
    image_files = list(input_dir.glob("*.webp")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    
    if not image_files:
        logger.error("No images found in data/input/")
        return False
    
    input_image = str(image_files[0])
    output_dir = Path(__file__).parent.parent / "data" / "outputs" / "demo_face_correction"
    
    logger.info(f"📸 Input: {input_image}")
    logger.info(f"📂 Output: {output_dir}")
    
    # Initialize pipeline
    pipeline = FaceCorrectionPipeline()
    
    # Process with professional settings
    result = pipeline.process_character_face(
        input_path=input_image,
        output_dir=str(output_dir / "professional"),
        character_type="female_portrait",
        model_primary="copax_realistic_xl",
        model_detail="proteus_xl",
        face_model="face_yolov8m.pt",
        enhancement_level="medium",
        quality_preset="balanced",
        save_intermediate=True
    )
    
    if result["success"]:
        logger.info("✅ Professional enhancement completed")
        logger.info(f"📄 Result: {result['final_output']}")
        logger.info(f"⏱️ Time: {result['processing_time']:.1f}s")
    else:
        logger.error("❌ Professional enhancement failed")
        return False
    
    return True

def demo_model_comparison():
    """Demo: Compare different SDXL models"""
    logger.info("\n🔬 Demo 2: Model Comparison")
    logger.info("="*50)
    
    # Input setup
    input_dir = Path(__file__).parent.parent / "data" / "input"
    image_files = list(input_dir.glob("*.webp")) + list(input_dir.glob("*.jpg"))
    
    if not image_files:
        logger.error("No images found for comparison")
        return False
    
    input_image = str(image_files[0])
    output_base = Path(__file__).parent.parent / "data" / "outputs" / "model_comparison"
    
    # Test different models
    models_to_test = [
        ("epicrealism_xl", "Epic Realism XL"),
        ("copax_realistic_xl", "Copax Realistic XL"),
        ("proteus_xl", "Proteus XL"),
        ("newreality_xl", "NewReality XL")
    ]
    
    pipeline = FaceCorrectionPipeline()
    
    for model_key, model_name in models_to_test:
        logger.info(f"🎨 Testing {model_name}...")
        
        result = pipeline.process_character_face(
            input_path=input_image,
            output_dir=str(output_base / model_key),
            character_type="female_portrait",
            model_primary=model_key,
            model_detail=model_key,
            face_model="face_yolov8s.pt",  # Faster for comparison
            enhancement_level="medium",
            quality_preset="balanced",
            skip_preprocessing=True,  # Focus on face correction
            save_intermediate=False
        )
        
        if result["success"]:
            logger.info(f"✅ {model_name}: {result['processing_time']:.1f}s")
        else:
            logger.error(f"❌ {model_name}: Failed")
    
    logger.info(f"📂 Comparison results saved to: {output_base}")
    return True

def demo_enhancement_levels():
    """Demo: Different enhancement strength levels"""
    logger.info("\n⚡ Demo 3: Enhancement Levels")
    logger.info("="*50)
    
    input_dir = Path(__file__).parent.parent / "data" / "input"
    image_files = list(input_dir.glob("*.webp")) + list(input_dir.glob("*.jpg"))
    
    if not image_files:
        logger.error("No images found for enhancement levels demo")
        return False
    
    input_image = str(image_files[0])
    output_base = Path(__file__).parent.parent / "data" / "outputs" / "enhancement_levels"
    
    enhancement_levels = [
        ("light", "Light Enhancement"),
        ("medium", "Medium Enhancement"),
        ("strong", "Strong Enhancement")
    ]
    
    pipeline = FaceCorrectionPipeline()
    
    for level_key, level_name in enhancement_levels:
        logger.info(f"🎚️ Testing {level_name}...")
        
        result = pipeline.process_character_face(
            input_path=input_image,
            output_dir=str(output_base / level_key),
            character_type="female_portrait",
            model_primary="copax_realistic_xl",
            model_detail="proteus_xl",
            face_model="face_yolov8s.pt",
            enhancement_level=level_key,
            quality_preset="balanced",
            skip_preprocessing=True,
            save_intermediate=False
        )
        
        if result["success"]:
            logger.info(f"✅ {level_name}: {result['processing_time']:.1f}s")
        else:
            logger.error(f"❌ {level_name}: Failed")
    
    logger.info(f"📂 Enhancement levels comparison: {output_base}")
    return True

def demo_face_detection_models():
    """Demo: Different face detection models"""
    logger.info("\n👁️ Demo 4: Face Detection Models")
    logger.info("="*50)
    
    input_dir = Path(__file__).parent.parent / "data" / "input"
    image_files = list(input_dir.glob("*.webp")) + list(input_dir.glob("*.jpg"))
    
    if not image_files:
        logger.error("No images found for face detection demo")
        return False
    
    input_image = str(image_files[0])
    output_base = Path(__file__).parent.parent / "data" / "outputs" / "face_detection_comparison"
    
    face_models = [
        ("face_yolov8n.pt", "YOLOv8n (Fast)"),
        ("face_yolov8s.pt", "YOLOv8s (Balanced)"),
        ("face_yolov8m.pt", "YOLOv8m (Accurate)")
    ]
    
    # Use enhanced ADetailer for direct comparison
    enhancer = EnhancedADetailer()
    
    for model_file, model_name in face_models:
        logger.info(f"🔍 Testing {model_name}...")
        
        output_path = output_base / f"{model_file.replace('.pt', '')}.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        success = enhancer.enhance_face(
            image_path=input_image,
            output_path=str(output_path),
            model_name="copax_realistic_xl",
            face_detection_model=model_file,
            prompt_preset="professional_headshot",
            enhancement_level="medium",
            quality_preset="balanced"
        )
        
        if success:
            logger.info(f"✅ {model_name}: Completed")
        else:
            logger.error(f"❌ {model_name}: Failed")
    
    logger.info(f"📂 Face detection comparison: {output_base}")
    return True

def demo_batch_processing():
    """Demo: Batch processing multiple images"""
    logger.info("\n📁 Demo 5: Batch Processing")
    logger.info("="*50)
    
    input_dir = Path(__file__).parent.parent / "data" / "input"
    
    if not input_dir.exists():
        logger.error("Input directory not found")
        return False
    
    output_dir = Path(__file__).parent.parent / "data" / "outputs" / "batch_demo"
    
    pipeline = FaceCorrectionPipeline()
    
    results = pipeline.batch_process_faces(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        character_type="female_portrait",
        model_primary="copax_realistic_xl",
        model_detail="proteus_xl",
        face_model="face_yolov8s.pt",
        enhancement_level="medium",
        quality_preset="balanced",
        skip_preprocessing=False,
        skip_postprocessing=False,
        save_intermediate=True
    )
    
    successful = sum(1 for r in results if r["success"])
    logger.info(f"📊 Batch results: {successful}/{len(results)} successful")
    
    return successful > 0

def run_all_demos():
    """Run all demonstration scenarios"""
    logger.info("🚀 Starting Face Correction Pipeline Demos")
    logger.info("="*60)
    
    demos = [
        ("Single Portrait Enhancement", demo_single_portrait),
        ("Model Comparison", demo_model_comparison),
        ("Enhancement Levels", demo_enhancement_levels),
        ("Face Detection Models", demo_face_detection_models),
        ("Batch Processing", demo_batch_processing)
    ]
    
    results = {}
    
    for demo_name, demo_func in demos:
        try:
            logger.info(f"\n🎯 Running: {demo_name}")
            success = demo_func()
            results[demo_name] = success
            
            if success:
                logger.info(f"✅ {demo_name}: PASSED")
            else:
                logger.error(f"❌ {demo_name}: FAILED")
                
        except Exception as e:
            logger.error(f"❌ {demo_name}: ERROR - {e}")
            results[demo_name] = False
    
    # Summary
    logger.info("\n📊 DEMO RESULTS SUMMARY")
    logger.info("="*60)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for demo_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"  {demo_name}: {status}")
    
    logger.info(f"\n🎯 Overall: {passed}/{total} demos passed")
    
    if passed == total:
        logger.info("🎉 All demos completed successfully!")
    elif passed > 0:
        logger.warning(f"⚠️ {total - passed} demos failed")
    else:
        logger.error("❌ All demos failed - check WebUI connection and model availability")

def main():
    """Main demo interface"""
    parser = argparse.ArgumentParser(
        description="Demonstrate enhanced face correction capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Demos:
  1. single      - Single portrait enhancement
  2. models      - Compare different SDXL models
  3. levels      - Test enhancement strength levels
  4. detection   - Compare face detection models
  5. batch       - Batch processing demonstration
  6. all         - Run all demos (default)

Examples:
  python demo_face_correction.py
  python demo_face_correction.py --demo single
  python demo_face_correction.py --demo models --verbose
        """
    )
    
    parser.add_argument("--demo", default="all",
                       choices=["single", "models", "levels", "detection", "batch", "all"],
                       help="Which demo to run (default: all)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    # Check WebUI availability
    logger.info("🔗 Checking WebUI connection...")
    
    try:
        import requests
        response = requests.get("http://127.0.0.1:7860/", timeout=5)
        logger.info("✅ WebUI is running")
    except:
        logger.error("❌ WebUI not available at http://127.0.0.1:7860")
        logger.info("💡 Start WebUI with: python utils/runpod_launcher.py")
        sys.exit(1)
    
    # Run selected demo
    if args.demo == "all":
        run_all_demos()
    elif args.demo == "single":
        demo_single_portrait()
    elif args.demo == "models":
        demo_model_comparison()
    elif args.demo == "levels":
        demo_enhancement_levels()
    elif args.demo == "detection":
        demo_face_detection_models()
    elif args.demo == "batch":
        demo_batch_processing()

if __name__ == "__main__":
    main()