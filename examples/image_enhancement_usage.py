#!/usr/bin/env python3
"""
Image Enhancement Pipeline Usage Examples
Demonstrates various ways to use the SDXL + ADetailer pipeline
"""

import os
import sys
from pathlib import Path
from PIL import Image
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from pipelines.inference.image_enhancement import ImageEnhancementPipeline
from runpod.manager import RunPodManager


def basic_enhancement_example():
    """Basic image enhancement example"""
    print("🖼️ Basic Image Enhancement Example")
    print("=" * 40)
    
    # Initialize pipeline
    pipeline = ImageEnhancementPipeline()
    
    # Setup pipeline
    if not pipeline.setup():
        print("❌ Failed to setup pipeline")
        return
    
    # Load model
    model_name = "epicrealism_xl"
    if not pipeline.load_model(model_name):
        print(f"❌ Failed to load model {model_name}")
        return
    
    print(f"✅ Loaded model: {model_name}")
    
    # Enhance image
    input_image_path = "data/input/your_image.jpg"  # Replace with your image path
    
    if os.path.exists(input_image_path):
        results = pipeline.run(
            input_data=input_image_path,
            model_name=model_name,
            output_dir="outputs/enhanced"
        )
        
        if results["status"] == "success":
            print(f"✅ Enhanced {results['total_images']} images")
            print(f"📁 Results saved to: outputs/enhanced")
        else:
            print(f"❌ Enhancement failed: {results.get('error')}")
    else:
        print(f"❌ Input image not found: {input_image_path}")
    
    # Cleanup
    pipeline.cleanup()


def batch_enhancement_example():
    """Batch enhancement of multiple images"""
    print("🔄 Batch Image Enhancement Example")
    print("=" * 40)
    
    # Initialize pipeline
    pipeline = ImageEnhancementPipeline()
    pipeline.setup()
    pipeline.load_model("realvis_xl_lightning")  # Fast model for batch processing
    
    # Get all images from input directory
    input_dir = "data/input"
    image_files = []
    
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
        image_files.extend(Path(input_dir).glob(ext))
    
    if not image_files:
        print(f"❌ No images found in {input_dir}")
        return
    
    print(f"📂 Found {len(image_files)} images to enhance")
    
    # Enhancement configuration for batch processing
    batch_config = {
        "strength": 0.25,  # Lower strength for faster processing
        "guidance_scale": 6.0,
        "num_inference_steps": 8,  # Lightning model optimized for 8 steps
        "face_enhancement_strength": 0.35
    }
    
    # Process batch
    results = pipeline.run(
        input_data=[str(f) for f in image_files],
        model_name="realvis_xl_lightning",
        enhancement_config=batch_config,
        output_dir="outputs/batch_enhanced"
    )
    
    if results["status"] == "success":
        print(f"✅ Batch processing completed!")
        print(f"📊 Successfully enhanced: {results['successful_enhancements']}/{results['total_images']}")
    else:
        print(f"❌ Batch processing failed: {results.get('error')}")
    
    pipeline.cleanup()


def custom_enhancement_presets():
    """Demonstrate different enhancement presets"""
    print("🎨 Custom Enhancement Presets Example")
    print("=" * 40)
    
    # Load model configuration
    with open("configs/models/image_enhancement_models.json", "r") as f:
        model_config = json.load(f)
    
    presets = model_config["enhancement_presets"]
    
    pipeline = ImageEnhancementPipeline()
    pipeline.setup()
    pipeline.load_model("juggernaut_xl")
    
    input_image = "data/input/sample_portrait.jpg"  # Replace with your image
    
    if not os.path.exists(input_image):
        print(f"❌ Sample image not found: {input_image}")
        return
    
    # Test different presets
    for preset_name, preset_config in presets.items():
        print(f"🔄 Testing preset: {preset_name}")
        
        output_dir = f"outputs/presets/{preset_name}"
        
        results = pipeline.run(
            input_data=input_image,
            model_name="juggernaut_xl",
            enhancement_config=preset_config,
            output_dir=output_dir
        )
        
        if results["status"] == "success":
            print(f"✅ {preset_name} preset completed")
        else:
            print(f"❌ {preset_name} preset failed")
    
    pipeline.cleanup()


def runpod_usage_example():
    """Example for RunPod cloud usage"""
    print("☁️ RunPod Cloud Usage Example")
    print("=" * 40)
    
    # This example shows how to use the pipeline on RunPod
    # Assumes you've already deployed using deploy_runpod.py
    
    # Pipeline paths on RunPod
    runpod_paths = {
        "models": "/runpod-volume/models",
        "data": "/runpod-volume/data", 
        "outputs": "/runpod-volume/outputs",
        "cache": "/runpod-volume/cache"
    }
    
    # Initialize pipeline with RunPod paths
    pipeline = ImageEnhancementPipeline()
    
    # Override default paths for RunPod
    pipeline.volume_path = "/runpod-volume"
    pipeline.models_path = runpod_paths["models"]
    pipeline.cache_path = runpod_paths["cache"]
    
    # Set environment variables for RunPod
    os.environ["TRANSFORMERS_CACHE"] = runpod_paths["cache"]
    os.environ["HF_HOME"] = runpod_paths["cache"]
    
    pipeline.setup()
    
    print("🔧 Pipeline configured for RunPod")
    print(f"📁 Models path: {pipeline.models_path}")
    print(f"📁 Cache path: {pipeline.cache_path}")
    
    # Load model (should load from persistent volume)
    if pipeline.load_model("epicrealism_xl"):
        print("✅ Model loaded from RunPod volume")
        
        # Process images from RunPod data directory
        input_data = f"{runpod_paths['data']}/input_images"
        output_dir = f"{runpod_paths['outputs']}/enhanced_results"
        
        print(f"🔄 Processing images from: {input_data}")
        print(f"💾 Saving results to: {output_dir}")
        
        # Note: In actual RunPod usage, you would have uploaded images to the data directory
        print("📝 Note: Upload your images to /runpod-volume/data/ before processing")
    
    pipeline.cleanup()


def face_enhancement_analysis():
    """Analyze face detection and enhancement results"""
    print("👤 Face Enhancement Analysis Example")
    print("=" * 40)
    
    pipeline = ImageEnhancementPipeline()
    pipeline.setup()
    pipeline.load_model("epicrealism_xl")
    
    input_image_path = "data/input/portrait_sample.jpg"  # Replace with your image
    
    if not os.path.exists(input_image_path):
        print(f"❌ Sample image not found: {input_image_path}")
        return
    
    # Load image for analysis
    image = Image.open(input_image_path)
    
    # Detect faces
    faces = pipeline.detect_faces(image)
    
    print(f"👥 Detected {len(faces)} faces in the image")
    
    for i, face in enumerate(faces):
        bbox = face["bbox"]
        confidence = face["confidence"]
        area = face["area"]
        
        print(f"  Face {i+1}:")
        print(f"    📐 Bounding box: {bbox}")
        print(f"    🎯 Confidence: {confidence:.3f}")
        print(f"    📏 Area: {area} pixels")
        print(f"    ✅ Size check: {'PASS' if area >= pipeline.default_config['min_face_size']**2 else 'FAIL'}")
    
    # Run enhancement with detailed results
    results = pipeline.run(
        input_data=image,
        model_name="epicrealism_xl",
        output_dir="outputs/face_analysis"
    )
    
    if results["status"] == "success":
        result = results["results"][0]
        print(f"\\n📊 Enhancement Results:")
        print(f"  👤 Faces processed: {result['faces_detected']}")
        print(f"  🤖 Model used: {result['model_used']}")
        print(f"  ⚙️ Config: {json.dumps(result['config_used'], indent=2)}")
    
    pipeline.cleanup()


def compare_models_example():
    """Compare different SDXL models on the same image"""
    print("⚖️ Model Comparison Example")
    print("=" * 40)
    
    pipeline = ImageEnhancementPipeline()
    pipeline.setup()
    
    input_image = "data/input/comparison_sample.jpg"  # Replace with your image
    
    if not os.path.exists(input_image):
        print(f"❌ Sample image not found: {input_image}")
        return
    
    models_to_compare = ["epicrealism_xl", "realvis_xl_lightning", "juggernaut_xl"]
    
    for model_name in models_to_compare:
        print(f"🔄 Testing model: {model_name}")
        
        if pipeline.load_model(model_name):
            # Get model-specific optimal settings
            model_config = pipeline.available_models[model_name]
            
            enhancement_config = {
                "strength": 0.3,
                "guidance_scale": model_config["optimal_guidance"],
                "num_inference_steps": model_config["optimal_steps"],
                "face_enhancement_strength": 0.4
            }
            
            output_dir = f"outputs/model_comparison/{model_name}"
            
            results = pipeline.run(
                input_data=input_image,
                model_name=model_name,
                enhancement_config=enhancement_config,
                output_dir=output_dir
            )
            
            if results["status"] == "success":
                print(f"✅ {model_name} completed")
                print(f"📁 Results saved to: {output_dir}")
            else:
                print(f"❌ {model_name} failed: {results.get('error')}")
        else:
            print(f"❌ Failed to load {model_name}")
    
    print("\\n📊 Model comparison completed! Check the output folders to compare results.")
    pipeline.cleanup()


def main():
    """Run all examples"""
    print("🚀 Image Enhancement Pipeline Examples")
    print("=" * 50)
    
    examples = [
        ("Basic Enhancement", basic_enhancement_example),
        ("Batch Processing", batch_enhancement_example), 
        ("Enhancement Presets", custom_enhancement_presets),
        ("RunPod Usage", runpod_usage_example),
        ("Face Analysis", face_enhancement_analysis),
        ("Model Comparison", compare_models_example)
    ]
    
    print("Available examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\\nEnter example number to run (1-6), or 'all' to run all examples:")
    choice = input("> ").strip().lower()
    
    if choice == "all":
        for name, func in examples:
            print(f"\\n{'='*20} {name} {'='*20}")
            try:
                func()
            except Exception as e:
                print(f"❌ Example failed: {e}")
            print("\\n")
    
    elif choice.isdigit() and 1 <= int(choice) <= len(examples):
        name, func = examples[int(choice) - 1]
        print(f"\\n{'='*20} {name} {'='*20}")
        try:
            func()
        except Exception as e:
            print(f"❌ Example failed: {e}")
    
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()