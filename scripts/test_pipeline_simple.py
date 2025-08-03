#!/usr/bin/env python3
"""
test_pipeline_simple.py
Simple test script that demonstrates the pipeline without requiring WebUI.
Uses identity metrics to test the evaluation system.
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path to import from scripts
sys.path.append(str(Path(__file__).parent.parent))

from scripts.identity_metrics import IdentityMetrics

def test_identity_evaluation():
    """Test the identity metrics evaluation system."""
    print("🧪 Testing Identity Metrics System")
    print("=" * 50)
    
    # Initialize metrics calculator
    try:
        metrics = IdentityMetrics(device="auto")
        print("✅ Identity metrics initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize identity metrics: {e}")
        return False
    
    # Test with sample images
    input_dir = Path("assets/input")
    output_dir = Path("assets/output")
    output_dir.mkdir(exist_ok=True)
    
    # Get first two images for testing
    test_images = list(input_dir.glob("*.webp"))[:2]
    if len(test_images) < 2:
        print("❌ Need at least 2 test images in assets/input/")
        return False
    
    original_path = str(test_images[0])
    enhanced_path = str(test_images[1])
    
    print(f"📊 Comparing images:")
    print(f"   Original: {original_path}")
    print(f"   Enhanced: {enhanced_path}")
    
    # Run evaluation
    try:
        results = metrics.evaluate_enhancement(
            original_path, 
            enhanced_path,
            save_results=True,
            output_path=str(output_dir / "pipeline_test_results.json")
        )
        
        print("\n📈 Results:")
        print(f"   CLIP Similarity: {results['metrics']['clip_similarity']:.3f}")
        print(f"   Face Similarity: {results['metrics']['face_similarity']:.3f}")
        print(f"   SSIM: {results['metrics']['ssim']:.3f}")
        print(f"   Overall Quality: {results['metrics']['overall_quality']:.3f}")
        print(f"   Assessment: {results['quality_assessment']['level']}")
        
        if results['quality_assessment']['recommendations']:
            print("\n💡 Recommendations:")
            for rec in results['quality_assessment']['recommendations']:
                print(f"   - {rec}")
        
        print(f"\n📄 Detailed results saved to: pipeline_test_results.json")
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False

def test_config_loading():
    """Test configuration file loading."""
    print("\n🔧 Testing Configuration Loading")
    print("=" * 50)
    
    import yaml
    
    # Test models config
    try:
        with open("config/models.yaml", 'r') as f:
            models_config = yaml.safe_load(f)
        print(f"✅ Models config loaded: {len(models_config.get('models', {}))} models found")
        
        # Show first model
        first_model = next(iter(models_config['models'].values()))
        print(f"   Sample model: {first_model['name']}")
        
    except Exception as e:
        print(f"❌ Failed to load models config: {e}")
        return False
    
    # Test prompts config
    try:
        with open("config/prompt_settings.yaml", 'r') as f:
            prompts_config = yaml.safe_load(f)
        print(f"✅ Prompts config loaded")
        
        # Show available prompt types
        if 'character_prompts' in prompts_config:
            prompt_types = list(prompts_config['character_prompts'].keys())
            print(f"   Available prompt types: {', '.join(prompt_types)}")
        
    except Exception as e:
        print(f"❌ Failed to load prompts config: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Test pipeline functionality without WebUI")
    parser.add_argument("--skip-metrics", action="store_true", help="Skip identity metrics test")
    parser.add_argument("--skip-config", action="store_true", help="Skip config loading test")
    
    args = parser.parse_args()
    
    print("🚀 Simple Pipeline Test")
    print("=" * 60)
    print("This test verifies core functionality without requiring WebUI.")
    print()
    
    success = True
    
    if not args.skip_config:
        success &= test_config_loading()
    
    if not args.skip_metrics:
        success &= test_identity_evaluation()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! Core pipeline functionality is working.")
        print("\n🔥 Next steps:")
        print("   1. Run 'python utils/runpod_launcher.py' to set up WebUI")
        print("   2. Use 'python scripts/demo_pipeline.py' for full enhancement")
    else:
        print("❌ Some tests failed. Check the output above for details.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())