#!/usr/bin/env python3
"""
Enhanced Image Enhancement Pipeline Usage Examples
Demonstrates the new refactored architecture with service-oriented design.
"""

import os
import sys
from pathlib import Path
import time

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.config import get_config_manager, EnhancementConfig
from core.logging import setup_pipeline_logging, get_pipeline_logger, LoggingContext, get_performance_logger
from core.container import create_default_container
from pipelines.inference.enhanced_image_pipeline import EnhancedImagePipeline


def setup_examples_logging():
    """Setup logging for examples"""
    setup_pipeline_logging(
        base_level="INFO",
        pipeline_level="INFO", 
        services_level="INFO",
        log_dir="logs/examples"
    )


def basic_enhancement_example():
    """Basic image enhancement with the new architecture"""
    logger = get_pipeline_logger("basic_example")
    perf_logger = get_performance_logger("basic")
    
    logger.info("🖼️ Basic Image Enhancement Example")
    logger.info("=" * 40)
    
    try:
        with LoggingContext(logger, "basic_enhancement_setup"):
            # Initialize components
            config_manager = get_config_manager()
            container = create_default_container()
            pipeline = EnhancedImagePipeline(config_manager, container)
            
            # Setup pipeline
            if not pipeline.setup():
                logger.error("Failed to setup pipeline")
                return False
        
        # Load model
        model_name = "epicrealism_xl"
        with LoggingContext(logger, "model_loading", model=model_name):
            if not pipeline.load_model(model_name):
                logger.error(f"Failed to load model {model_name}")
                return False
        
        # Check for sample images
        input_dir = Path("data/input")
        if not input_dir.exists():
            logger.warning("Sample input directory not found. Creating with example...")
            input_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Please place sample images in data/input/ directory")
            return True
        
        # Find sample images
        sample_images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")) + list(input_dir.glob("*.webp"))
        
        if not sample_images:
            logger.warning("No sample images found in data/input/")
            return True
        
        # Enhance first image
        sample_image = sample_images[0]
        logger.info(f"Enhancing sample image: {sample_image}")
        
        start_time = time.time()
        
        results = pipeline.run(
            input_data=str(sample_image),
            model_name=model_name,
            enhancement_config="moderate_enhancement",  # Use preset
            output_dir="outputs/examples/basic",
            create_comparison=True
        )
        
        duration = time.time() - start_time
        perf_logger.log_timing("basic_enhancement", duration, 
                             model=model_name, images=1)
        
        if results["status"] == "success":
            logger.info("✅ Basic enhancement completed successfully!")
            metadata = results["metadata"]
            logger.info(f"📁 Results saved to: outputs/examples/basic")
            logger.info(f"⏱️ Processing time: {metadata['processing_time']:.2f}s")
            logger.info(f"👤 Faces detected: {metadata['total_faces_detected']}")
        else:
            logger.error(f"❌ Enhancement failed: {results.get('error')}")
            return False
        
        # Cleanup
        pipeline.cleanup()
        return True
        
    except Exception as e:
        logger.error(f"Basic enhancement example failed: {e}", exc_info=True)
        return False


def batch_processing_example():
    """Demonstrate batch processing with optimization"""
    logger = get_pipeline_logger("batch_example")
    perf_logger = get_performance_logger("batch")
    
    logger.info("🔄 Batch Processing Example")
    logger.info("=" * 40)
    
    try:
        with LoggingContext(logger, "batch_setup"):
            config_manager = get_config_manager()
            container = create_default_container()
            pipeline = EnhancedImagePipeline(config_manager, container)
            
            if not pipeline.setup():
                logger.error("Failed to setup pipeline")
                return False
        
        # Use fast model for batch processing
        model_name = "realvis_xl_lightning"
        with LoggingContext(logger, "fast_model_loading", model=model_name):
            if not pipeline.load_model(model_name):
                logger.error(f"Failed to load model {model_name}")
                return False
        
        # Find input images
        input_dir = Path("data/input")
        if not input_dir.exists():
            logger.warning("Input directory not found")
            return True
        
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            image_files.extend(input_dir.glob(ext))
        
        if not image_files:
            logger.warning("No images found for batch processing")
            return True
        
        # Limit to first 5 for example
        image_files = image_files[:5]
        logger.info(f"Processing {len(image_files)} images in batch")
        
        # Optimize for batch processing
        with LoggingContext(logger, "batch_optimization"):
            pipeline.optimize_for_batch_processing(len(image_files))
        
        # Create fast batch configuration
        batch_config = EnhancementConfig(
            strength=0.25,  # Lower strength for faster processing
            face_enhancement_strength=0.35,
            guidance_scale=6.0,
            num_inference_steps=8,  # Lightning model optimal
            description="Fast batch processing config"
        )
        
        start_time = time.time()
        
        # Process batch
        with LoggingContext(logger, "batch_processing", 
                          num_images=len(image_files)):
            results = pipeline.run(
                input_data=[str(f) for f in image_files],
                model_name=model_name,
                enhancement_config=batch_config,
                output_dir="outputs/examples/batch",
                create_comparison=True
            )
        
        duration = time.time() - start_time
        
        if results["status"] == "success":
            metadata = results["metadata"]
            success_rate = metadata["successful_enhancements"] / metadata["total_images"]
            
            perf_logger.log_batch_metrics(
                len(image_files), 
                metadata["processing_time"], 
                success_rate
            )
            
            logger.info("✅ Batch processing completed!")
            logger.info(f"📊 Processed: {metadata['total_images']} images")
            logger.info(f"⚡ Throughput: {len(image_files) / duration:.2f} images/second")
            logger.info(f"✨ Success rate: {success_rate:.1%}")
            logger.info(f"👥 Total faces detected: {metadata['total_faces_detected']}")
        else:
            logger.error(f"❌ Batch processing failed: {results.get('error')}")
            return False
        
        pipeline.cleanup()
        return True
        
    except Exception as e:
        logger.error(f"Batch processing example failed: {e}", exc_info=True)
        return False


def custom_configuration_example():
    """Demonstrate custom enhancement configurations"""
    logger = get_pipeline_logger("custom_config_example")
    
    logger.info("⚙️ Custom Configuration Example")
    logger.info("=" * 40)
    
    try:
        config_manager = get_config_manager()
        container = create_default_container()
        pipeline = EnhancedImagePipeline(config_manager, container)
        
        if not pipeline.setup():
            logger.error("Failed to setup pipeline")
            return False
        
        # Load high-quality model
        model_name = "juggernaut_xl"
        if not pipeline.load_model(model_name):
            logger.error(f"Failed to load model {model_name}")
            return False
        
        # Create custom configurations for different scenarios
        configs = {
            "high_quality_portrait": EnhancementConfig(
                strength=0.35,
                face_enhancement_strength=0.5,
                guidance_scale=8.0,
                num_inference_steps=30,
                face_padding=0.15,
                negative_prompt=config_manager.get_negative_prompt("portrait_focused"),
                description="High quality portrait enhancement"
            ),
            
            "restoration": EnhancementConfig(
                strength=0.6,
                face_enhancement_strength=0.7,
                guidance_scale=9.0,
                num_inference_steps=35,
                face_padding=0.2,
                negative_prompt=config_manager.get_negative_prompt("default"),
                description="Heavy restoration for damaged images"
            ),
            
            "subtle_improvement": EnhancementConfig(
                strength=0.2,
                face_enhancement_strength=0.3,
                guidance_scale=6.0,
                num_inference_steps=20,
                face_padding=0.1,
                negative_prompt=config_manager.get_negative_prompt("character_preservation"),
                description="Subtle enhancement preserving original character"
            )
        }
        
        # Find a sample image
        input_dir = Path("data/input")
        if not input_dir.exists():
            logger.warning("Input directory not found")
            return True
        
        sample_images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
        if not sample_images:
            logger.warning("No sample images found")
            return True
        
        sample_image = sample_images[0]
        
        # Test each configuration
        for config_name, config in configs.items():
            logger.info(f"Testing configuration: {config_name}")
            
            with LoggingContext(logger, f"custom_config_{config_name}"):
                results = pipeline.run(
                    input_data=str(sample_image),
                    model_name=model_name,
                    enhancement_config=config,
                    output_dir=f"outputs/examples/custom_configs/{config_name}"
                )
            
            if results["status"] == "success":
                metadata = results["metadata"]
                logger.info(f"  ✅ {config_name}: {metadata['processing_time']:.2f}s, "
                          f"{metadata['total_faces_detected']} faces")
            else:
                logger.error(f"  ❌ {config_name} failed: {results.get('error')}")
        
        pipeline.cleanup()
        return True
        
    except Exception as e:
        logger.error(f"Custom configuration example failed: {e}", exc_info=True)
        return False


def model_comparison_example():
    """Compare different models on the same image"""
    logger = get_pipeline_logger("model_comparison")
    perf_logger = get_performance_logger("comparison")
    
    logger.info("🆚 Model Comparison Example")
    logger.info("=" * 40)
    
    try:
        config_manager = get_config_manager()
        container = create_default_container()
        pipeline = EnhancedImagePipeline(config_manager, container)
        
        if not pipeline.setup():
            logger.error("Failed to setup pipeline")
            return False
        
        # Models to compare
        models_to_test = ["epicrealism_xl", "realvis_xl_lightning", "juggernaut_xl"]
        
        # Find sample image
        input_dir = Path("data/input")
        if not input_dir.exists():
            logger.warning("Input directory not found")
            return True
        
        sample_images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
        if not sample_images:
            logger.warning("No sample images found")
            return True
        
        sample_image = sample_images[0]
        logger.info(f"Comparing models on: {sample_image}")
        
        # Use consistent configuration for fair comparison
        comparison_config = config_manager.get_enhancement_preset("moderate_enhancement")
        
        results_summary = []
        
        for model_name in models_to_test:
            logger.info(f"Testing model: {model_name}")
            
            # Switch to model
            if not pipeline.switch_model(model_name, unload_current=True):
                logger.error(f"Failed to load model {model_name}")
                continue
            
            start_time = time.time()
            
            with LoggingContext(logger, f"model_test_{model_name}"):
                results = pipeline.run(
                    input_data=str(sample_image),
                    model_name=model_name,
                    enhancement_config=comparison_config,
                    output_dir=f"outputs/examples/model_comparison/{model_name}"
                )
            
            if results["status"] == "success":
                metadata = results["metadata"]
                model_time = metadata["processing_time"]
                
                perf_logger.log_timing(f"model_{model_name}", model_time, 
                                     model=model_name, config="moderate")
                
                results_summary.append({
                    "model": model_name,
                    "time": model_time,
                    "faces": metadata["total_faces_detected"],
                    "success": True
                })
                
                logger.info(f"  ✅ {model_name}: {model_time:.2f}s, "
                          f"{metadata['total_faces_detected']} faces")
            else:
                logger.error(f"  ❌ {model_name} failed: {results.get('error')}")
                results_summary.append({
                    "model": model_name,
                    "success": False,
                    "error": results.get("error")
                })
        
        # Display comparison summary
        logger.info("\n📊 Model Comparison Summary:")
        logger.info("-" * 40)
        
        successful_results = [r for r in results_summary if r.get("success")]
        if successful_results:
            # Sort by processing time
            successful_results.sort(key=lambda x: x["time"])
            
            logger.info("⚡ Performance (fastest to slowest):")
            for i, result in enumerate(successful_results):
                rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
                logger.info(f"  {rank} {result['model']}: {result['time']:.2f}s")
        
        pipeline.cleanup()
        return True
        
    except Exception as e:
        logger.error(f"Model comparison example failed: {e}", exc_info=True)
        return False


def resource_monitoring_example():
    """Demonstrate resource monitoring and management"""
    logger = get_pipeline_logger("resource_monitoring")
    
    logger.info("📊 Resource Monitoring Example")
    logger.info("=" * 40)
    
    try:
        config_manager = get_config_manager()
        container = create_default_container()
        pipeline = EnhancedImagePipeline(config_manager, container)
        
        if not pipeline.setup():
            logger.error("Failed to setup pipeline")
            return False
        
        # Get initial system status
        status = pipeline.get_pipeline_status()
        resource_status = status["resource_status"]
        
        logger.info("📈 Initial System Status:")
        logger.info(f"  Device: {resource_status['device_info']['device']}")
        
        if resource_status['device_info']['gpu_available']:
            gpu_info = resource_status['memory']['gpu_memory']
            logger.info(f"  GPU: {resource_status['device_info']['gpu_name']}")
            logger.info(f"  GPU Memory: {gpu_info['allocated']} allocated of {gpu_info['total']}")
        
        sys_mem = resource_status['memory']['system_memory']
        logger.info(f"  System Memory: {sys_mem['used']} used of {sys_mem['total']}")
        
        # Load a model and monitor memory usage
        model_name = "epicrealism_xl"
        logger.info(f"\n⬇️ Loading model: {model_name}")
        
        if pipeline.load_model(model_name):
            # Check memory usage after model loading
            updated_status = pipeline.get_pipeline_status()
            resource_status = updated_status["resource_status"]
            
            if resource_status['device_info']['gpu_available']:
                gpu_info = resource_status['memory']['gpu_memory']
                logger.info(f"  GPU Memory after loading: {gpu_info['allocated']} allocated")
            
            # Perform system health check
            health = pipeline.resource_manager.get_system_health()
            logger.info(f"\n🏥 System Health: {health['status'].upper()}")
            
            if health['warnings']:
                for warning in health['warnings']:
                    logger.warning(f"  ⚠️ {warning}")
            
            if health['recommendations']:
                logger.info("💡 Recommendations:")
                for rec in health['recommendations']:
                    logger.info(f"  • {rec}")
        
        # Demonstrate cleanup and memory optimization
        logger.info("\n🧹 Performing cleanup...")
        cleanup_results = pipeline.resource_manager.cleanup_temporary_files()
        
        if cleanup_results['freed_space'] > 0:
            logger.info(f"  Freed {cleanup_results['freed_space']:.1f} MB of temporary files")
        
        pipeline.resource_manager.cleanup_memory()
        logger.info("  GPU memory cache cleared")
        
        pipeline.cleanup()
        return True
        
    except Exception as e:
        logger.error(f"Resource monitoring example failed: {e}", exc_info=True)
        return False


def run_all_examples():
    """Run all examples in sequence"""
    setup_examples_logging()
    
    logger = get_pipeline_logger("examples_runner")
    logger.info("🚀 Running Enhanced Pipeline Examples")
    logger.info("=" * 50)
    
    examples = [
        ("Basic Enhancement", basic_enhancement_example),
        ("Batch Processing", batch_processing_example),
        ("Custom Configuration", custom_configuration_example),
        ("Model Comparison", model_comparison_example),
        ("Resource Monitoring", resource_monitoring_example)
    ]
    
    results = {}
    
    for name, example_func in examples:
        logger.info(f"\n🎯 Running: {name}")
        try:
            start_time = time.time()
            success = example_func()
            duration = time.time() - start_time
            
            results[name] = {
                "success": success,
                "duration": duration
            }
            
            status = "✅ PASSED" if success else "❌ FAILED"
            logger.info(f"   {status} in {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"   ❌ FAILED with exception: {e}")
            results[name] = {
                "success": False,
                "error": str(e)
            }
    
    # Summary
    logger.info(f"\n📋 Examples Summary:")
    logger.info("=" * 30)
    
    passed = sum(1 for r in results.values() if r.get("success"))
    total = len(results)
    
    logger.info(f"Passed: {passed}/{total}")
    
    for name, result in results.items():
        if result.get("success"):
            logger.info(f"  ✅ {name}: {result.get('duration', 0):.2f}s")
        else:
            error = result.get("error", "Unknown error")
            logger.info(f"  ❌ {name}: {error}")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_examples()
    sys.exit(0 if success else 1)