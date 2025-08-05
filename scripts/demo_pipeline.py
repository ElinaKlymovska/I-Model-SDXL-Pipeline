"""
demo_pipeline.py
End-to-end CLI pipeline: ADetailer + SDXL img2img
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from run_adetailer import run_adetailer
    from run_img2img import run_img2img
except ImportError as e:
    logging.error(f"Failed to import pipeline modules: {e}")
    sys.exit(1)

def validate_inputs(input_path, output_path):
    """Validate input parameters and file paths."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    return True

def main(input_path, output_path, temp_path, model, prompt, negative_prompt):
    """Run the complete enhancement pipeline."""
    try:
        # Validate inputs
        validate_inputs(input_path, output_path)
        
        # Ensure temp directory exists
        temp_dir = os.path.dirname(temp_path)
        if temp_dir:
            os.makedirs(temp_dir, exist_ok=True)
        
        logging.info(f"Starting enhancement pipeline for: {input_path}")
        
        # Step 1: ADetailer enhancement
        logging.info("🧠 Step 1: Face Enhancement with ADetailer")
        run_adetailer(input_path, temp_path, model)
        
        if not os.path.exists(temp_path):
            raise RuntimeError("ADetailer failed to generate output")
        
        # Step 2: SDXL img2img stylization
        logging.info("🎨 Step 2: Stylizing with SDXL img2img")
        run_img2img(temp_path, output_path, prompt, negative_prompt, model)
        
        if not os.path.exists(output_path):
            raise RuntimeError("img2img failed to generate output")
        
        # Clean up temp file
        if os.path.exists(temp_path) and temp_path != output_path:
            os.remove(temp_path)
            logging.info(f"Cleaned up temporary file: {temp_path}")
        
        logging.info(f"✅ Pipeline completed! Final output saved at: {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"❌ Pipeline failed: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ADetailer + SDXL img2img pipeline")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path to final output image")
    parser.add_argument("--temp", default="assets/output/temp_adetailer.png", help="Temporary file for ADetailer result")
    parser.add_argument("--model", default="epiCRealismXL", help="Model name (SDXL checkpoint)")
    parser.add_argument("--prompt", required=True, help="Prompt for SDXL img2img")
    parser.add_argument("--negative", default="blurry, deformed, cartoon, extra limbs", help="Negative prompt")

    args = parser.parse_args()
    
    success = main(args.input, args.output, args.temp, args.model, args.prompt, args.negative)
    sys.exit(0 if success else 1)
