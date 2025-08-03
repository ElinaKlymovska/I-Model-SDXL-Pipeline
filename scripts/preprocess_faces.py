"""
preprocess_faces.py
Face detection and preprocessing using InsightFace and face_recognition libraries.
Handles face cropping, alignment, and validation before enhancement.
"""

import argparse
import logging
import os
import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple, Optional, Dict
import json

# Import face detection libraries with fallbacks
try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logging.warning("InsightFace not available, falling back to face_recognition")

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    logging.warning("face_recognition not available")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FacePreprocessor:
    """Face detection and preprocessing pipeline."""
    
    def __init__(self, method: str = "auto", target_size: Tuple[int, int] = (512, 512)):
        """
        Initialize face preprocessor.
        
        Args:
            method: Detection method ('insightface', 'face_recognition', 'auto')
            target_size: Target size for cropped faces (width, height)
        """
        self.method = method
        self.target_size = target_size
        self.face_detector = None
        
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize the face detection model."""
        if self.method == "auto":
            if INSIGHTFACE_AVAILABLE:
                self.method = "insightface"
            elif FACE_RECOGNITION_AVAILABLE:
                self.method = "face_recognition"
            else:
                raise RuntimeError("No face detection library available. Install insightface or face_recognition.")
        
        if self.method == "insightface" and INSIGHTFACE_AVAILABLE:
            logger.info("Initializing InsightFace detector...")
            self.face_detector = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.face_detector.prepare(ctx_id=0, det_size=(640, 640))
        
        elif self.method == "face_recognition" and FACE_RECOGNITION_AVAILABLE:
            logger.info("Using face_recognition library...")
            # face_recognition doesn't need initialization
            pass
        
        else:
            raise RuntimeError(f"Face detection method '{self.method}' not available")
    
    def detect_faces(self, image_path: str) -> List[Dict]:
        """
        Detect faces in an image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of face detection results with bounding boxes and landmarks
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        faces = []
        
        if self.method == "insightface":
            detections = self.face_detector.get(image)
            
            for detection in detections:
                bbox = detection.bbox.astype(int)
                landmarks = detection.kps
                confidence = detection.det_score
                
                faces.append({
                    'bbox': bbox,  # [x1, y1, x2, y2]
                    'landmarks': landmarks,
                    'confidence': confidence,
                    'method': 'insightface'
                })
        
        elif self.method == "face_recognition":
            # Convert BGR to RGB for face_recognition
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect face locations
            face_locations = face_recognition.face_locations(rgb_image, model="hog")
            face_landmarks = face_recognition.face_landmarks(rgb_image, face_locations)
            
            for location, landmarks in zip(face_locations, face_landmarks):
                # Convert from (top, right, bottom, left) to [x1, y1, x2, y2]
                top, right, bottom, left = location
                bbox = [left, top, right, bottom]
                
                faces.append({
                    'bbox': bbox,
                    'landmarks': landmarks,
                    'confidence': 1.0,  # face_recognition doesn't provide confidence
                    'method': 'face_recognition'
                })
        
        logger.info(f"Detected {len(faces)} faces in {image_path}")
        return faces
    
    def crop_face(self, image_path: str, face_data: Dict, padding: float = 0.3) -> np.ndarray:
        """
        Crop and align a face from an image.
        
        Args:
            image_path: Path to input image
            face_data: Face detection data
            padding: Padding around face bbox (0.3 = 30% padding)
            
        Returns:
            Cropped and resized face image as numpy array
        """
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        
        bbox = face_data['bbox']
        x1, y1, x2, y2 = bbox
        
        # Add padding
        face_w, face_h = x2 - x1, y2 - y1
        pad_w, pad_h = int(face_w * padding), int(face_h * padding)
        
        # Expand bbox with padding
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)
        
        # Crop face
        face_crop = image[y1:y2, x1:x2]
        
        # Resize to target size
        face_resized = cv2.resize(face_crop, self.target_size, interpolation=cv2.INTER_LANCZOS4)
        
        return face_resized
    
    def process_image(self, input_path: str, output_dir: str, 
                     min_confidence: float = 0.5, save_metadata: bool = True) -> List[str]:
        """
        Process an image: detect faces, crop, and save results.
        
        Args:
            input_path: Path to input image
            output_dir: Directory to save cropped faces
            min_confidence: Minimum confidence threshold for face detection
            save_metadata: Whether to save detection metadata
            
        Returns:
            List of output file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Detect faces
        faces = self.detect_faces(input_path)
        
        # Filter by confidence
        valid_faces = [f for f in faces if f['confidence'] >= min_confidence]
        
        if not valid_faces:
            logger.warning(f"No valid faces found in {input_path}")
            return []
        
        # Process each face
        output_paths = []
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        
        for i, face_data in enumerate(valid_faces):
            # Crop face
            face_crop = self.crop_face(input_path, face_data)
            
            # Save cropped face
            output_filename = f"{base_name}_face_{i:02d}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            cv2.imwrite(output_path, face_crop)
            output_paths.append(output_path)
            
            logger.info(f"Saved cropped face: {output_path}")
            
            # Save metadata if requested
            if save_metadata:
                metadata = {
                    'original_image': input_path,
                    'face_index': i,
                    'bbox': face_data['bbox'],
                    'confidence': face_data['confidence'],
                    'method': face_data['method'],
                    'target_size': self.target_size
                }
                
                metadata_path = os.path.join(output_dir, f"{base_name}_face_{i:02d}_metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
        
        return output_paths
    
    def batch_process(self, input_dir: str, output_dir: str, 
                     min_confidence: float = 0.5) -> Dict[str, List[str]]:
        """
        Process multiple images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save results
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dictionary mapping input paths to output paths
        """
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Supported image extensions
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        results = {}
        
        for filename in os.listdir(input_dir):
            if any(filename.lower().endswith(ext) for ext in extensions):
                input_path = os.path.join(input_dir, filename)
                
                try:
                    output_paths = self.process_image(input_path, output_dir, min_confidence)
                    results[input_path] = output_paths
                except Exception as e:
                    logger.error(f"Error processing {input_path}: {e}")
                    results[input_path] = []
        
        return results


def main():
    """CLI interface for face preprocessing."""
    parser = argparse.ArgumentParser(description="Face detection and preprocessing")
    parser.add_argument("--input", required=True, help="Input image or directory path")
    parser.add_argument("--output", required=True, help="Output directory for cropped faces")
    parser.add_argument("--method", choices=["insightface", "face_recognition", "auto"], 
                       default="auto", help="Face detection method")
    parser.add_argument("--target-size", nargs=2, type=int, default=[512, 512],
                       help="Target size for cropped faces (width height)")
    parser.add_argument("--min-confidence", type=float, default=0.5,
                       help="Minimum confidence threshold for face detection")
    parser.add_argument("--padding", type=float, default=0.3,
                       help="Padding around face bbox (0.3 = 30%)")
    parser.add_argument("--batch", action="store_true",
                       help="Process all images in input directory")
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = FacePreprocessor(
        method=args.method,
        target_size=tuple(args.target_size)
    )
    
    try:
        if args.batch or os.path.isdir(args.input):
            # Batch processing
            logger.info(f"Batch processing directory: {args.input}")
            results = preprocessor.batch_process(args.input, args.output, args.min_confidence)
            
            total_faces = sum(len(paths) for paths in results.values())
            logger.info(f"Processed {len(results)} images, extracted {total_faces} faces")
            
        else:
            # Single image processing
            logger.info(f"Processing single image: {args.input}")
            output_paths = preprocessor.process_image(args.input, args.output, args.min_confidence)
            logger.info(f"Extracted {len(output_paths)} faces")
    
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())