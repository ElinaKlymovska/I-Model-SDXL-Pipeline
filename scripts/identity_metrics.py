"""
identity_metrics.py
Identity preservation metrics using CLIP and face recognition models.
Validates that enhanced images maintain facial identity.
"""

import argparse
import logging
import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import json
from pathlib import Path

# Import libraries with fallbacks
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logging.warning("CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    logging.warning("face_recognition not available")

try:
    from transformers import CLIPProcessor, CLIPModel
    TRANSFORMERS_CLIP_AVAILABLE = True
except ImportError:
    TRANSFORMERS_CLIP_AVAILABLE = False

try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IdentityMetrics:
    """
    Comprehensive identity preservation metrics for face enhancement.
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize identity metrics calculator.
        
        Args:
            device: Device to use ('cuda', 'cpu', or 'auto')
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing IdentityMetrics on device: {self.device}")
        
        # Initialize available models
        self.clip_model = None
        self.clip_processor = None
        self.face_app = None
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all available identity comparison models."""
        
        # Initialize CLIP (OpenAI version)
        if CLIP_AVAILABLE:
            try:
                logger.info("Loading CLIP model...")
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
                logger.info("CLIP model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load CLIP: {e}")
                self.clip_model = None
        
        # Initialize CLIP (Transformers version as fallback)
        if not self.clip_model and TRANSFORMERS_CLIP_AVAILABLE:
            try:
                logger.info("Loading CLIP from transformers...")
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_transformers_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
                logger.info("Transformers CLIP model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load transformers CLIP: {e}")
                self.clip_processor = None
        
        # Initialize InsightFace
        if INSIGHTFACE_AVAILABLE:
            try:
                logger.info("Loading InsightFace model...")
                self.face_app = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                self.face_app.prepare(ctx_id=0 if self.device == "cuda" else -1, det_size=(640, 640))
                logger.info("InsightFace model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load InsightFace: {e}")
                self.face_app = None
    
    def compute_clip_similarity(self, image1_path: str, image2_path: str) -> float:
        """
        Compute CLIP cosine similarity between two images.
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            
        Returns:
            Cosine similarity score (0-1)
        """
        if not self.clip_model and not self.clip_processor:
            logger.warning("No CLIP model available")
            return 0.0
        
        try:
            # Load images
            image1 = Image.open(image1_path).convert('RGB')
            image2 = Image.open(image2_path).convert('RGB')
            
            if self.clip_model:
                # Use OpenAI CLIP
                with torch.no_grad():
                    image1_features = self.clip_model.encode_image(
                        self.clip_preprocess(image1).unsqueeze(0).to(self.device)
                    )
                    image2_features = self.clip_model.encode_image(
                        self.clip_preprocess(image2).unsqueeze(0).to(self.device)
                    )
                    
                    # Normalize features
                    image1_features = F.normalize(image1_features, p=2, dim=1)
                    image2_features = F.normalize(image2_features, p=2, dim=1)
                    
                    # Compute cosine similarity
                    similarity = torch.cosine_similarity(image1_features, image2_features).item()
                    
            elif self.clip_processor:
                # Use Transformers CLIP
                inputs1 = self.clip_processor(images=image1, return_tensors="pt").to(self.device)
                inputs2 = self.clip_processor(images=image2, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    image1_features = self.clip_transformers_model.get_image_features(**inputs1)
                    image2_features = self.clip_transformers_model.get_image_features(**inputs2)
                    
                    # Normalize features
                    image1_features = F.normalize(image1_features, p=2, dim=1)
                    image2_features = F.normalize(image2_features, p=2, dim=1)
                    
                    # Compute cosine similarity
                    similarity = torch.cosine_similarity(image1_features, image2_features).item()
            
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.error(f"Error computing CLIP similarity: {e}")
            return 0.0
    
    def compute_face_similarity(self, image1_path: str, image2_path: str, method: str = "auto") -> float:
        """
        Compute face similarity using face recognition models.
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            method: Method to use ('insightface', 'face_recognition', 'auto')
            
        Returns:
            Face similarity score (0-1)
        """
        if method == "auto":
            if self.face_app:
                method = "insightface"
            elif FACE_RECOGNITION_AVAILABLE:
                method = "face_recognition"
            else:
                logger.warning("No face recognition model available")
                return 0.0
        
        try:
            if method == "insightface" and self.face_app:
                return self._compute_insightface_similarity(image1_path, image2_path)
            elif method == "face_recognition" and FACE_RECOGNITION_AVAILABLE:
                return self._compute_face_recognition_similarity(image1_path, image2_path)
            else:
                logger.warning(f"Face recognition method '{method}' not available")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error computing face similarity: {e}")
            return 0.0
    
    def _compute_insightface_similarity(self, image1_path: str, image2_path: str) -> float:
        """Compute face similarity using InsightFace."""
        import cv2
        
        # Load images
        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)
        
        # Get face embeddings
        faces1 = self.face_app.get(img1)
        faces2 = self.face_app.get(img2)
        
        if not faces1 or not faces2:
            logger.warning("No faces detected in one or both images")
            return 0.0
        
        # Use the first (most confident) face from each image
        embedding1 = faces1[0].normed_embedding
        embedding2 = faces2[0].normed_embedding
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2)
        return max(0.0, min(1.0, similarity))
    
    def _compute_face_recognition_similarity(self, image1_path: str, image2_path: str) -> float:
        """Compute face similarity using face_recognition library."""
        
        # Load images
        img1 = face_recognition.load_image_file(image1_path)
        img2 = face_recognition.load_image_file(image2_path)
        
        # Get face encodings
        encodings1 = face_recognition.face_encodings(img1)
        encodings2 = face_recognition.face_encodings(img2)
        
        if not encodings1 or not encodings2:
            logger.warning("No faces detected in one or both images")
            return 0.0
        
        # Compute face distance (lower = more similar)
        face_distance = face_recognition.face_distance([encodings1[0]], encodings2[0])[0]
        
        # Convert distance to similarity (0-1 scale)
        similarity = 1.0 - min(1.0, face_distance)
        return max(0.0, similarity)
    
    def compute_structural_similarity(self, image1_path: str, image2_path: str) -> Dict[str, float]:
        """
        Compute structural similarity metrics (SSIM, PSNR, etc.).
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            
        Returns:
            Dictionary of structural similarity metrics
        """
        try:
            from skimage.metrics import structural_similarity as ssim
            from skimage.metrics import peak_signal_noise_ratio as psnr
            import cv2
            
            # Load images
            img1 = cv2.imread(image1_path)
            img2 = cv2.imread(image2_path)
            
            # Resize to same dimensions if needed
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            # Convert to grayscale for SSIM
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            # Compute metrics
            ssim_score = ssim(gray1, gray2, data_range=gray1.max() - gray1.min())
            psnr_score = psnr(gray1, gray2, data_range=gray1.max() - gray1.min())
            
            # Compute MSE
            mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
            
            return {
                'ssim': ssim_score,
                'psnr': psnr_score,
                'mse': mse
            }
            
        except Exception as e:
            logger.error(f"Error computing structural similarity: {e}")
            return {'ssim': 0.0, 'psnr': 0.0, 'mse': float('inf')}
    
    def evaluate_enhancement(self, original_path: str, enhanced_path: str, 
                           save_results: bool = True, output_path: Optional[str] = None) -> Dict:
        """
        Comprehensive evaluation of face enhancement quality.
        
        Args:
            original_path: Path to original image
            enhanced_path: Path to enhanced image
            save_results: Whether to save results to file
            output_path: Path to save results (auto-generated if None)
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        logger.info(f"Evaluating enhancement: {original_path} -> {enhanced_path}")
        
        results = {
            'original_image': original_path,
            'enhanced_image': enhanced_path,
            'timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else str(__import__('datetime').datetime.now()),
            'metrics': {}
        }
        
        # CLIP similarity
        clip_sim = self.compute_clip_similarity(original_path, enhanced_path)
        results['metrics']['clip_similarity'] = clip_sim
        
        # Face similarity
        face_sim = self.compute_face_similarity(original_path, enhanced_path)
        results['metrics']['face_similarity'] = face_sim
        
        # Structural similarity
        struct_metrics = self.compute_structural_similarity(original_path, enhanced_path)
        results['metrics'].update(struct_metrics)
        
        # Overall quality score (weighted combination)
        quality_score = (
            0.4 * clip_sim +
            0.4 * face_sim +
            0.2 * struct_metrics.get('ssim', 0)
        )
        results['metrics']['overall_quality'] = quality_score
        
        # Quality assessment
        if quality_score >= 0.8:
            quality_level = "Excellent"
        elif quality_score >= 0.6:
            quality_level = "Good"
        elif quality_score >= 0.4:
            quality_level = "Fair"
        else:
            quality_level = "Poor"
        
        results['quality_assessment'] = {
            'score': quality_score,
            'level': quality_level,
            'recommendations': self._generate_recommendations(results['metrics'])
        }
        
        # Save results if requested
        if save_results:
            if output_path is None:
                output_path = os.path.splitext(enhanced_path)[0] + "_metrics.json"
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {output_path}")
        
        return results
    
    def _generate_recommendations(self, metrics: Dict) -> List[str]:
        """Generate improvement recommendations based on metrics."""
        recommendations = []
        
        if metrics.get('clip_similarity', 0) < 0.6:
            recommendations.append("Consider reducing denoising strength to preserve overall appearance")
        
        if metrics.get('face_similarity', 0) < 0.5:
            recommendations.append("Facial identity significantly changed - use more conservative enhancement settings")
        
        if metrics.get('ssim', 0) < 0.7:
            recommendations.append("Structural details lost - try reducing enhancement strength")
        
        if metrics.get('psnr', 0) < 20:
            recommendations.append("High noise or artifacts - check model quality and settings")
        
        if not recommendations:
            recommendations.append("Enhancement quality is good - results are well-preserved")
        
        return recommendations
    
    def batch_evaluate(self, image_pairs: List[Tuple[str, str]], 
                      output_dir: str = "evaluation_results") -> Dict:
        """
        Evaluate multiple image pairs in batch.
        
        Args:
            image_pairs: List of (original, enhanced) image path tuples
            output_dir: Directory to save individual results
            
        Returns:
            Aggregated evaluation results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        all_results = []
        summary_stats = {
            'clip_similarity': [],
            'face_similarity': [],
            'ssim': [],
            'overall_quality': []
        }
        
        for i, (original, enhanced) in enumerate(image_pairs):
            logger.info(f"Evaluating pair {i+1}/{len(image_pairs)}")
            
            result_path = os.path.join(output_dir, f"evaluation_{i:03d}.json")
            result = self.evaluate_enhancement(original, enhanced, True, result_path)
            
            all_results.append(result)
            
            # Collect stats
            metrics = result['metrics']
            summary_stats['clip_similarity'].append(metrics.get('clip_similarity', 0))
            summary_stats['face_similarity'].append(metrics.get('face_similarity', 0))
            summary_stats['ssim'].append(metrics.get('ssim', 0))
            summary_stats['overall_quality'].append(metrics.get('overall_quality', 0))
        
        # Compute summary statistics
        summary = {}
        for metric, values in summary_stats.items():
            summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        # Save summary
        summary_path = os.path.join(output_dir, "summary_statistics.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Batch evaluation complete. Summary saved to: {summary_path}")
        
        return {
            'individual_results': all_results,
            'summary_statistics': summary,
            'total_images': len(image_pairs)
        }


def main():
    """CLI interface for identity metrics evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate face enhancement identity preservation")
    parser.add_argument("--original", required=True, help="Path to original image")
    parser.add_argument("--enhanced", required=True, help="Path to enhanced image")
    parser.add_argument("--output", help="Path to save evaluation results")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto",
                       help="Device to use for computation")
    parser.add_argument("--batch", help="JSON file with image pairs for batch evaluation")
    parser.add_argument("--batch-dir", help="Directory for batch evaluation results")
    
    args = parser.parse_args()
    
    # Initialize metrics calculator
    metrics = IdentityMetrics(device=args.device)
    
    try:
        if args.batch:
            # Batch evaluation
            with open(args.batch, 'r') as f:
                image_pairs = json.load(f)
            
            batch_dir = args.batch_dir or "batch_evaluation_results"
            results = metrics.batch_evaluate(image_pairs, batch_dir)
            
            print(f"Batch evaluation completed for {results['total_images']} image pairs")
            print(f"Average overall quality: {results['summary_statistics']['overall_quality']['mean']:.3f}")
            
        else:
            # Single evaluation
            results = metrics.evaluate_enhancement(
                args.original, 
                args.enhanced, 
                save_results=True, 
                output_path=args.output
            )
            
            print(f"Enhancement Evaluation Results:")
            print(f"CLIP Similarity: {results['metrics']['clip_similarity']:.3f}")
            print(f"Face Similarity: {results['metrics']['face_similarity']:.3f}")
            print(f"SSIM: {results['metrics']['ssim']:.3f}")
            print(f"Overall Quality: {results['metrics']['overall_quality']:.3f}")
            print(f"Assessment: {results['quality_assessment']['level']}")
            
            if results['quality_assessment']['recommendations']:
                print("\nRecommendations:")
                for rec in results['quality_assessment']['recommendations']:
                    print(f"- {rec}")
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import datetime
    try:
        import pandas as pd
    except ImportError:
        pass
    
    exit(main())