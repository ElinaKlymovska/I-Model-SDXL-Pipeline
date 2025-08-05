"""
Face Detection Services
Provides pluggable face detection implementations using different backends.
"""

import cv2
import numpy as np
import logging
from typing import List, Optional
from PIL import Image

from core.interfaces import BaseFaceDetector, FaceDetectionResult
from core.config import ConfigManager
from core.exceptions import FaceDetectionError

logger = logging.getLogger(__name__)


class MediaPipeFaceDetector(BaseFaceDetector):
    """Face detection using Google MediaPipe"""
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        self.mp_face_detection = None
        self.face_detector = None
    
    def _setup_detector(self) -> bool:
        """Setup MediaPipe face detector"""
        try:
            import mediapipe as mp
            
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detector = self.mp_face_detection.FaceDetection(
                model_selection=self.face_config.mediapipe_model_selection,
                min_detection_confidence=self.face_config.mediapipe_confidence
            )
            
            logger.info("MediaPipe face detector initialized")
            return True
            
        except ImportError:
            logger.error("MediaPipe not available. Install with: pip install mediapipe")
            return False
        except Exception as e:
            logger.error(f"Failed to setup MediaPipe face detector: {e}")
            return False
    
    def _detect_faces_implementation(self, image: Image.Image) -> List[FaceDetectionResult]:
        """Detect faces using MediaPipe"""
        try:
            # Convert PIL to numpy array
            img_array = np.array(image)
            
            # MediaPipe expects RGB, convert from RGB to BGR for processing
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            results = self.face_detector.process(img_rgb)
            
            faces = []
            if results.detections:
                height, width = img_array.shape[:2]
                
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    
                    # Convert relative coordinates to absolute
                    x = int(bbox.xmin * width)
                    y = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)
                    
                    # Check minimum face size
                    min_size = self.face_config.min_face_size if hasattr(self.face_config, 'min_face_size') else 64
                    if w >= min_size and h >= min_size:
                        faces.append(FaceDetectionResult(
                            bbox=(x, y, w, h),
                            confidence=detection.score[0],
                            area=w * h
                        ))
            
            logger.debug(f"MediaPipe detected {len(faces)} faces")
            return faces
            
        except Exception as e:
            logger.error(f"MediaPipe face detection failed: {e}")
            raise FaceDetectionError(f"MediaPipe detection failed: {e}") from e
    
    def cleanup(self):
        """Cleanup MediaPipe resources"""
        if self.face_detector:
            self.face_detector.close()
            self.face_detector = None
        logger.debug("MediaPipe face detector cleaned up")


class OpenCVFaceDetector(BaseFaceDetector):
    """Face detection using OpenCV Haar Cascades"""
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        self.cascade_classifier = None
    
    def _setup_detector(self) -> bool:
        """Setup OpenCV face detector"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.cascade_classifier = cv2.CascadeClassifier(cascade_path)
            
            if self.cascade_classifier.empty():
                logger.error("Failed to load OpenCV face cascade classifier")
                return False
            
            logger.info("OpenCV face detector initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup OpenCV face detector: {e}")
            return False
    
    def _detect_faces_implementation(self, image: Image.Image) -> List[FaceDetectionResult]:
        """Detect faces using OpenCV"""
        try:
            # Convert PIL to numpy array and grayscale
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Detect faces
            detected_faces = self.cascade_classifier.detectMultiScale(
                gray,
                scaleFactor=self.face_config.opencv_scale_factor,
                minNeighbors=self.face_config.opencv_min_neighbors,
                minSize=self.face_config.opencv_min_size
            )
            
            faces = []
            for (x, y, w, h) in detected_faces:
                faces.append(FaceDetectionResult(
                    bbox=(x, y, w, h),
                    confidence=1.0,  # OpenCV doesn't provide confidence scores
                    area=w * h
                ))
            
            logger.debug(f"OpenCV detected {len(faces)} faces")
            return faces
            
        except Exception as e:
            logger.error(f"OpenCV face detection failed: {e}")
            raise FaceDetectionError(f"OpenCV detection failed: {e}") from e
    
    def cleanup(self):
        """Cleanup OpenCV resources"""
        self.cascade_classifier = None
        logger.debug("OpenCV face detector cleaned up")


class FaceDetectionService:
    """
    Service that manages face detection with fallback strategies.
    Tries MediaPipe first, falls back to OpenCV if needed.
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.primary_detector: Optional[BaseFaceDetector] = None
        self.fallback_detector: Optional[BaseFaceDetector] = None
        self.current_detector: Optional[BaseFaceDetector] = None
    
    def setup(self) -> bool:
        """Setup face detection with primary and fallback detectors"""
        try:
            # Try to setup MediaPipe as primary
            self.primary_detector = MediaPipeFaceDetector(self.config_manager)
            if self.primary_detector.setup():
                self.current_detector = self.primary_detector
                logger.info("Primary detector (MediaPipe) setup successful")
            else:
                logger.warning("Primary detector (MediaPipe) setup failed")
            
            # Setup OpenCV as fallback
            self.fallback_detector = OpenCVFaceDetector(self.config_manager)
            if self.fallback_detector.setup():
                if self.current_detector is None:
                    self.current_detector = self.fallback_detector
                    logger.info("Using fallback detector (OpenCV)")
                else:
                    logger.info("Fallback detector (OpenCV) setup successful")
            else:
                logger.warning("Fallback detector (OpenCV) setup failed")
            
            if self.current_detector is None:
                logger.error("No face detection method available")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Face detection service setup failed: {e}")
            return False
    
    def detect_faces(self, image: Image.Image, use_fallback_on_error: bool = True) -> List[FaceDetectionResult]:
        """
        Detect faces with automatic fallback on errors
        
        Args:
            image: Input image
            use_fallback_on_error: Whether to try fallback detector on primary failure
            
        Returns:
            List of detected faces
        """
        if self.current_detector is None:
            raise FaceDetectionError("Face detection service not setup")
        
        try:
            # Try primary detector
            return self.current_detector.detect_faces(image)
            
        except Exception as e:
            logger.warning(f"Primary face detector failed: {e}")
            
            if use_fallback_on_error and self.fallback_detector and self.fallback_detector != self.current_detector:
                try:
                    logger.info("Attempting fallback face detection")
                    return self.fallback_detector.detect_faces(image)
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback face detector also failed: {fallback_error}")
                    raise FaceDetectionError("Both primary and fallback face detection failed") from fallback_error
            else:
                raise FaceDetectionError(f"Face detection failed: {e}") from e
    
    def get_detector_info(self) -> dict:
        """Get information about available detectors"""
        return {
            "primary_detector": type(self.primary_detector).__name__ if self.primary_detector else None,
            "fallback_detector": type(self.fallback_detector).__name__ if self.fallback_detector else None,
            "current_detector": type(self.current_detector).__name__ if self.current_detector else None,
            "primary_available": self.primary_detector is not None and self.primary_detector.is_setup,
            "fallback_available": self.fallback_detector is not None and self.fallback_detector.is_setup
        }
    
    def switch_to_fallback(self) -> bool:
        """Manually switch to fallback detector"""
        if self.fallback_detector and self.fallback_detector.is_setup:
            self.current_detector = self.fallback_detector
            logger.info("Switched to fallback face detector")
            return True
        return False
    
    def switch_to_primary(self) -> bool:
        """Manually switch to primary detector"""
        if self.primary_detector and self.primary_detector.is_setup:
            self.current_detector = self.primary_detector
            logger.info("Switched to primary face detector")
            return True
        return False
    
    def cleanup(self):
        """Cleanup all detectors"""
        if self.primary_detector:
            self.primary_detector.cleanup()
        if self.fallback_detector:
            self.fallback_detector.cleanup()
        
        self.primary_detector = None
        self.fallback_detector = None
        self.current_detector = None
        
        logger.info("Face detection service cleaned up")