"""
Face utilities: detection and cropping using insightface (preferred) with
face_recognition as fallback. Outputs square crops expanded by a ratio.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

try:
    import insightface  # type: ignore
    _HAS_INSIGHT = True
except Exception:
    _HAS_INSIGHT = False

try:
    import face_recognition  # type: ignore
    _HAS_FR = True
except Exception:
    _HAS_FR = False

try:
    import cv2  # type: ignore
    _HAS_CV = True
except Exception:
    _HAS_CV = False


@dataclass
class FaceBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


def _detect_insight(img: np.ndarray) -> List[FaceBox]:
    model = insightface.app.FaceAnalysis(name="buffalo_l")
    model.prepare(ctx_id=0)
    faces = model.get(img)
    boxes: List[FaceBox] = []
    for f in faces:
        x1, y1, x2, y2 = [int(v) for v in f.bbox]
        boxes.append(FaceBox(x1, y1, x2, y2))
    return boxes


def _detect_face_recognition(img: np.ndarray) -> List[FaceBox]:
    boxes: List[FaceBox] = []
    locs = face_recognition.face_locations(img)
    for top, right, bottom, left in locs:
        boxes.append(FaceBox(left, top, right, bottom))
    return boxes


def detect_faces(image: Image.Image) -> List[FaceBox]:
    np_img = np.array(image)
    if _HAS_INSIGHT:
        try:
            return _detect_insight(np_img)
        except Exception:
            pass
    if _HAS_FR:
        try:
            return _detect_face_recognition(np_img)
        except Exception:
            pass
    # OpenCV Haar cascade fallback (works without GPU and heavy deps)
    if _HAS_CV:
        try:
            gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            detector = cv2.CascadeClassifier(cascade_path)
            detected = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))
            boxes: List[FaceBox] = []
            for (x, y, w, h) in detected:
                boxes.append(FaceBox(int(x), int(y), int(x + w), int(y + h)))
            return boxes
        except Exception:
            pass
    return []


def expand_square_crop(box: FaceBox, img_w: int, img_h: int, expand_ratio: float) -> FaceBox:
    cx = (box.x1 + box.x2) / 2.0
    cy = (box.y1 + box.y2) / 2.0
    size = max(box.x2 - box.x1, box.y2 - box.y1)
    size = int(size * expand_ratio)

    half = size // 2
    x1 = int(max(0, cx - half))
    y1 = int(max(0, cy - half))
    x2 = int(min(img_w, cx + half))
    y2 = int(min(img_h, cy + half))
    # Adjust to square after clamping
    w = x2 - x1
    h = y2 - y1
    if w != h:
        if w > h:
            diff = w - h
            y1 = max(0, y1 - diff // 2)
            y2 = min(img_h, y1 + w)
        else:
            diff = h - w
            x1 = max(0, x1 - diff // 2)
            x2 = min(img_w, x1 + h)
    return FaceBox(int(x1), int(y1), int(x2), int(y2))


def crop_image_to_box(image: Image.Image, box: FaceBox, target_size: int) -> Image.Image:
    crop = image.crop(box.as_tuple())
    return crop.resize((target_size, target_size), Image.LANCZOS)


