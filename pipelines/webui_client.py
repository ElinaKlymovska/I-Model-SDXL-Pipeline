"""
Thin client around Stable Diffusion WebUI HTTP API.
The pipeline assumes WebUI is running inside the pod on port 7860.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import base64
import json
import requests


class WebUI:
    def __init__(self, base_url: str = "http://127.0.0.1:7860") -> None:
        self.base_url = base_url.rstrip("/")

    # --- Helpers ---
    def _post(self, path: str, payload: Dict[str, Any], timeout: int = 600) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        resp = requests.post(url, json=payload, timeout=timeout)
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            # Attach body for easier debugging upstream
            msg = f"{e}; body={resp.text[:1000]}"
            raise requests.HTTPError(msg, response=resp) from None
        return resp.json()

    def _b64_to_bytes(self, b64: str) -> bytes:
        return base64.b64decode(b64)

    # --- Public API ---
    def set_model(self, ckpt_name: str) -> None:
        """Set model by checkpoint filename or title.

        Attempts to resolve to an available model title by checking /sdapi/v1/sd-models.
        """
        try:
            models = requests.get(f"{self.base_url}/sdapi/v1/sd-models", timeout=30).json()
            resolved = None
            # Try exact title match first
            for m in models:
                if m.get("title") == ckpt_name:
                    resolved = m.get("title")
                    break
            if not resolved:
                # Try match by filename substring
                for m in models:
                    if ckpt_name in (m.get("model_name") or "") or ckpt_name in (m.get("filename") or ""):
                        resolved = m.get("title")
                        break
            self._post("/sdapi/v1/options", {"sd_model_checkpoint": resolved or ckpt_name})
        except Exception:
            # Ignore if model switch fails; WebUI may continue with current model
            pass

    def img2img(self, **kwargs: Any) -> Dict[str, Any]:
        return self._post("/sdapi/v1/img2img", kwargs)

    def preprocess_canny(self, image_b64: str, low: int = 100, high: int = 200) -> str:
        try:
            payload = {
                "controlnet_module": "canny",
                "controlnet_input_images": [image_b64],
                "controlnet_processor_res": 512,
                "controlnet_threshold_a": low,
                "controlnet_threshold_b": high,
            }
            res = self._post("/controlnet/detect", payload)
            return res[0]
        except Exception:
            return image_b64

    def preprocess_depth(self, image_b64: str) -> str:
        try:
            payload = {
                "controlnet_module": "depth_midas",
                "controlnet_input_images": [image_b64],
                "controlnet_processor_res": 512,
            }
            res = self._post("/controlnet/detect", payload)
            return res[0]
        except Exception:
            return image_b64

    # --- Discovery ---
    def list_scripts(self) -> Dict[str, Any]:
        try:
            return self._post("/sdapi/v1/scripts", {})
        except Exception:
            return {}

    def has_img2img_script(self, name: str) -> bool:
        scripts = self.list_scripts() or {}
        names = scripts.get("img2img", []) or []
        return name in names


