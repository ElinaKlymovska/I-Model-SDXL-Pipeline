"""
Resource Management Service
Handles system resources like memory, storage, and device management.
"""

import os
import psutil
import torch
import logging
import shutil
from typing import Dict, Any, Optional
from pathlib import Path

from core.config import ConfigManager
from core.exceptions import ResourceError

logger = logging.getLogger(__name__)


class ResourceManager:
    """
    Service for managing system resources including GPU memory, storage, and monitoring.
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.runpod_config = config_manager.get_runpod_config()
        
        # Device information
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_available = torch.cuda.is_available()
        
        if self.gpu_available:
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_total_memory = torch.cuda.get_device_properties(0).total_memory
        else:
            self.gpu_name = None
            self.gpu_total_memory = 0
        
        logger.info(f"ResourceManager initialized - Device: {self.device}")
        if self.gpu_available:
            logger.info(f"GPU: {self.gpu_name}, Total VRAM: {self.gpu_total_memory / 1024**3:.1f} GB")
    
    def check_available_memory(self) -> Dict[str, Any]:
        """
        Check available system and GPU memory
        
        Returns:
            Dictionary with memory information
        """
        memory_info = {
            "device": self.device,
            "system_memory": self._get_system_memory_info(),
            "gpu_memory": self._get_gpu_memory_info() if self.gpu_available else None
        }
        
        return memory_info
    
    def _get_system_memory_info(self) -> Dict[str, str]:
        """Get system RAM information"""
        memory = psutil.virtual_memory()
        
        return {
            "total": f"{memory.total / 1024**3:.1f} GB",
            "available": f"{memory.available / 1024**3:.1f} GB",
            "used": f"{memory.used / 1024**3:.1f} GB",
            "percentage": f"{memory.percent:.1f}%"
        }
    
    def _get_gpu_memory_info(self) -> Dict[str, str]:
        """Get GPU memory information"""
        if not self.gpu_available:
            return {"error": "GPU not available"}
        
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        total = self.gpu_total_memory / 1024**3
        free = total - allocated
        
        return {
            "gpu_name": self.gpu_name,
            "total": f"{total:.1f} GB",
            "allocated": f"{allocated:.1f} GB",
            "cached": f"{cached:.1f} GB",
            "free": f"{free:.1f} GB",
            "utilization": f"{(allocated / total) * 100:.1f}%"
        }
    
    def cleanup_memory(self) -> None:
        """Free up memory resources"""
        if self.gpu_available:
            torch.cuda.empty_cache()
            logger.info("GPU memory cache cleared")
        
        # Force garbage collection
        import gc
        gc.collect()
        logger.debug("Garbage collection performed")
    
    def ensure_storage_space(self, required_gb: float, path: Optional[str] = None) -> bool:
        """
        Ensure sufficient storage space is available
        
        Args:
            required_gb: Required space in GB
            path: Path to check (defaults to models path)
            
        Returns:
            True if sufficient space is available
        """
        check_path = path or self.runpod_config.models_path
        
        try:
            # Create directory if it doesn't exist
            Path(check_path).mkdir(parents=True, exist_ok=True)
            
            # Check available space
            available_space = shutil.disk_usage(check_path).free / 1024**3
            
            logger.debug(f"Available space at {check_path}: {available_space:.1f} GB")
            logger.debug(f"Required space: {required_gb} GB")
            
            if available_space < required_gb:
                logger.error(f"Insufficient storage space. Required: {required_gb} GB, Available: {available_space:.1f} GB")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check storage space: {e}")
            return False
    
    def get_storage_info(self) -> Dict[str, Dict[str, str]]:
        """Get storage information for important paths"""
        paths_to_check = {
            "models": self.runpod_config.models_path,
            "cache": self.runpod_config.cache_path,
            "temp": self.runpod_config.temp_path,
            "outputs": self.runpod_config.outputs_path
        }
        
        storage_info = {}
        
        for name, path in paths_to_check.items():
            try:
                if os.path.exists(path):
                    usage = shutil.disk_usage(path)
                    total = usage.total / 1024**3
                    free = usage.free / 1024**3
                    used = (usage.total - usage.free) / 1024**3
                    
                    storage_info[name] = {
                        "path": path,
                        "total": f"{total:.1f} GB",
                        "used": f"{used:.1f} GB",
                        "free": f"{free:.1f} GB",
                        "percentage": f"{(used / total) * 100:.1f}%"
                    }
                else:
                    storage_info[name] = {
                        "path": path,
                        "status": "path_not_exists"
                    }
                    
            except Exception as e:
                storage_info[name] = {
                    "path": path,
                    "error": str(e)
                }
        
        return storage_info
    
    def check_gpu_requirements(self, model_name: str) -> Dict[str, Any]:
        """
        Check if GPU meets requirements for a specific model
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            Dictionary with requirement check results
        """
        if not self.gpu_available:
            return {
                "meets_requirements": False,
                "reason": "No GPU available",
                "recommendations": ["Use CPU-based models or enable GPU support"]
            }
        
        try:
            model_config = self.config_manager.get_model_config(model_name)
            required_vram_str = model_config.vram_requirement
            
            # Parse VRAM requirement (e.g., "10GB" -> 10.0)
            required_vram = float(required_vram_str.replace('GB', '').strip())
            available_vram = self.gpu_total_memory / 1024**3
            
            meets_requirements = available_vram >= required_vram
            
            result = {
                "model_name": model_name,
                "meets_requirements": meets_requirements,
                "required_vram": f"{required_vram} GB",
                "available_vram": f"{available_vram:.1f} GB",
                "gpu_name": self.gpu_name
            }
            
            if not meets_requirements:
                result.update({
                    "reason": f"Insufficient VRAM. Required: {required_vram} GB, Available: {available_vram:.1f} GB",
                    "recommendations": [
                        "Use a model with lower VRAM requirements",
                        "Enable attention slicing to reduce memory usage",
                        "Use a GPU with more VRAM"
                    ]
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to check GPU requirements for {model_name}: {e}")
            return {
                "meets_requirements": False,
                "reason": f"Error checking requirements: {e}",
                "recommendations": ["Check model configuration and GPU status"]
            }
    
    def monitor_resources(self) -> Dict[str, Any]:
        """
        Get comprehensive resource monitoring information
        
        Returns:
            Dictionary with current resource status
        """
        return {
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "memory": self.check_available_memory(),
            "storage": self.get_storage_info(),
            "device_info": {
                "device": self.device,
                "gpu_available": self.gpu_available,
                "gpu_name": self.gpu_name,
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "pytorch_version": torch.__version__
            },
            "system_info": {
                "platform": __import__('platform').platform(),
                "cpu_count": os.cpu_count(),
                "cpu_usage": f"{psutil.cpu_percent():.1f}%"
            }
        }
    
    def cleanup_temporary_files(self) -> Dict[str, Any]:
        """
        Clean up temporary files and caches
        
        Returns:
            Cleanup results
        """
        cleanup_results = {
            "cleaned_paths": [],
            "freed_space": 0,
            "errors": []
        }
        
        temp_paths = [
            self.runpod_config.temp_path,
            "/tmp/pytorch_cache",
            "/tmp/diffusers_cache"
        ]
        
        for temp_path in temp_paths:
            try:
                if os.path.exists(temp_path):
                    # Calculate size before cleanup
                    size_before = self._get_directory_size(temp_path)
                    
                    # Remove temporary files
                    if os.path.isdir(temp_path):
                        for item in os.listdir(temp_path):
                            item_path = os.path.join(temp_path, item)
                            if os.path.isfile(item_path) and item.startswith('tmp'):
                                os.remove(item_path)
                    
                    # Calculate freed space
                    size_after = self._get_directory_size(temp_path) if os.path.exists(temp_path) else 0
                    freed = (size_before - size_after) / 1024**2  # MB
                    
                    cleanup_results["cleaned_paths"].append(temp_path)
                    cleanup_results["freed_space"] += freed
                    
            except Exception as e:
                cleanup_results["errors"].append(f"Failed to clean {temp_path}: {e}")
                logger.warning(f"Failed to clean temporary path {temp_path}: {e}")
        
        # Clear GPU cache
        if self.gpu_available:
            self.cleanup_memory()
        
        logger.info(f"Cleanup completed. Freed {cleanup_results['freed_space']:.1f} MB")
        return cleanup_results
    
    def _get_directory_size(self, path: str) -> int:
        """Get total size of directory in bytes"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except Exception as e:
            logger.debug(f"Error calculating directory size for {path}: {e}")
        
        return total_size
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health status
        
        Returns:
            System health report
        """
        memory_info = self.check_available_memory()
        storage_info = self.get_storage_info()
        
        health_status = "healthy"
        warnings = []
        errors = []
        
        # Check system memory
        sys_mem_usage = float(memory_info["system_memory"]["percentage"].replace('%', ''))
        if sys_mem_usage > 90:
            health_status = "critical"
            errors.append("System memory usage is critical (>90%)")
        elif sys_mem_usage > 80:
            health_status = "warning"
            warnings.append("System memory usage is high (>80%)")
        
        # Check GPU memory if available
        if self.gpu_available and memory_info["gpu_memory"]:
            gpu_usage = float(memory_info["gpu_memory"]["utilization"].replace('%', ''))
            if gpu_usage > 95:
                health_status = "critical"
                errors.append("GPU memory usage is critical (>95%)")
            elif gpu_usage > 85:
                if health_status != "critical":
                    health_status = "warning"
                warnings.append("GPU memory usage is high (>85%)")
        
        # Check storage
        for path_name, path_info in storage_info.items():
            if "percentage" in path_info:
                storage_usage = float(path_info["percentage"].replace('%', ''))
                if storage_usage > 95:
                    health_status = "critical"
                    errors.append(f"Storage usage for {path_name} is critical (>95%)")
                elif storage_usage > 85:
                    if health_status != "critical":
                        health_status = "warning"
                    warnings.append(f"Storage usage for {path_name} is high (>85%)")
        
        return {
            "status": health_status,
            "warnings": warnings,
            "errors": errors,
            "memory": memory_info,
            "storage": storage_info,
            "recommendations": self._get_health_recommendations(health_status, warnings, errors)
        }
    
    def _get_health_recommendations(self, status: str, warnings: list, errors: list) -> list:
        """Get recommendations based on health status"""
        recommendations = []
        
        if status == "critical":
            recommendations.extend([
                "Free up memory and storage space immediately",
                "Stop non-essential processes",
                "Consider using smaller models or reducing batch sizes"
            ])
        elif status == "warning":
            recommendations.extend([
                "Monitor resource usage closely",
                "Consider cleaning up temporary files",
                "Use memory optimization techniques"
            ])
        
        if any("memory" in msg.lower() for msg in warnings + errors):
            recommendations.extend([
                "Enable attention slicing for models",
                "Reduce batch size",
                "Use gradient checkpointing if training"
            ])
        
        if any("storage" in msg.lower() for msg in warnings + errors):
            recommendations.extend([
                "Clean up old model checkpoints",
                "Remove temporary files",
                "Consider using external storage"
            ])
        
        return recommendations