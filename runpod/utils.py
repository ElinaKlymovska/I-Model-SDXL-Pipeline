"""
RunPod Utilities
Utility functions for RunPod operations
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from .manager import manager
from .config import config

logger = logging.getLogger(__name__)

def ensure_pod_running(auto_start: bool = False) -> bool:
    """Ensure pod is running, optionally start if not"""
    # Check if pod is running by testing SSH connection
    if manager.test_ssh_connection():
        return True
    
    if auto_start:
        logger.info("Pod not running, attempting to start...")
        return manager.start_pod()
    
    return False

def upload_project_files(local_path: str = "./") -> bool:
    """Upload project files to RunPod"""
    if not ensure_pod_running():
        logger.error("No running pod available")
        return False
    
    # Create a temporary archive
    archive_name = "project_files.tar.gz"
    try:
        # Create archive
        subprocess.run([
            "tar", "-czf", archive_name, 
            "--exclude=node_modules", 
            "--exclude=.git", 
            "--exclude=__pycache__",
            "--exclude=*.pyc",
            local_path
        ], check=True)
        
        # Upload archive
        success = manager.upload_file(archive_name, f"/workspace/{archive_name}")
        
        if success:
            # Extract on RunPod
            cmd = f"cd /workspace && tar -xzf {archive_name} && rm {archive_name}"
            success, stdout, stderr = manager.execute_ssh_command(cmd)
            
            if success:
                logger.info("Project files uploaded and extracted successfully")
                return True
        
        # Cleanup
        if os.path.exists(archive_name):
            os.remove(archive_name)
            
    except Exception as e:
        logger.error(f"Error uploading project files: {e}")
    
    return False

def get_system_info() -> Dict[str, Any]:
    """Get system information from RunPod"""
    if not ensure_pod_running():
        return {"error": "No running pod available"}
    
    info = {}
    
    # GPU info
    success, stdout, stderr = manager.execute_ssh_command("nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits")
    if success and stdout.strip():
        gpu_info = stdout.strip().split(', ')
        info['gpu'] = f"{gpu_info[0]} ({gpu_info[1]}MB total, {gpu_info[2]}MB used)"
    else:
        info['gpu'] = "Not available"
    
    # Memory info
    success, stdout, stderr = manager.execute_ssh_command("free -h | grep Mem")
    if success and stdout.strip():
        parts = stdout.split()
        info['memory'] = f"{parts[1]} total, {parts[2]} used, {parts[3]} free"
    else:
        info['memory'] = "Not available"
    
    # Disk info
    success, stdout, stderr = manager.execute_ssh_command("df -h /workspace")
    if success and stdout.strip():
        lines = stdout.strip().split('\n')
        if len(lines) > 1:
            parts = lines[1].split()
            info['disk'] = f"{parts[1]} total, {parts[2]} used, {parts[3]} available"
    else:
        info['disk'] = "Not available"
    
    # Python version
    success, stdout, stderr = manager.execute_ssh_command("python3 --version")
    if success:
        info['python'] = stdout.strip()
    else:
        info['python'] = "Not available"
    
    return info

def create_backup(path: str, backup_name: str = None) -> bool:
    """Create backup of specified path on RunPod"""
    if not ensure_pod_running():
        logger.error("No running pod available")
        return False
    
    if not backup_name:
        from datetime import datetime
        backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tar.gz"
    
    cmd = f"cd /workspace && tar -czf {backup_name} {path}"
    success, stdout, stderr = manager.execute_ssh_command(cmd)
    
    if success:
        logger.info(f"Backup created: {backup_name}")
        return True
    else:
        logger.error(f"Backup failed: {stderr}")
        return False

def restore_backup(backup_path: str, target_path: str = "/workspace") -> bool:
    """Restore backup from specified path"""
    if not ensure_pod_running():
        logger.error("No running pod available")
        return False
    
    cmd = f"cd {target_path} && tar -xzf {backup_path}"
    success, stdout, stderr = manager.execute_ssh_command(cmd)
    
    if success:
        logger.info(f"Backup restored from: {backup_path}")
        return True
    else:
        logger.error(f"Restore failed: {stderr}")
        return False

def monitor_process(process_name: str, timeout: int = 3600) -> bool:
    """Monitor a process on RunPod"""
    if not ensure_pod_running():
        logger.error("No running pod available")
        return False
    
    import time
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        # Check if process is running
        cmd = f"pgrep -f '{process_name}'"
        success, stdout, stderr = manager.execute_ssh_command(cmd)
        
        if success and stdout.strip():
            logger.info(f"Process {process_name} is running (PID: {stdout.strip()})")
            time.sleep(30)  # Check every 30 seconds
        else:
            logger.info(f"Process {process_name} is not running")
            return False
    
    logger.warning(f"Process monitoring timed out after {timeout} seconds")
    return False

def get_logs(log_path: str = "/workspace/logs", lines: int = 50) -> str:
    """Get recent logs from RunPod"""
    if not ensure_pod_running():
        return "No running pod available"
    
    cmd = f"find {log_path} -name '*.log' -exec tail -n {lines} {{}} \\;"
    success, stdout, stderr = manager.execute_ssh_command(cmd)
    
    if success:
        return stdout
    else:
        return f"Error getting logs: {stderr}"

# Export functions directly
__all__ = [
    'ensure_pod_running',
    'upload_project_files', 
    'get_system_info',
    'create_backup',
    'restore_backup',
    'monitor_process',
    'get_logs',
    'download_and_extract_outputs'
]

def download_and_extract_outputs(remote_dir: str, local_base_dir: str = "data/output_images",
                                 archive_name_prefix: str = "facial_outputs") -> Optional[str]:
    """Archive remote output directory on the pod, download and extract locally into timestamped folder.

    Args:
        remote_dir: Remote directory path on the pod (absolute or relative to /workspace)
        local_base_dir: Local base directory for extraction
        archive_name_prefix: Prefix for temporary archive names

    Returns:
        Optional[str]: Path to the local extracted directory if successful, None otherwise
    """
    if not ensure_pod_running():
        logger.error("No running pod available")
        return None

    # Normalize remote paths
    remote_dir = remote_dir.strip()
    remote_dir_quoted = remote_dir.replace("'", "'\\''")

    # Verify remote directory exists
    check_cmd = f"test -d '{remote_dir_quoted}' && echo OK || echo MISSING"
    ok, stdout, _ = manager.execute_ssh_command(check_cmd)
    if not ok or 'OK' not in stdout:
        logger.error(f"Remote directory not found: {remote_dir}")
        return None

    # Timestamp for names
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    remote_archive = f"/workspace/{archive_name_prefix}_{ts}.tar.gz"

    # Create archive on pod (archive by parent dir and name to ensure relative paths)
    # Example: tar -C /workspace/data/output_images -czf /workspace/facial_outputs_...tar.gz batch
    parent_dir_cmd = f"python3 -c 'import os,sys; p=\"{remote_dir_quoted}\"; import pathlib; q=str(pathlib.Path(p).parent); print(q)'"
    ok, parent_stdout, _ = manager.execute_ssh_command(parent_dir_cmd)
    if not ok or not parent_stdout.strip():
        logger.error("Failed to resolve remote parent directory")
        return None
    remote_parent = parent_stdout.strip().splitlines()[0]
    remote_name_cmd = f"python3 -c 'import pathlib; print(pathlib.Path(\"{remote_dir_quoted}\").name)'"
    ok, name_stdout, _ = manager.execute_ssh_command(remote_name_cmd)
    if not ok or not name_stdout.strip():
        logger.error("Failed to resolve remote directory name")
        return None
    remote_name = name_stdout.strip().splitlines()[0]

    tar_cmd = f"tar -C '{remote_parent}' -czf '{remote_archive}' '{remote_name}'"
    ok, _, stderr = manager.execute_ssh_command(tar_cmd)
    if not ok:
        logger.error(f"Failed to create remote archive: {stderr}")
        return None

    # Download archive locally
    local_archive = Path(f"./{Path(remote_archive).name}")
    if not manager.download_file(remote_archive, str(local_archive)):
        logger.error("Failed to download remote archive")
        return None

    # Prepare local extraction directory
    local_base = Path(local_base_dir)
    target_dir = local_base / f"batch_{ts}"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Extract archive
    try:
        import tarfile
        with tarfile.open(local_archive, 'r:gz') as tf:
            tf.extractall(path=target_dir)
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return None
    finally:
        # Clean up local archive
        try:
            if local_archive.exists():
                local_archive.unlink()
        except Exception:
            pass

    # Optionally remove remote archive
    manager.execute_ssh_command(f"rm -f {remote_archive}")

    logger.info(f"Outputs downloaded and extracted to: {target_dir}")
    return str(target_dir)
