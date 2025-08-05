"""
RunPod Connection Manager
Handles connection, file transfer, and command execution on RunPod instances
"""

import json
import time
import requests
import paramiko
import subprocess
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

from .config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RunPodManager:
    """Manages connections and operations with RunPod instances"""
    
    def __init__(self, pod_id: str = None):
        """Initialize RunPod manager
        
        Args:
            pod_id: RunPod instance ID. If None, uses config.pod_id
        """
        self.pod_id = pod_id or config.pod_id
        self.ssh_client = None
        self.pod_info = None
        
        if not config.api_key:
            logger.warning("RUNPOD_API_KEY not found in environment variables")
    
    def get_pod_info(self) -> Optional[Dict]:
        """Get pod information from RunPod GraphQL API
        
        Returns:
            Dict containing pod information or None if error
        """
        if not config.api_key:
            logger.error("API key required for pod information")
            return None
            
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        
        # GraphQL query to get specific pod info
        query = f'''
        {{
          myself {{
            pods {{
              id
              name
              desiredStatus
              machine {{
                runpodIp
              }}
              runtime {{
                ports {{
                  privatePort
                  publicPort
                }}
              }}
            }}
          }}
        }}
        '''
        
        payload = {"query": query}
        
        try:
            response = requests.post(
                "https://api.runpod.io/graphql",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data and 'myself' in data['data'] and 'pods' in data['data']['myself']:
                    pods = data['data']['myself']['pods']
                    
                    # Find our specific pod
                    for pod in pods:
                        if pod.get('id') == self.pod_id:
                            self.pod_info = pod
                            logger.info(f"Successfully retrieved pod info for {self.pod_id}")
                            return self.pod_info
                    
                    logger.error(f"Pod {self.pod_id} not found in user's pods")
                    return None
                else:
                    logger.error("Invalid response structure from GraphQL API")
                    return None
            else:
                logger.error(f"Failed to get pod info: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting pod info: {e}")
            return None
    
    def wait_for_pod_ready(self, max_wait_time: int = 300) -> bool:
        """Wait for pod to be in running state
        
        Args:
            max_wait_time: Maximum time to wait in seconds
            
        Returns:
            True if pod is ready, False if timeout or error
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            pod_info = self.get_pod_info()
            
            if pod_info and pod_info.get("desiredStatus") == "RUNNING":
                logger.info("Pod is ready!")
                return True
                
            logger.info("Waiting for pod to be ready...")
            time.sleep(10)
        
        logger.error("Timeout waiting for pod to be ready")
        return False
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information for the pod
        
        Returns:
            Dict with connection details
        """
        if not self.pod_info:
            self.get_pod_info()
        
        if not self.pod_info:
            return {"error": "Pod info not available"}
        
        runtime = self.pod_info.get("runtime", {})
        machine = self.pod_info.get("machine", {})
        ports = runtime.get("ports", [])
        
        # Extract IP
        pod_ip_with_mask = machine.get("runpodIp")
        pod_ip = pod_ip_with_mask.split('/')[0] if pod_ip_with_mask and '/' in pod_ip_with_mask else pod_ip_with_mask
        
        # Find service ports
        service_ports = {}
        for port in ports:
            private_port = port.get("privatePort")
            public_port = port.get("publicPort")
            
            if private_port == 22:
                service_ports['ssh'] = public_port
            elif private_port == 8888:
                service_ports['jupyter'] = public_port
                service_ports['jupyter_url'] = f"https://{public_port}-{self.pod_id}.proxy.runpod.net"
            elif private_port == 7860:
                service_ports['gradio'] = public_port
                service_ports['gradio_url'] = f"https://{public_port}-{self.pod_id}.proxy.runpod.net"
        
        return {
            "pod_id": self.pod_id,
            "pod_name": self.pod_info.get("name"),
            "status": self.pod_info.get("desiredStatus"),
            "ip": pod_ip,
            "ports": service_ports
        }
    
    def connect_ssh(self, username: str = "root", ssh_key_path: str = None) -> bool:
        """Connect to pod via SSH
        
        Args:
            username: SSH username (default: root)
            ssh_key_path: Path to SSH private key file
            
        Returns:
            True if connection successful, False otherwise
        """
        if not self.pod_info:
            logger.error("Pod info not available. Call get_pod_info() first")
            return False
        
        # Extract connection details from pod info
        runtime = self.pod_info.get("runtime", {})
        ports = runtime.get("ports", [])
        
        # Find SSH port (usually 22 is mapped to a different external port)
        ssh_port = None
        for port in ports:
            if port.get("privatePort") == 22:
                ssh_port = port.get("publicPort")
                break
        
        if not ssh_port:
            logger.error("SSH port not found in pod configuration")
            return False
        
        # Get IP from machine info
        machine = self.pod_info.get("machine", {})
        pod_ip_with_mask = machine.get("runpodIp")
        
        if not pod_ip_with_mask:
            logger.error("Pod IP not found in machine info")
            return False
        
        # Extract IP address (remove /mask if present)
        pod_ip = pod_ip_with_mask.split('/')[0] if '/' in pod_ip_with_mask else pod_ip_with_mask
        
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            if ssh_key_path:
                # Use SSH key
                key = paramiko.RSAKey.from_private_key_file(ssh_key_path)
                self.ssh_client.connect(
                    hostname=pod_ip,
                    port=ssh_port,
                    username=username,
                    pkey=key,
                    timeout=30
                )
            else:
                # You might need to set up password or key-based auth
                logger.warning("No SSH key provided. You may need to configure authentication")
                return False
            
            logger.info(f"Successfully connected to pod via SSH at {pod_ip}:{ssh_port}")
            return True
            
        except Exception as e:
            logger.error(f"SSH connection failed: {e}")
            return False
    
    def execute_command(self, command: str, working_dir: str = None) -> Dict[str, Any]:
        """Execute command on the pod
        
        Args:
            command: Command to execute
            working_dir: Working directory for command execution
            
        Returns:
            Dict with stdout, stderr, and exit_code
        """
        if not self.ssh_client:
            return {"error": "SSH not connected"}
        
        try:
            if working_dir:
                command = f"cd {working_dir} && {command}"
            
            stdin, stdout, stderr = self.ssh_client.exec_command(command)
            
            stdout_text = stdout.read().decode('utf-8')
            stderr_text = stderr.read().decode('utf-8')
            exit_code = stdout.channel.recv_exit_status()
            
            return {
                "stdout": stdout_text,
                "stderr": stderr_text,
                "exit_code": exit_code
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def upload_file(self, local_path: str, remote_path: str) -> bool:
        """Upload file to pod
        
        Args:
            local_path: Local file path
            remote_path: Remote file path on pod
            
        Returns:
            True if successful, False otherwise
        """
        if not self.ssh_client:
            logger.error("SSH not connected")
            return False
        
        try:
            sftp = self.ssh_client.open_sftp()
            sftp.put(local_path, remote_path)
            sftp.close()
            
            logger.info(f"Successfully uploaded {local_path} to {remote_path}")
            return True
            
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False
    
    def upload_directory(self, local_dir: str, remote_dir: str) -> bool:
        """Upload entire directory to pod
        
        Args:
            local_dir: Local directory path
            remote_dir: Remote directory path on pod
            
        Returns:
            True if successful, False otherwise
        """
        if not self.ssh_client:
            logger.error("SSH not connected")
            return False
        
        try:
            # Create remote directory
            self.execute_command(f"mkdir -p {remote_dir}")
            
            sftp = self.ssh_client.open_sftp()
            
            # Upload all files recursively
            for local_file in Path(local_dir).rglob("*"):
                if local_file.is_file():
                    relative_path = local_file.relative_to(local_dir)
                    remote_file = Path(remote_dir) / relative_path
                    
                    # Create remote directory if needed
                    remote_parent = remote_file.parent
                    self.execute_command(f"mkdir -p {remote_parent}")
                    
                    sftp.put(str(local_file), str(remote_file))
            
            sftp.close()
            
            logger.info(f"Successfully uploaded directory {local_dir} to {remote_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Directory upload failed: {e}")
            return False
    
    def download_file(self, remote_path: str, local_path: str) -> bool:
        """Download file from pod
        
        Args:
            remote_path: Remote file path on pod
            local_path: Local file path
            
        Returns:
            True if successful, False otherwise
        """
        if not self.ssh_client:
            logger.error("SSH not connected")
            return False
        
        try:
            # Create local directory if needed
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            sftp = self.ssh_client.open_sftp()
            sftp.get(remote_path, local_path)
            sftp.close()
            
            logger.info(f"Successfully downloaded {remote_path} to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
    
    def install_requirements(self, requirements_file: str = "requirements.txt") -> bool:
        """Install Python requirements on the pod
        
        Args:
            requirements_file: Path to requirements file on pod
            
        Returns:
            True if successful, False otherwise
        """
        result = self.execute_command(f"pip install -r {requirements_file}")
        
        if result.get("exit_code") == 0:
            logger.info("Requirements installed successfully")
            return True
        else:
            logger.error(f"Requirements installation failed: {result.get('stderr')}")
            return False
    
    def close_connection(self):
        """Close SSH connection"""
        if self.ssh_client:
            self.ssh_client.close()
            self.ssh_client = None
            logger.info("SSH connection closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close_connection()