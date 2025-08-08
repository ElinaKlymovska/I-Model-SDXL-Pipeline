#!/usr/bin/env python3
"""
RunPod CLI Interface
Command-line interface for managing RunPod instances
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

from .manager import manager
from .utils import get_system_info, ensure_pod_running, upload_project_files, create_backup, restore_backup, monitor_process, get_logs
from .config import config

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def cmd_status(args):
    """Show pod status"""
    print("🔍 Checking pod status...")
    
    pod_info = manager.get_pod_info()
    if not pod_info:
        print("❌ Could not get pod information")
        return 1
    
    pod_id = pod_info.get("id", "UNKNOWN")
    runtime = pod_info.get("runtime", {})
    
    print(f"📊 Pod ID: {pod_id}")
    
    # Determine status based on runtime
    if runtime:
        print("✅ Pod is running")
        
        # Get connection details
        if manager.wait_for_pod_ready():
            pod_status = manager.get_pod_status()
            if pod_status:
                print(f"🔗 SSH: {pod_status.ssh_host}:{pod_status.ssh_port}")
                if pod_status.jupyter_url:
                    print(f"📓 Jupyter: {pod_status.jupyter_url}")
                
                # Test SSH connection
                if manager.test_ssh_connection():
                    print("✅ SSH connection: OK")
                else:
                    print("❌ SSH connection: Failed")
        
        # Get system info
        print("\n💻 System Information:")
        system_info = get_system_info()
        for key, value in system_info.items():
            print(f"   {key}: {value}")
    else:
        print("⏸️  Pod is stopped or starting")
    
    return 0

def cmd_start(args):
    """Start the pod"""
    print("🚀 Starting pod...")
    
    if manager.start_pod():
        print("✅ Pod started successfully")
        
        if args.wait:
            print("⏳ Waiting for pod to be ready...")
            if manager.wait_for_pod_ready():
                print("✅ Pod is ready")
                return 0
            else:
                print("❌ Pod failed to become ready")
                return 1
        else:
            print("💡 Use 'runpod status' to check when pod is ready")
            return 0
    else:
        print("❌ Failed to start pod")
        return 1

def cmd_stop(args):
    """Stop the pod"""
    print("🛑 Stopping pod...")
    
    if manager.stop_pod():
        print("✅ Pod stopped successfully")
        return 0
    else:
        print("❌ Failed to stop pod")
        return 1

def cmd_connect(args):
    """Connect to the pod"""
    print("🔗 Connecting to pod...")
    
    if not ensure_pod_running(auto_start=args.auto_start):
        print("❌ Could not ensure pod is running")
        return 1
    
    if manager.test_ssh_connection():
        print("✅ Successfully connected to pod")
        
        if args.cmd:
            print(f"🔧 Executing command: {args.cmd}")
            success, stdout, stderr = manager.execute_ssh_command(args.cmd)
            if success:
                print("✅ Command executed successfully")
                if stdout:
                    print("📤 Output:")
                    print(stdout)
            else:
                print("❌ Command failed")
                if stderr:
                    print(f"Error: {stderr}")
                return 1
        else:
            print("💡 Use 'runpod connect --cmd \"your_command\"' to execute commands")
        
        return 0
    else:
        print("❌ Failed to connect to pod")
        return 1

def cmd_upload(args):
    """Upload files to the pod"""
    print("📤 Uploading files...")
    
    if not ensure_pod_running(auto_start=args.auto_start):
        print("❌ Could not ensure pod is running")
        return 1
    
    local_path = Path(args.local_path)
    if not local_path.exists():
        print(f"❌ Local path does not exist: {local_path}")
        return 1
    
    remote_path = args.remote_path or f"/workspace/{local_path.name}"
    
    if local_path.is_file():
        success = manager.upload_file(str(local_path), remote_path)
    else:
        success = upload_project_files(str(local_path))
    
    if success:
        print(f"✅ Uploaded {local_path} to {remote_path}")
        return 0
    else:
        print("❌ Upload failed")
        return 1

def cmd_download(args):
    """Download files from the pod"""
    print("📥 Downloading files...")
    
    if not ensure_pod_running(auto_start=args.auto_start):
        print("❌ Could not ensure pod is running")
        return 1
    
    local_path = Path(args.local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    success = manager.download_file(args.remote_path, str(local_path))
    
    if success:
        print(f"✅ Downloaded {args.remote_path} to {local_path}")
        return 0
    else:
        print("❌ Download failed")
        return 1

def cmd_setup(args):
    """Setup workspace on the pod"""
    print("🔧 Setting up workspace...")
    
    if not ensure_pod_running(auto_start=args.auto_start):
        print("❌ Could not ensure pod is running")
        return 1
    
    if manager.setup_workspace():
        print("✅ Workspace setup completed")
        
        if args.upload:
            print("📤 Uploading project files to pod...")
            if upload_project_files("./"):
                print("✅ Project uploaded")
            else:
                print("❌ Project upload failed")
                return 1

        if args.install_deps:
            print("📦 Installing dependencies...")
            if manager.install_dependencies():
                print("✅ Dependencies installed successfully")
            else:
                print("❌ Failed to install dependencies")
                return 1
        
        return 0
    else:
        print("❌ Workspace setup failed")
        return 1

def cmd_execute(args):
    """Execute a script on the pod"""
    print(f"🔧 Executing script: {args.script}")
    
    if not ensure_pod_running(auto_start=args.auto_start):
        print("❌ Could not ensure pod is running")
        return 1
    
    script_args = args.args if args.args else []
    
    # Execute script using manager
    success, stdout, stderr = manager.execute_ssh_command(f"python3 {args.script} {' '.join(script_args)}")
    if success:
        print("✅ Script executed successfully")
        return 0
    else:
        print("❌ Script execution failed")
        return 1

def cmd_monitor(args):
    """Monitor a process on the pod"""
    print(f"📊 Monitoring process: {args.process}")
    
    if not ensure_pod_running(auto_start=args.auto_start):
        print("❌ Could not ensure pod is running")
        return 1
    
    if monitor_process(args.process, args.timeout):
        print("✅ Process completed successfully")
        return 0
    else:
        print("❌ Process monitoring failed or timed out")
        return 1

def cmd_backup(args):
    """Create or restore backup"""
    if args.action == "create":
        print(f"💾 Creating backup of {args.path}")
        
        if not ensure_pod_running(auto_start=args.auto_start):
            print("❌ Could not ensure pod is running")
            return 1
        
        if create_backup(args.path, args.name):
            print("✅ Backup created successfully")
            return 0
        else:
            print("❌ Backup creation failed")
            return 1
    
    elif args.action == "restore":
        print(f"📦 Restoring backup from {args.backup_path}")
        
        if not ensure_pod_running(auto_start=args.auto_start):
            print("❌ Could not ensure pod is running")
            return 1
        
        if restore_backup(args.backup_path, args.path):
            print("✅ Backup restored successfully")
            return 0
        else:
            print("❌ Backup restoration failed")
            return 1

def cmd_config(args):
    """Show or update configuration"""
    if args.show:
        print("⚙️  Current Configuration:")
        print(f"   Pod ID: {config.pod_id}")
        print(f"   SSH User: {config.ssh_user}")
        print(f"   SSH Key: {config.ssh_key_path}")
        print(f"   GPU Type: {config.gpu_type}")
        print(f"   Container: {config.container_image}")
        print(f"   Workspace: {config.workspace_path}")
        
        if config.api_key:
            print(f"   API Key: {'*' * 10}{config.api_key[-4:]}")
        else:
            print("   API Key: Not configured")
    
    return 0

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="RunPod Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  runpod status                    # Check pod status
  runpod start --wait              # Start pod and wait for ready
  runpod connect --command "ls"    # Execute command on pod
  runpod upload ./data /workspace  # Upload local directory
  runpod setup --install-deps      # Setup workspace and install dependencies
        """
    )
    
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--auto-start', action='store_true', help='Auto-start pod if stopped')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)
    
    # Status command
    subparsers.add_parser('status', help='Show pod status')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start the pod')
    start_parser.add_argument('--wait', action='store_true', help='Wait for pod to be ready')
    
    # Stop command
    subparsers.add_parser('stop', help='Stop the pod')
    
    # Connect command
    connect_parser = subparsers.add_parser('connect', help='Connect to the pod')
    connect_parser.add_argument('--cmd', help='Command to execute')
    
    # Upload command
    upload_parser = subparsers.add_parser('upload', help='Upload files to the pod')
    upload_parser.add_argument('local_path', help='Local file or directory path')
    upload_parser.add_argument('remote_path', nargs='?', help='Remote path (optional)')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download files from the pod')
    download_parser.add_argument('remote_path', help='Remote file path')
    download_parser.add_argument('local_path', help='Local file path')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup workspace on the pod')
    setup_parser.add_argument('--install-deps', action='store_true', help='Install dependencies')
    setup_parser.add_argument('--upload', action='store_true', help='Upload current project to /workspace')
    
    # Execute command
    execute_parser = subparsers.add_parser('execute', help='Execute a script on the pod')
    execute_parser.add_argument('script', help='Script path on the pod')
    execute_parser.add_argument('args', nargs='*', help='Script arguments')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor a process on the pod')
    monitor_parser.add_argument('process', help='Process name to monitor')
    monitor_parser.add_argument('--timeout', type=int, default=3600, help='Timeout in seconds')
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Create or restore backup')
    backup_parser.add_argument('action', choices=['create', 'restore'], help='Backup action')
    backup_parser.add_argument('path', help='Path to backup/restore')
    backup_parser.add_argument('--name', help='Backup name (for create)')
    backup_parser.add_argument('--backup-path', help='Backup path (for restore)')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Show configuration')
    config_parser.add_argument('--show', action='store_true', help='Show current configuration')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Execute command
    if args.command == 'status':
        return cmd_status(args)
    elif args.command == 'start':
        return cmd_start(args)
    elif args.command == 'stop':
        return cmd_stop(args)
    elif args.command == 'connect':
        return cmd_connect(args)
    elif args.command == 'upload':
        return cmd_upload(args)
    elif args.command == 'download':
        return cmd_download(args)
    elif args.command == 'setup':
        return cmd_setup(args)
    elif args.command == 'execute':
        return cmd_execute(args)
    elif args.command == 'monitor':
        return cmd_monitor(args)
    elif args.command == 'backup':
        return cmd_backup(args)
    elif args.command == 'config':
        return cmd_config(args)
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())
