#!/bin/bash
# RunPod Quick Start для Enhanced Face Correction Pipeline

echo "🚀 RunPod Enhanced Face Correction - Quick Start"
echo "================================================"

# Detect if running on RunPod
if [ ! -z "$RUNPOD_POD_ID" ]; then
    echo "✅ RunPod environment detected: Pod ID $RUNPOD_POD_ID"
    WORKSPACE="/workspace"
else
    echo "💻 Local environment detected"
    WORKSPACE="$(pwd)"
fi

# Project setup
PROJECT_DIR="$WORKSPACE/I-Model"
REPO_URL="https://github.com/ElinaKlymovska/I-Model-SDXL-Pipeline.git"

echo ""
echo "📋 Setup Options:"
echo "1. 🚀 Full Setup (clone + install + models + WebUI)"
echo "2. 📦 Quick Setup (basic models only)"
echo "3. 🔄 Update Project (git pull)"
echo "4. 💻 Launch WebUI Only"
echo "5. 🎭 Deploy Face Correction (automated)"

read -p "Choose option (1-5): " option

case $option in
    1)
        echo "🚀 Full Setup Starting..."
        
        # Clone or update project
        if [ -d "$PROJECT_DIR" ]; then
            echo "📁 Updating existing project..."
            cd "$PROJECT_DIR"
            git pull origin develop
        else
            echo "📥 Cloning project..."
            git clone -b develop "$REPO_URL" "$PROJECT_DIR"
            cd "$PROJECT_DIR"
        fi
        
        # Install dependencies
        echo "📦 Installing Python dependencies..."
        pip install --upgrade pip
        pip install -r requirements_runpod.txt
        
        # Setup WebUI with professional models
        echo "🎨 Setting up Stable Diffusion WebUI..."
        python utils/runpod_launcher.py --setup-only --preset professional --face-models
        
        # Download models
        echo "📥 Downloading professional models..."
        python utils/runpod_launcher.py --download-only --preset professional --face-models
        
        # Start WebUI
        echo "🌐 Starting WebUI..."
        python utils/runpod_launcher.py --launch &
        
        echo "✅ Full setup completed!"
        echo "🔗 WebUI will be available at:"
        if [ ! -z "$RUNPOD_POD_ID" ]; then
            echo "   https://$RUNPOD_POD_ID-3000.proxy.runpod.net"
        else
            echo "   http://localhost:3000"
        fi
        ;;
        
    2)
        echo "📦 Quick Setup Starting..."
        
        # Clone or update
        if [ -d "$PROJECT_DIR" ]; then
            cd "$PROJECT_DIR"
            git pull origin develop
        else
            git clone -b develop "$REPO_URL" "$PROJECT_DIR"
            cd "$PROJECT_DIR"
        fi
        
        # Basic install
        pip install --upgrade pip
        pip install -r requirements_runpod.txt
        
        # Basic setup
        python utils/runpod_launcher.py --setup-only --preset basic
        python utils/runpod_launcher.py --download-only --preset basic
        python utils/runpod_launcher.py --launch &
        
        echo "✅ Quick setup completed!"
        ;;
        
    3)
        echo "🔄 Updating project..."
        if [ -d "$PROJECT_DIR" ]; then
            cd "$PROJECT_DIR"
            git pull origin develop
            echo "✅ Project updated!"
        else
            echo "❌ Project directory not found"
            exit 1
        fi
        ;;
        
    4)
        echo "💻 Launching WebUI..."
        if [ -d "$PROJECT_DIR" ]; then
            cd "$PROJECT_DIR"
            python utils/runpod_launcher.py --launch
        else
            echo "❌ Project not found. Run setup first."
            exit 1
        fi
        ;;
        
    5)
        echo "🎭 Automated Face Correction Deployment..."
        
        # Clone or update
        if [ -d "$PROJECT_DIR" ]; then
            cd "$PROJECT_DIR"
            git pull origin develop
        else
            git clone -b develop "$REPO_URL" "$PROJECT_DIR"
            cd "$PROJECT_DIR"
        fi
        
        # Run automated deployment
        pip install --upgrade pip
        pip install -r requirements_runpod.txt
        python scripts/deploy_runpod_face_correction.py --preset professional --demo
        
        echo "✅ Automated deployment completed!"
        ;;
        
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac

echo ""
echo "🎉 Setup complete! Next steps:"
echo ""
echo "📁 Project location: $PROJECT_DIR"
echo ""
echo "🎭 Face Correction Commands:"
echo "  cd $PROJECT_DIR"
echo "  python scripts/demo_face_correction.py  # Demo"
echo "  python scripts/demo_pipeline.py --help  # Full pipeline"
echo ""
echo "🌐 WebUI Access:"
if [ ! -z "$RUNPOD_POD_ID" ]; then
    echo "  https://$RUNPOD_POD_ID-3000.proxy.runpod.net"
else
    echo "  http://localhost:3000"
fi
echo ""
echo "🔄 To restart later:"
echo "  bash $PROJECT_DIR/start_auto_processing.sh"