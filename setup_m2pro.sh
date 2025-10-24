#!/bin/bash

echo "🚀 Setting up ORION for M2 Pro Mac..."

# Check if we're in the right directory
if [ ! -f "setup.py" ]; then
    echo "❌ Please run this script from the navchat root directory"
    exit 1
fi

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "orion"; then
    echo "📦 Creating conda environment..."
    conda create -n orion python=3.8 -y
fi

echo "🔄 Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate orion

# Install PyTorch for M2 Pro (MPS support)
echo "🔥 Installing PyTorch for M2 Pro..."
pip install torch torchvision torchaudio

# Install habitat-sim for M2 Pro
echo "🏠 Installing habitat-sim..."
pip install habitat-sim==0.2.2

# Install main requirements
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Install submodules
echo "🔧 Setting up submodules..."
git submodule update --init --recursive

# Install habitat-lab
echo "🏗️ Installing habitat-lab..."
cd third_party/habitat-lab/
pip install -r requirements.txt
python setup.py develop --all
cd ../..

# Install GroundingSAM (without CUDA)
echo "🎯 Installing GroundingSAM..."
cd third_party/Grounded-Segment-Anything
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=False  # Disable CUDA for M2 Pro

python -m pip install -e segment_anything
python -m pip install -e GroundingDINO
cd ../..

# Install ORION itself
echo "🎉 Installing ORION..."
python setup.py develop

echo "✅ Setup complete! You can now run ORION demos."
echo ""
echo "🚀 Quick start commands:"
echo "  conda activate orion"
echo "  python demos/play_chatgpt_api.py --api-type openai --model-type gpt35"
echo "  python demos/play_interactive_terminal.py --scene_id TEEsavR23oF --method-type orion"
echo "  python demos/play_interactive_gradio.py"
