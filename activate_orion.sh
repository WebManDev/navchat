#!/bin/bash

# Orion Environment Activation Script
# This script activates the Orion conda environment and sets necessary environment variables

echo "🚀 Activating Orion environment..."

# Activate conda environment
conda activate orion

# Set environment variables to fix common issues
export KMP_DUPLICATE_LIB_OK=TRUE

echo "✅ Orion environment activated!"
echo "📁 Working directory: $(pwd)"
echo "🐍 Python: $(which python)"
echo ""
echo "🎯 You can now run Orion commands like:"
echo "   python demos/play_chatgpt_api.py"
echo "   python demos/play_lseg.py"
echo "   python demos/play_groundingSAM.py"
echo ""
echo "⚠️  Note: Some components may show warnings about PyTorch/torchvision compatibility"
echo "   This is normal and doesn't affect functionality."
