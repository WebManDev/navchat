# 🚀 ORION Quick Setup for M2 Pro Mac

## ✅ What's Already Working
- ✅ PyTorch MPS support (M2 Pro GPU acceleration)
- ✅ Scene data (TEEsavR23oF scene available)
- ✅ Pretrained models (SAM, GroundingDINO, LSeg)
- ✅ API keys configured
- ✅ CUDA dependencies removed

## 🔧 Quick Installation Steps

### 1. Install Core Dependencies
```bash
# Install PyTorch for M2 Pro (already working!)
pip install torch torchvision torchaudio

# Install basic requirements
pip install opencv-python transformers scipy matplotlib
pip install imageio ftfy regex tqdm timm
pip install open_clip_torch openai scikit-fmm scikit-image
pip install Pillow requests gradio
```

### 2. Install Habitat-Sim (M2 Pro compatible)
```bash
# Install habitat-sim for Apple Silicon
pip install habitat-sim==0.2.2
```

### 3. Install CLIP
```bash
pip install git+https://github.com/openai/CLIP.git
```

### 4. Install Submodules
```bash
# Install habitat-lab
cd third_party/habitat-lab/
pip install -r requirements.txt
python setup.py develop --all
cd ../..

# Install GroundingSAM (without CUDA)
cd third_party/Grounded-Segment-Anything
export BUILD_WITH_CUDA=False
python -m pip install -e segment_anything
python -m pip install -e GroundingDINO
cd ../..
```

### 5. Install ORION
```bash
python setup.py develop
```

## 🎮 Quick Start Commands

### Test Setup
```bash
python test_setup.py
```

### Simple Demo
```bash
python simple_demo.py
```

### Full Interactive Demo
```bash
# Terminal-based interaction
python demos/play_interactive_terminal.py --scene_id TEEsavR23oF --method-type orion

# Gradio web interface
python demos/play_interactive_gradio.py

# Test ChatGPT API
python demos/play_chatgpt_api.py --api-type openai --model-type gpt35
```

## 🎯 Available Scenes
- `TEEsavR23oF` (recommended for testing)
- `wcojb4TFT35`
- `k1cupFYWXJ6`
- `BHXhpBwSMLh`
- And more in `data/scene_datasets/`

## 🚀 What You Can Do
1. **Chat with ORION**: Natural language navigation commands
2. **Visual Navigation**: "Go to the chair", "Find the kitchen"
3. **Interactive Exploration**: Real-time environment interaction
4. **Web Interface**: Gradio-based GUI for easy interaction

## 🔧 Troubleshooting
- If MPS issues occur, the code will automatically fall back to CPU
- All CUDA dependencies have been removed for M2 Pro compatibility
- Scene data and models are already available in your setup

## 📝 Next Steps
1. Run the installation commands above
2. Test with `python test_setup.py`
3. Try the simple demo: `python simple_demo.py`
4. Launch full ORION: `python demos/play_interactive_terminal.py --scene_id TEEsavR23oF`
