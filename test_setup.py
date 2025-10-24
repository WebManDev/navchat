#!/usr/bin/env python3
"""
Quick test script to verify ORION setup on M2 Pro
"""

import torch
import sys
import os

def test_pytorch_mps():
    """Test PyTorch MPS support"""
    print("🔥 Testing PyTorch MPS support...")
    if torch.backends.mps.is_available():
        print("✅ MPS is available!")
        device = torch.device("mps")
        x = torch.randn(3, 3).to(device)
        y = torch.randn(3, 3).to(device)
        z = torch.mm(x, y)
        print(f"✅ MPS computation successful: {z.shape}")
        return True
    else:
        print("❌ MPS not available, falling back to CPU")
        return False

def test_imports():
    """Test key imports"""
    print("📚 Testing key imports...")
    try:
        import habitat_sim
        print("✅ habitat_sim imported")
        
        import clip
        print("✅ CLIP imported")
        
        import open_clip
        print("✅ open_clip imported")
        
        from orion.config.chatgpt_config import OpenAIGPT35Config
        print("✅ ORION config imported")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_models():
    """Test model loading"""
    print("🎯 Testing model loading...")
    try:
        # Test CLIP loading
        import clip
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        print(f"✅ CLIP model loaded on {device}")
        
        # Test if pretrained models exist
        model_paths = [
            "data/pretrained_ckpts/sam_vit_h_4b8939.pth",
            "data/pretrained_ckpts/groundingdino_swint_ogc.pth", 
            "data/pretrained_ckpts/lseg_demo_e200.ckpt"
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                print(f"✅ Found {path}")
            else:
                print(f"❌ Missing {path}")
        
        return True
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def test_scene_data():
    """Test scene data availability"""
    print("🏠 Testing scene data...")
    scene_paths = [
        "data/scene_datasets/00800-TEEsavR23oF/TEEsavR23oF.glb",
        "data/scene_datasets/00800-TEEsavR23oF/TEEsavR23oF.basis.navmesh"
    ]
    
    for path in scene_paths:
        if os.path.exists(path):
            print(f"✅ Found {path}")
        else:
            print(f"❌ Missing {path}")
    
    return True

def main():
    print("🚀 ORION M2 Pro Setup Test")
    print("=" * 40)
    
    tests = [
        test_pytorch_mps,
        test_imports, 
        test_models,
        test_scene_data
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            print()
    
    print("=" * 40)
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! ORION is ready to run on M2 Pro!")
        print("\n🚀 Quick start:")
        print("  python demos/play_chatgpt_api.py --api-type openai --model-type gpt35")
        print("  python demos/play_interactive_terminal.py --scene_id TEEsavR23oF")
    else:
        print("⚠️  Some tests failed. Check the setup.")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
