#!/usr/bin/env python3
"""
Simple test to verify what's working on M2 Pro
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

def test_clip():
    """Test CLIP model loading"""
    print("🎯 Testing CLIP model loading...")
    try:
        import clip
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        print(f"✅ CLIP model loaded on {device}")
        
        # Test inference
        import numpy as np
        from PIL import Image
        
        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image = Image.fromarray(dummy_image)
        
        # Preprocess and encode
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
        
        print(f"✅ CLIP inference successful: {image_features.shape}")
        return True
    except Exception as e:
        print(f"❌ CLIP test failed: {e}")
        return False

def test_openai():
    """Test OpenAI API"""
    print("🤖 Testing OpenAI API...")
    try:
        import os
        from openai import OpenAI
        
        # Test with a simple API call
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            print("❌ OPENAI_API_KEY environment variable not set")
            return False
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello! Can you help me navigate to a chair?"}],
            max_tokens=50
        )
        
        print(f"✅ OpenAI API working: {response.choices[0].message.content[:50]}...")
        return True
    except Exception as e:
        print(f"❌ OpenAI API test failed: {e}")
        return False

def test_models():
    """Test if pretrained models exist"""
    print("📁 Testing pretrained models...")
    model_paths = [
        "data/pretrained_ckpts/sam_vit_h_4b8939.pth",
        "data/pretrained_ckpts/groundingdino_swint_ogc.pth", 
        "data/pretrained_ckpts/lseg_demo_e200.ckpt"
    ]
    
    found = 0
    for path in model_paths:
        if os.path.exists(path):
            print(f"✅ Found {path}")
            found += 1
        else:
            print(f"❌ Missing {path}")
    
    return found == len(model_paths)

def test_scene_data():
    """Test scene data availability"""
    print("🏠 Testing scene data...")
    scene_paths = [
        "data/scene_datasets/00800-TEEsavR23oF/TEEsavR23oF.glb",
        "data/scene_datasets/00800-TEEsavR23oF/TEEsavR23oF.basis.navmesh"
    ]
    
    found = 0
    for path in scene_paths:
        if os.path.exists(path):
            print(f"✅ Found {path}")
            found += 1
        else:
            print(f"❌ Missing {path}")
    
    return found == len(scene_paths)

def main():
    print("🚀 ORION M2 Pro Component Test")
    print("=" * 40)
    
    tests = [
        ("PyTorch MPS", test_pytorch_mps),
        ("CLIP Model", test_clip),
        ("OpenAI API", test_openai),
        ("Pretrained Models", test_models),
        ("Scene Data", test_scene_data)
    ]
    
    passed = 0
    for name, test_func in tests:
        print(f"\n--- Testing {name} ---")
        try:
            if test_func():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            print()
    
    print("=" * 40)
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed >= 4:  # At least 4 out of 5 tests should pass
        print("🎉 Most components working! You can run basic ORION demos!")
        print("\n🚀 What you can do:")
        print("1. Test CLIP: python demos/play_lseg.py")
        print("2. Test GradCAM: python demos/play_gradcam.py") 
        print("3. Test GroundingSAM: python demos/play_groundingSAM.py")
        print("4. Test ChatGPT: python demos/play_chatgpt_api.py --api-type openai --model-type gpt35")
    else:
        print("⚠️  Some components need attention.")
    
    return passed >= 4

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
