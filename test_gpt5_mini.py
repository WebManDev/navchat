#!/usr/bin/env python3
"""
Test GPT-5-mini deployment for ORION
"""

import requests
import json

def test_gpt5_mini():
    """Test GPT-5-mini deployment"""
    print("🚀 Testing GPT-5-mini deployment...")
    
    api_key = "BgRXUeusbzVlVKMVQ05BEFkG5fH5HvLWsTiqF8DZVCwgFXa7TCIKJQQJ99BJACYeBjFXJ3w3AAABACOG0kpo"
    endpoint = "https://research1212.openai.azure.com/"
    deployment_name = "gpt-5-mini"
    
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json"
    }
    
    data = {
        "messages": [
            {"role": "user", "content": "Hello! I'm testing ORION navigation. Can you help me find a chair in the room?"}
        ],
        "max_tokens": 150
    }
    
    try:
        response = requests.post(
            f"{endpoint}openai/deployments/{deployment_name}/chat/completions?api-version=2023-12-01-preview",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ GPT-5-mini deployment working!")
            print(f"Response: {result['choices'][0]['message']['content']}")
            return True
        else:
            print(f"❌ Deployment Error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_orion_chat():
    """Test ORION ChatGPT integration"""
    print("\n🤖 Testing ORION ChatGPT integration...")
    try:
        from orion.chatgpt.api import ChatAPI
        from orion.config.chatgpt_config import AzureGPT35Config
        
        chat_api = ChatAPI(config=AzureGPT35Config())
        chat_api.add_user_message("Hello! Can you help me navigate to a chair?")
        response = chat_api.get_system_response()
        print(f"✅ ORION ChatGPT working!")
        print(f"Response: {response}")
        return True
    except Exception as e:
        print(f"❌ ORION ChatGPT failed: {e}")
        return False

def main():
    print("🎯 GPT-5-mini ORION Test")
    print("=" * 40)
    
    # Test direct deployment
    deployment_works = test_gpt5_mini()
    
    if deployment_works:
        # Test ORION integration
        orion_works = test_orion_chat()
        
        print("\n" + "=" * 40)
        print("📊 Results:")
        print(f"GPT-5-mini Deployment: {'✅ Working' if deployment_works else '❌ Failed'}")
        print(f"ORION Integration: {'✅ Working' if orion_works else '❌ Failed'}")
        
        if orion_works:
            print("\n🎉 ORION is ready with GPT-5-mini!")
            print("You can now run: python demos/play_interactive_terminal.py")
        else:
            print("\n⚠️ Deployment works but ORION integration needs fixing")
    else:
        print("\n❌ GPT-5-mini deployment failed. Check Azure portal.")

if __name__ == "__main__":
    main()
