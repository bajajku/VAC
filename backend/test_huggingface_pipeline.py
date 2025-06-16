#!/usr/bin/env python3
"""
Test script for HuggingFace Pipeline LLM

This script tests the HuggingFace pipeline LLM initialization and basic functionality
to debug issues with the third LLM in the evaluation jury.
"""

import os
import sys
import traceback
from pathlib import Path

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(__file__))

from models.llm import LLM
from dotenv import load_dotenv

load_dotenv()

def test_huggingface_pipeline_basic():
    """Test basic HuggingFace pipeline initialization."""
    print("=" * 60)
    print("TEST 1: Basic HuggingFace Pipeline Initialization")
    print("=" * 60)
    
    try:
        print("🔧 Testing HuggingFace Pipeline LLM initialization...")
        
        # Test with quantization enabled (default)
        llm = LLM(
            provider="huggingface_pipeline",
            model_name="mistralai/Mistral-7B-Instruct-v0.3",
            quantization=True
        )
        
        print("✅ HuggingFace Pipeline LLM initialized successfully (with quantization)!")
        print(f"   Provider: {llm.provider}")
        print(f"   Model: {llm.model_name}")
        
        # Test creating chat
        chat = llm.create_chat()
        print("✅ Chat interface created successfully!")
        
        return llm, True
        
    except Exception as e:
        print(f"❌ Failed to initialize HuggingFace Pipeline LLM (with quantization): {e}")
        print(f"   Error type: {type(e).__name__}")
        traceback.print_exc()
        return None, False

def test_huggingface_pipeline_no_quantization():
    """Test HuggingFace pipeline without quantization."""
    print("\n" + "=" * 60)
    print("TEST 2: HuggingFace Pipeline Without Quantization")
    print("=" * 60)
    
    try:
        print("🔧 Testing HuggingFace Pipeline LLM without quantization...")
        
        llm = LLM(
            provider="huggingface_pipeline",
            model_name="mistralai/Mistral-7B-Instruct-v0.3",
            quantization=False  # Disable quantization
        )
        
        print("✅ HuggingFace Pipeline LLM initialized successfully (without quantization)!")
        print(f"   Provider: {llm.provider}")
        print(f"   Model: {llm.model_name}")
        
        # Test creating chat
        chat = llm.create_chat()
        print("✅ Chat interface created successfully!")
        
        return llm, True
        
    except Exception as e:
        print(f"❌ Failed to initialize HuggingFace Pipeline LLM (without quantization): {e}")
        print(f"   Error type: {type(e).__name__}")
        traceback.print_exc()
        return None, False

def test_huggingface_pipeline_smaller_model():
    """Test with a smaller, more reliable model."""
    print("\n" + "=" * 60)
    print("TEST 3: HuggingFace Pipeline with Smaller Model")
    print("=" * 60)
    
    try:
        print("🔧 Testing HuggingFace Pipeline with smaller model...")
        
        llm = LLM(
            provider="huggingface_pipeline",
            model_name="microsoft/DialoGPT-medium",  # Smaller, more reliable model
            quantization=False
        )
        
        print("✅ HuggingFace Pipeline LLM initialized successfully (smaller model)!")
        print(f"   Provider: {llm.provider}")
        print(f"   Model: {llm.model_name}")
        
        # Test creating chat
        chat = llm.create_chat()
        print("✅ Chat interface created successfully!")
        
        return llm, True
        
    except Exception as e:
        print(f"❌ Failed to initialize HuggingFace Pipeline LLM (smaller model): {e}")
        print(f"   Error type: {type(e).__name__}")
        traceback.print_exc()
        return None, False

def test_llm_response(llm, test_name):
    """Test getting a response from the LLM."""
    print(f"\n📝 Testing response generation for {test_name}...")
    
    try:
        chat = llm.create_chat()
        test_prompt = "What is the capital of France? Answer in one sentence."
        
        print(f"   Prompt: {test_prompt}")
        response = chat.invoke(test_prompt)
        print("____________________")
        print(response)
        # Handle different response types
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
        
        print(f"   Response: {response_text[:100]}...")
        print("✅ Response generation successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to generate response: {e}")
        print(f"   Error type: {type(e).__name__}")
        traceback.print_exc()
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    print("=" * 60)
    print("DEPENDENCY CHECK")
    print("=" * 60)
    
    dependencies = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'bitsandbytes': 'BitsAndBytes (for quantization)',
        'sentencepiece': 'SentencePiece (for tokenization)',
        'accelerate': 'Accelerate (for device mapping)'
    }
    
    missing_deps = []
    
    for module, description in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {description}: Available")
        except ImportError:
            print(f"❌ {description}: Missing")
            missing_deps.append(module)
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("   Install with: pip install " + " ".join(missing_deps))
        return False
    else:
        print("\n✅ All dependencies available!")
        return True

def test_jury_integration():
    """Test the LLM in a jury configuration like the evaluator uses."""
    print("\n" + "=" * 60)
    print("TEST 4: Jury Integration Test")
    print("=" * 60)
    
    try:
        from models.jury import create_jury
        
        # Same configuration as in the evaluator
        jury_configs = [
            {
                "provider": "huggingface_pipeline",
                "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
                "quantization": False  # Start without quantization
            }
        ]
        
        print("🔧 Creating jury with HuggingFace Pipeline LLM...")
        jury = create_jury(jury_configs, voting_strategy="majority")
        
        print("✅ Jury created successfully!")
        print(f"   Jury size: {jury.jury_size}")
        
        # Test deliberation
        test_prompt = "Rate the quality of this response on a scale of 1-10: 'The sky is blue.' Respond in JSON format: {\"score\": X, \"reasoning\": \"...\", \"confidence\": Y}"
        
        print("📝 Testing jury deliberation...")
        result = jury.deliberate(test_prompt, return_individual_responses=True)
        
        print("✅ Jury deliberation successful!")
        print(f"   Consensus: {result['consensus'][:100]}...")
        print(f"   Individual responses: {len(result['individual_responses'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Jury integration test failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🚀 HUGGINGFACE PIPELINE LLM TESTING")
    print("=" * 60)
    print("This script will test the HuggingFace Pipeline LLM to identify")
    print("why it's not working in the evaluation jury.")
    print()
    
    # Check dependencies first
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n⚠️  Some dependencies are missing. This may cause test failures.")
        print("   Consider installing missing packages before proceeding.")
        input("   Press Enter to continue anyway, or Ctrl+C to exit...")
    
    # Run tests
    test_results = []
    
    # Test 1: Basic initialization with quantization
    llm1, success1 = test_huggingface_pipeline_basic()
    test_results.append(("Basic (with quantization)", success1))
    
    if success1 and llm1:
        response_success = test_llm_response(llm1, "basic with quantization")
        test_results.append(("Response (with quantization)", response_success))
    
    # Test 2: Without quantization
    llm2, success2 = test_huggingface_pipeline_no_quantization()
    test_results.append(("Without quantization", success2))
    
    if success2 and llm2:
        response_success = test_llm_response(llm2, "without quantization")
        test_results.append(("Response (without quantization)", response_success))
    
    # Test 3: Smaller model
    llm3, success3 = test_huggingface_pipeline_smaller_model()
    test_results.append(("Smaller model", success3))
    
    if success3 and llm3:
        response_success = test_llm_response(llm3, "smaller model")
        test_results.append(("Response (smaller model)", response_success))
    
    # Test 4: Jury integration
    jury_success = test_jury_integration()
    test_results.append(("Jury integration", jury_success))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, success in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    successful_tests = sum(1 for _, success in test_results if success)
    total_tests = len(test_results)
    
    print(f"\nResults: {successful_tests}/{total_tests} tests passed")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if successful_tests == total_tests:
        print("🎉 All tests passed! HuggingFace Pipeline LLM should work in the jury.")
    elif any(success for _, success in test_results):
        print("⚠️  Some tests passed. Consider using the working configuration:")
        if test_results[1][1]:  # Without quantization worked
            print("   - Disable quantization: quantization=False")
        if test_results[2][1]:  # Smaller model worked
            print("   - Use smaller model: microsoft/DialoGPT-medium")
    else:
        print("❌ All tests failed. Consider:")
        print("   1. Installing missing dependencies")
        print("   2. Using a different provider (chatopenai, openrouter)")
        print("   3. Checking available GPU memory")
        print("   4. Using HuggingFace Endpoint instead of Pipeline")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Testing interrupted by user.")
    except Exception as e:
        print(f"\n\n💥 Unexpected error during testing: {e}")
        traceback.print_exc() 