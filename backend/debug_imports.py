#!/usr/bin/env python3

print("Testing imports in data_cleaner order...")

# Test imports in the exact order they appear in data_cleaner.py
try:
    import json
    print("✅ json import successful")
except Exception as e:
    print(f"❌ json import failed: {e}")

try:
    import os
    print("✅ os import successful")
except Exception as e:
    print(f"❌ os import failed: {e}")

try:
    from typing import List, Dict, Any
    print("✅ typing imports successful")
except Exception as e:
    print(f"❌ typing imports failed: {e}")

try:
    from langchain_core.documents import Document
    print("✅ Document import successful")
except Exception as e:
    print(f"❌ Document import failed: {e}")

try:
    from models.rag_chain import NormalChain
    print("✅ NormalChain import successful")
except Exception as e:
    print(f"❌ NormalChain import failed: {e}")

try:
    from models.llm import LLM
    print("✅ LLM import successful")
except Exception as e:
    print(f"❌ LLM import failed: {e}")

try:
    from utils.prompt import Prompt
    print("✅ Prompt import successful")
except Exception as e:
    print(f"❌ Prompt import failed: {e}")

try:
    from dotenv import load_dotenv
    print("✅ load_dotenv import successful")
except Exception as e:
    print(f"❌ load_dotenv import failed: {e}")

print("\nNow testing if we can access LLM after all imports...")
try:
    llm_instance = LLM("test", "test")
    print("✅ LLM class accessible")
except NameError as e:
    print(f"❌ LLM class not accessible: {e}")
except Exception as e:
    print(f"❌ LLM instantiation failed (but class accessible): {e}") 