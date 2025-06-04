#!/usr/bin/env python3

# Minimal reproduction of the data_cleaner import issue

print("Step 1: Testing individual imports...")
from models.rag_chain import NormalChain
from models.llm import LLM
print("Individual imports successful")

print("Step 2: Testing in a class context...")
class TestClass:
    def __init__(self):
        self.chain = None
    
    def test_method(self):
        print("About to create LLM...")
        llm = LLM(
            provider="chatopenai", 
            model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            api_key="test_key"
        )
        print("LLM created successfully")

print("Step 3: Instantiating test class...")
test = TestClass()
print("Step 4: Calling test method...")
test.test_method()
print("All tests passed!") 