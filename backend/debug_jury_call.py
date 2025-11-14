"""
Debug script to test the actual jury evaluator call that's failing.
This simulates how the jury evaluator actually invokes the LLM.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.llm import LLM
from models.jury import create_jury
from langchain_core.messages import HumanMessage
import traceback

def test_jury_evaluator_call():
    """Test the exact way the jury evaluator calls the LLM."""
    print("=" * 80)
    print("TEST: Jury Evaluator Call (simulating actual usage)")
    print("=" * 80)
    
    # This is the exact config from test_evaluation.py
    jury_evaluator_configs = [
        {'provider': 'chatopenai', 'model_name': 'ibm-granite/granite-3.3-8b-instruct', 'api_key': "EMPTY", "base_url": "http://100.96.237.56:8000/v1"},
        {'provider': 'chatopenai', 'model_name': 'openai/gpt-oss-20b', 'api_key': "EMPTY", "base_url": "http://100.96.237.56:8001/v1"},
        {'provider': 'chatopenai', 'model_name': 'Zyphra/Zamba2-7B-instruct', 'api_key': "EMPTY", "base_url": "http://100.96.237.56:8002/v1", "streaming": False},
    ]
    
    print(f"Creating jury with {len(jury_evaluator_configs)} members...")
    jury = create_jury(jury_evaluator_configs, voting_strategy="weighted")
    
    # Create a test prompt similar to what the evaluator sends
    test_prompt = """EVALUATION TASK: RETRIEVAL_RELEVANCE

Query: What is machine learning?

Response: Machine learning is a subset of artificial intelligence.

Context Documents:
Document 1: Machine learning is a method of data analysis.

Evaluate the retrieval relevance on a scale of 0-10."""
    
    print(f"\nTest prompt (first 200 chars): {test_prompt[:200]}...")
    print(f"Prompt type: {type(test_prompt)}")
    print(f"Prompt length: {len(test_prompt)}")
    
    print("\nCalling jury.deliberate()...")
    try:
        result = jury.deliberate(test_prompt, return_individual_responses=True)
        print("✅ SUCCESS: Jury deliberation completed")
        print(f"Result type: {type(result)}")
        
        if isinstance(result, dict):
            print(f"Result keys: {result.keys()}")
            if 'individual_responses' in result:
                print(f"\nIndividual responses:")
                for i, resp in enumerate(result['individual_responses']):
                    print(f"  Member {i+1}:")
                    print(f"    Success: {resp.get('success', 'N/A')}")
                    print(f"    Model: {resp.get('model', 'N/A')}")
                    if not resp.get('success'):
                        print(f"    Error: {resp.get('error', 'N/A')}")
                    else:
                        print(f"    Response (first 100 chars): {resp.get('response', 'N/A')[:100]}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print(f"Error type: {type(e).__name__}")
        traceback.print_exc()
        return False

def test_direct_llm_call_with_string():
    """Test calling LLM directly with a string (like jury does)."""
    print("\n" + "=" * 80)
    print("TEST: Direct LLM Call with String (matching jury behavior)")
    print("=" * 80)
    
    llm = LLM(
        provider='chatopenai',
        model_name='Zyphra/Zamba2-7B-instruct',
        api_key="EMPTY",
        base_url="http://100.96.237.56:8002/v1",
        streaming=False
    )
    
    test_prompt = "Hello, say something simple."
    
    print(f"Calling chat.invoke() with string: '{test_prompt}'")
    print(f"String type: {type(test_prompt)}")
    
    try:
        chat = llm.create_chat()
        print(f"Chat client type: {type(chat)}")
        
        # This is what the jury does - passes a string directly
        response = chat.invoke(test_prompt)
        
        print(f"✅ SUCCESS: Got response")
        print(f"Response type: {type(response)}")
        if hasattr(response, 'content'):
            print(f"Response content: {response.content[:200]}")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print(f"Error type: {type(e).__name__}")
        traceback.print_exc()
        return False

def test_direct_llm_call_with_message():
    """Test calling LLM with proper message format."""
    print("\n" + "=" * 80)
    print("TEST: Direct LLM Call with HumanMessage (proper format)")
    print("=" * 80)
    
    llm = LLM(
        provider='chatopenai',
        model_name='Zyphra/Zamba2-7B-instruct',
        api_key="EMPTY",
        base_url="http://100.96.237.56:8002/v1",
        streaming=False
    )
    
    test_message = HumanMessage(content="Hello, say something simple.")
    
    print(f"Calling chat.invoke() with HumanMessage")
    print(f"Message type: {type(test_message)}")
    print(f"Message content: {test_message.content}")
    
    try:
        chat = llm.create_chat()
        response = chat.invoke([test_message])
        
        print(f"✅ SUCCESS: Got response")
        print(f"Response type: {type(response)}")
        if hasattr(response, 'content'):
            print(f"Response content: {response.content[:200]}")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print(f"Error type: {type(e).__name__}")
        traceback.print_exc()
        return False

def test_long_prompt():
    """Test with a longer prompt that might cause issues."""
    print("\n" + "=" * 80)
    print("TEST: Long Prompt (simulating evaluation prompt)")
    print("=" * 80)
    
    llm = LLM(
        provider='chatopenai',
        model_name='Zyphra/Zamba2-7B-instruct',
        api_key="EMPTY",
        base_url="http://100.96.237.56:8002/v1",
        streaming=False
    )
    
    # Create a long prompt similar to evaluation prompts
    long_prompt = """EVALUATION TASK: RETRIEVAL_RELEVANCE

Query: What is machine learning?

Response: Machine learning is a subset of artificial intelligence that enables systems to learn from data.

Context Documents:
Document 1: Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.
Document 2: There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.
Document 3: Machine learning algorithms build mathematical models based on training data in order to make predictions or decisions.

Evaluate the retrieval relevance. Consider:
1. How well do the retrieved documents address the query?
2. Are the documents relevant to the question asked?
3. Do the documents provide useful context for the response?

Provide a score from 0-10 and reasoning."""
    
    print(f"Prompt length: {len(long_prompt)} characters")
    print(f"Prompt (first 300 chars): {long_prompt[:300]}...")
    
    try:
        chat = llm.create_chat()
        response = chat.invoke(long_prompt)
        
        print(f"✅ SUCCESS: Got response")
        if hasattr(response, 'content'):
            print(f"Response content: {response.content[:200]}")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print(f"Error type: {type(e).__name__}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 Debugging Jury Evaluator Call for Port 8002")
    print("=" * 80)
    
    # Test 1: Direct call with string
    test1 = test_direct_llm_call_with_string()
    
    # Test 2: Direct call with message
    test2 = test_direct_llm_call_with_message()
    
    # Test 3: Long prompt
    test3 = test_long_prompt()
    
    # Test 4: Actual jury call
    test4 = test_jury_evaluator_call()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Direct LLM call (string): {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Direct LLM call (message): {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"Long prompt test: {'✅ PASS' if test3 else '❌ FAIL'}")
    print(f"Jury evaluator call: {'✅ PASS' if test4 else '❌ FAIL'}")

