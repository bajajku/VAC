"""
Debug script to run an actual evaluation and capture the exact failing request.
This will help identify what's different during real evaluation runs.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rag_evaluator import create_rag_evaluator
from models.llm import LLM
from core.app import get_app
from utils.prompt import Prompt
from config.constants import PROMPT_TEMPLATE
import traceback
import logging

# Enable detailed logging for httpx to see actual requests
logging.basicConfig(level=logging.DEBUG)
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.DEBUG)

def test_actual_evaluation():
    """Run an actual evaluation to see where it fails."""
    print("=" * 80)
    print("TEST: Actual RAG Evaluation")
    print("=" * 80)
    
    # Initialize the evaluation system exactly as in test_evaluation.py
    jury_evaluator_configs = [
        {'provider': 'chatopenai', 'model_name': 'ibm-granite/granite-3.3-8b-instruct', 'api_key': "EMPTY", "base_url": "http://100.96.237.56:8000/v1"},
        {'provider': 'chatopenai', 'model_name': 'openai/gpt-oss-20b', 'api_key': "EMPTY", "base_url": "http://100.96.237.56:8001/v1"},
        {'provider': 'chatopenai', 'model_name': 'Zyphra/Zamba2-7B-instruct', 'api_key': "EMPTY", "base_url": "http://100.96.237.56:8002/v1", "streaming": False},
    ]
    
    print("Creating RAG evaluator...")
    evaluator = create_rag_evaluator(jury_evaluator_configs)
    
    # Create a simple test case
    test_query = "What is machine learning?"
    test_response = "Machine learning is a subset of artificial intelligence."
    test_context = [
        "Machine learning is a method of data analysis that automates analytical model building.",
        "There are three main types of machine learning: supervised, unsupervised, and reinforcement learning."
    ]
    
    print(f"\nTest Query: {test_query}")
    print(f"Test Response: {test_response}")
    print(f"Test Context: {len(test_context)} documents")
    
    print("\nRunning evaluation...")
    try:
        # Evaluate just one criterion to isolate the issue
        from models.rag_evaluator import EvaluationCriteria
        
        report = evaluator.evaluate_rag_response(
            query=test_query,
            response=test_response,
            context_documents=test_context,
            criteria=[EvaluationCriteria.RETRIEVAL_RELEVANCE]  # Just one criterion
        )
        
        print("✅ SUCCESS: Evaluation completed")
        print(f"Overall Score: {report.overall_score}")
        print(f"Pass/Fail: {report.overall_pass_fail}")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print(f"Error type: {type(e).__name__}")
        traceback.print_exc()
        return False

def test_with_request_logging():
    """Test with request/response logging enabled."""
    print("\n" + "=" * 80)
    print("TEST: With Request/Response Logging")
    print("=" * 80)
    
    # Monkey patch to log requests
    import httpx
    
    original_request = httpx.Client.request
    
    def logged_request(self, method, url, **kwargs):
        print(f"\n📤 REQUEST:")
        print(f"  Method: {method}")
        print(f"  URL: {url}")
        if 'headers' in kwargs:
            print(f"  Headers: {dict(kwargs['headers'])}")
        if 'json' in kwargs:
            import json
            print(f"  JSON Body:")
            print(json.dumps(kwargs['json'], indent=2))
        
        try:
            response = original_request(self, method, url, **kwargs)
            print(f"\n📥 RESPONSE:")
            print(f"  Status: {response.status_code}")
            if response.status_code != 200:
                print(f"  Response Text: {response.text[:500]}")
            return response
        except Exception as e:
            print(f"\n❌ REQUEST FAILED: {e}")
            raise
    
    # Patch the client
    httpx.Client.request = logged_request
    
    try:
        return test_actual_evaluation()
    finally:
        # Restore
        httpx.Client.request = original_request

def test_concurrent_calls():
    """Test if concurrent calls cause the issue."""
    print("\n" + "=" * 80)
    print("TEST: Concurrent Calls (simulating ThreadPoolExecutor)")
    print("=" * 80)
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    llm = LLM(
        provider='chatopenai',
        model_name='Zyphra/Zamba2-7B-instruct',
        api_key="EMPTY",
        base_url="http://100.96.237.56:8002/v1",
        streaming=False
    )
    
    def make_request(i):
        try:
            chat = llm.create_chat()
            response = chat.invoke(f"Test message {i}")
            return {'success': True, 'index': i, 'response': response.content[:50]}
        except Exception as e:
            return {'success': False, 'index': i, 'error': str(e)}
    
    print("Making 5 concurrent requests...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request, i) for i in range(5)]
        results = [f.result() for f in as_completed(futures)]
    
    results.sort(key=lambda x: x['index'])
    
    print("\nResults:")
    for r in results:
        if r['success']:
            print(f"  Request {r['index']}: ✅ SUCCESS")
        else:
            print(f"  Request {r['index']}: ❌ FAILED - {r['error']}")
    
    all_success = all(r['success'] for r in results)
    return all_success

if __name__ == "__main__":
    print("🔍 Debugging Actual Evaluation for Port 8002")
    print("=" * 80)
    
    # Test 1: Concurrent calls
    test1 = test_concurrent_calls()
    
    # Test 2: Actual evaluation
    test2 = test_actual_evaluation()
    
    # Test 3: With logging
    test3 = test_with_request_logging()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Concurrent calls: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Actual evaluation: {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"With request logging: {'✅ PASS' if test3 else '❌ FAIL'}")

