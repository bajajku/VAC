"""
Debug script to test 14 concurrent requests (matching the 14 criteria evaluation).
This will help identify if it's a concurrency issue with port 8002.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.llm import LLM
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

def test_14_concurrent_requests():
    """Test 14 concurrent requests to port 8002 (matching 14 criteria)."""
    print("=" * 80)
    print("TEST: 14 Concurrent Requests to Port 8002")
    print("=" * 80)
    
    # Create LLM with streaming=False (matching the config)
    llm = LLM(
        provider='chatopenai',
        model_name='Zyphra/Zamba2-7B-instruct',
        api_key="EMPTY",
        base_url="http://100.96.237.56:8002/v1",
        streaming=False
    )
    
    # Verify streaming is actually False
    chat = llm.create_chat()
    if hasattr(chat, 'streaming'):
        print(f"Chat streaming setting: {chat.streaming}")
    if hasattr(chat, 'client') and hasattr(chat.client, 'streaming'):
        print(f"Client streaming setting: {chat.client.streaming}")
    
    # Create a longer prompt similar to evaluation prompts
    test_prompt = """EVALUATION TASK: RETRIEVAL_RELEVANCE

Query: Why do I still get panic attacks years after deployment?

Response: Panic attacks can persist for various reasons after deployment.

Context Documents:
Document 1: Post-deployment panic attacks can be related to trauma experiences.
Document 2: Treatment options include therapy and medication.

Evaluate the retrieval relevance on a scale of 0-10."""
    
    def make_request(i):
        try:
            chat = llm.create_chat()
            response = chat.invoke(f"{test_prompt}\n\nRequest {i}")
            return {
                'success': True, 
                'index': i, 
                'response': response.content[:50] if hasattr(response, 'content') else str(response)[:50]
            }
        except Exception as e:
            error_details = str(e)
            if hasattr(e, 'response'):
                try:
                    if hasattr(e.response, 'status_code'):
                        error_details += f" | Status: {e.response.status_code}"
                    if hasattr(e.response, 'text'):
                        error_details += f" | Response: {e.response.text[:200]}"
                except:
                    pass
            return {
                'success': False, 
                'index': i, 
                'error': error_details,
                'error_type': type(e).__name__
            }
    
    print(f"\nMaking 14 concurrent requests with prompt length: {len(test_prompt)} chars...")
    print("This simulates the 14 criteria being evaluated in parallel.\n")
    
    with ThreadPoolExecutor(max_workers=14) as executor:
        futures = [executor.submit(make_request, i) for i in range(14)]
        results = [f.result() for f in as_completed(futures)]
    
    results.sort(key=lambda x: x['index'])
    
    print("\nResults:")
    success_count = 0
    fail_count = 0
    for r in results:
        if r['success']:
            print(f"  Request {r['index']:2d}: ✅ SUCCESS - {r['response']}")
            success_count += 1
        else:
            print(f"  Request {r['index']:2d}: ❌ FAILED ({r['error_type']})")
            print(f"              Error: {r['error'][:200]}")
            fail_count += 1
    
    print(f"\nSummary: {success_count} succeeded, {fail_count} failed")
    return fail_count == 0

def test_with_different_prompt_lengths():
    """Test if prompt length affects the 400 error."""
    print("\n" + "=" * 80)
    print("TEST: Different Prompt Lengths")
    print("=" * 80)
    
    llm = LLM(
        provider='chatopenai',
        model_name='Zyphra/Zamba2-7B-instruct',
        api_key="EMPTY",
        base_url="http://100.96.237.56:8002/v1",
        streaming=False
    )
    
    prompts = [
        ("Short", "Hello"),
        ("Medium", "EVALUATION TASK: RETRIEVAL_RELEVANCE\n\nQuery: Test\n\nResponse: Test response."),
        ("Long", """EVALUATION TASK: RETRIEVAL_RELEVANCE

Query: Why do I still get panic attacks years after deployment?

Response: Panic attacks can persist for various reasons after deployment, including unresolved trauma, ongoing stress, and physiological responses to triggers.

Context Documents:
Document 1: Post-deployment panic attacks can be related to trauma experiences during service. Many veterans experience persistent anxiety and panic symptoms.
Document 2: Treatment options include cognitive behavioral therapy, exposure therapy, and medication management.
Document 3: Panic attacks may be triggered by specific memories, sounds, or situations that remind the individual of their deployment experiences.

Evaluate the retrieval relevance on a scale of 0-10. Consider how well the retrieved documents address the query."""),
    ]
    
    for name, prompt in prompts:
        print(f"\nTesting {name} prompt ({len(prompt)} chars)...")
        try:
            chat = llm.create_chat()
            response = chat.invoke(prompt)
            print(f"  ✅ SUCCESS")
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                print(f"  Response: {e.response.text[:200]}")

if __name__ == "__main__":
    print("🔍 Debugging 14 Concurrent Requests (Matching 14 Criteria)")
    print("=" * 80)
    
    # Test 1: 14 concurrent requests
    test1 = test_14_concurrent_requests()
    
    # Test 2: Different prompt lengths
    test_with_different_prompt_lengths()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"14 Concurrent requests: {'✅ PASS' if test1 else '❌ FAIL'}")

