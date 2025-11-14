"""
Debug script to compare working curl request with LangChain ChatOpenAI request
for port 8002 to identify what's causing the 400 Bad Request error.
"""
import sys
import os
import json
import httpx
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.llm import LLM
from langchain_core.messages import HumanMessage

def test_direct_http_request():
    """Test the exact same request as the working curl command."""
    print("=" * 80)
    print("TEST 1: Direct HTTP Request (matching working curl)")
    print("=" * 80)
    
    url = "http://100.96.237.56:8002/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer EMPTY"
    }
    payload = {
        "model": "Zyphra/Zamba2-7B-instruct",
        "messages": [
            {"role": "user", "content": "Hello, say something simple."}
        ]
    }
    
    print(f"URL: {url}")
    print(f"Headers: {json.dumps(headers, indent=2)}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print()
    
    try:
        response = httpx.post(url, headers=headers, json=payload, timeout=30.0)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Body: {response.text[:500]}")
        
        if response.status_code == 200:
            print("✅ SUCCESS: Direct HTTP request works!")
            return True
        else:
            print(f"❌ FAILED: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_langchain_request(streaming=False):
    """Test what LangChain ChatOpenAI actually sends."""
    print("\n" + "=" * 80)
    print(f"TEST 2: LangChain ChatOpenAI Request (streaming={streaming})")
    print("=" * 80)
    
    try:
        llm = LLM(
            provider='chatopenai',
            model_name='Zyphra/Zamba2-7B-instruct',
            api_key="EMPTY",
            base_url="http://100.96.237.56:8002/v1",
            streaming=streaming
        )
        
        print(f"LLM Provider: {llm.provider}")
        print(f"Model Name: {llm.model_name}")
        print(f"Streaming: {streaming}")
        
        # Get the underlying client
        chat = llm.create_chat()
        print(f"Chat client type: {type(chat)}")
        
        # Try to inspect the client configuration
        if hasattr(chat, 'client'):
            client = chat.client
            print(f"Client type: {type(client)}")
            if hasattr(client, 'model_name'):
                print(f"Client model_name: {client.model_name}")
            if hasattr(client, 'model'):
                print(f"Client model: {client.model}")
            if hasattr(client, 'base_url'):
                print(f"Client base_url: {client.base_url}")
            if hasattr(client, 'streaming'):
                print(f"Client streaming: {client.streaming}")
        
        print("\nAttempting to invoke...")
        response = chat.invoke([HumanMessage(content="Hello, say something simple.")])
        
        print(f"✅ SUCCESS: Got response")
        print(f"Response type: {type(response)}")
        if hasattr(response, 'content'):
            print(f"Response content (first 200 chars): {response.content[:200]}")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Try to extract more error details
        error_str = str(e)
        print(f"Error string: {error_str}")
        
        # Check if it's an HTTP error
        if hasattr(e, 'response'):
            print(f"\nHTTP Response details:")
            if hasattr(e.response, 'status_code'):
                print(f"  Status Code: {e.response.status_code}")
            if hasattr(e.response, 'headers'):
                print(f"  Headers: {dict(e.response.headers)}")
            if hasattr(e.response, 'text'):
                print(f"  Response Text: {e.response.text[:500]}")
            if hasattr(e.response, 'json'):
                try:
                    print(f"  Response JSON: {json.dumps(e.response.json(), indent=2)}")
                except:
                    pass
        
        return False

def test_with_request_interception():
    """Test with request interception to see exact request being sent."""
    print("\n" + "=" * 80)
    print("TEST 3: Intercepting actual HTTP request from LangChain")
    print("=" * 80)
    
    # Monkey patch httpx to log requests
    original_post = httpx.post
    
    def logged_post(*args, **kwargs):
        print("\n📤 HTTP Request being sent:")
        print(f"  URL: {args[0] if args else kwargs.get('url', 'N/A')}")
        print(f"  Method: POST")
        
        if 'headers' in kwargs:
            print(f"  Headers:")
            for k, v in kwargs['headers'].items():
                print(f"    {k}: {v}")
        
        if 'json' in kwargs:
            print(f"  JSON Body:")
            print(json.dumps(kwargs['json'], indent=4))
        elif 'data' in kwargs:
            print(f"  Data: {kwargs['data']}")
        
        # Make the actual request
        response = original_post(*args, **kwargs)
        
        print(f"\n📥 HTTP Response received:")
        print(f"  Status Code: {response.status_code}")
        print(f"  Response (first 500 chars): {response.text[:500]}")
        
        return response
    
    # Temporarily replace httpx.post
    import httpx
    httpx.post = logged_post
    
    try:
        # Also need to patch the client's httpx usage
        # This is trickier, so let's just test and see what happens
        test_langchain_request(streaming=False)
    finally:
        # Restore original
        httpx.post = original_post

def compare_requests():
    """Compare the working request format with what LangChain might be sending."""
    print("\n" + "=" * 80)
    print("TEST 4: Comparing Request Formats")
    print("=" * 80)
    
    # Working request format (from curl)
    working_format = {
        "model": "Zyphra/Zamba2-7B-instruct",
        "messages": [
            {"role": "user", "content": "Hello, say something simple."}
        ]
    }
    
    print("Working request format (from curl):")
    print(json.dumps(working_format, indent=2))
    
    print("\nPossible LangChain variations:")
    
    # Check what parameters LangChain might add
    variations = [
        {"model": "Zyphra/Zamba2-7B-instruct", "messages": [{"role": "user", "content": "Hello"}], "stream": False},
        {"model": "Zyphra/Zamba2-7B-instruct", "messages": [{"role": "user", "content": "Hello"}], "temperature": 0.7},
        {"model": "Zyphra/Zamba2-7B-instruct", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": None},
    ]
    
    for i, var in enumerate(variations, 1):
        print(f"\nVariation {i}:")
        print(json.dumps(var, indent=2, default=str))

def test_minimal_langchain():
    """Test with minimal LangChain configuration."""
    print("\n" + "=" * 80)
    print("TEST 5: Minimal LangChain Configuration")
    print("=" * 80)
    
    try:
        from langchain_openai import ChatOpenAI
        
        # Try with absolute minimum config
        chat = ChatOpenAI(
            base_url="http://100.96.237.56:8002/v1",
            model="Zyphra/Zamba2-7B-instruct",
            api_key="EMPTY",
            streaming=False,
            temperature=0.7,
        )
        
        print("Configuration:")
        print(f"  base_url: {chat.base_url}")
        print(f"  model: {chat.model_name}")
        print(f"  streaming: {chat.streaming}")
        print(f"  temperature: {chat.temperature}")
        
        print("\nAttempting invoke...")
        response = chat.invoke([HumanMessage(content="Hello")])
        print(f"✅ SUCCESS: {response.content[:100]}")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print(f"Error type: {type(e).__name__}")
        if hasattr(e, 'response'):
            try:
                print(f"Response: {e.response.text[:500]}")
            except:
                pass
        return False

if __name__ == "__main__":
    print("🔍 Debugging Port 8002 400 Bad Request Issue")
    print("=" * 80)
    
    # Test 1: Direct HTTP (should work)
    test1_result = test_direct_http_request()
    
    # Test 2: LangChain without streaming
    test2_result = test_langchain_request(streaming=False)
    
    # Test 3: LangChain with streaming (should fail)
    test3_result = test_langchain_request(streaming=True)
    
    # Test 4: Compare formats
    compare_requests()
    
    # Test 5: Minimal LangChain
    test5_result = test_minimal_langchain()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Direct HTTP Request: {'✅ PASS' if test1_result else '❌ FAIL'}")
    print(f"LangChain (no streaming): {'✅ PASS' if test2_result else '❌ FAIL'}")
    print(f"LangChain (streaming): {'✅ PASS' if test3_result else '❌ FAIL'}")
    print(f"Minimal LangChain: {'✅ PASS' if test5_result else '❌ FAIL'}")
    
    if not test2_result:
        print("\n💡 Suggestion: Check the actual HTTP request being sent by LangChain")
        print("   The server might be rejecting additional parameters or headers")

