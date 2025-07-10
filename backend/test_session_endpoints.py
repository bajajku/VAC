#!/usr/bin/env python3
"""
Test script for session management endpoints
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_session_endpoints():
    """Test all session management endpoints"""
    
    print("🧪 Testing Session Management Endpoints")
    print("=" * 50)
    
    # Test 1: Create a new session
    print("\n1. Creating new session...")
    response = requests.post(f"{BASE_URL}/sessions/new", json={
        "title": "Test Session",
        "user_id": "test_user"
    })
    
    if response.status_code == 200:
        session_data = response.json()
        session_id = session_data["session_id"]
        print(f"✅ Session created: {session_id}")
        print(f"   Title: {session_data['title']}")
        print(f"   Message count: {session_data['message_count']}")
    else:
        print(f"❌ Failed to create session: {response.status_code}")
        print(response.text)
        return
    
    # Test 2: List sessions
    print("\n2. Listing sessions...")
    response = requests.get(f"{BASE_URL}/sessions")
    
    if response.status_code == 200:
        sessions = response.json()
        print(f"✅ Found {len(sessions)} sessions")
        for session in sessions[:3]:  # Show first 3
            print(f"   - {session['title']} ({session['message_count']} messages)")
    else:
        print(f"❌ Failed to list sessions: {response.status_code}")
    
    # Test 3: Get specific session
    print(f"\n3. Getting session {session_id}...")
    response = requests.get(f"{BASE_URL}/sessions/{session_id}")
    
    if response.status_code == 200:
        session = response.json()
        print(f"✅ Retrieved session: {session['title']}")
        print(f"   Messages: {len(session['messages'])}")
    else:
        print(f"❌ Failed to get session: {response.status_code}")
    
    # Test 4: Send a message to test storage
    print(f"\n4. Sending test message to session {session_id}...")
    response = requests.post(f"{BASE_URL}/stream_async", json={
        "question": "Hello, this is a test message",
        "session_id": session_id
    })
    
    if response.status_code == 200:
        print("✅ Message sent successfully")
        # Wait a bit for processing
        time.sleep(2)
    else:
        print(f"❌ Failed to send message: {response.status_code}")
    
    # Test 5: Get session messages
    print(f"\n5. Getting messages for session {session_id}...")
    response = requests.get(f"{BASE_URL}/sessions/{session_id}/messages")
    
    if response.status_code == 200:
        messages = response.json()
        print(f"✅ Found {len(messages)} messages")
        for msg in messages:
            print(f"   - {msg['sender']}: {msg['content'][:50]}...")
    else:
        print(f"❌ Failed to get messages: {response.status_code}")
    
    # Test 6: Update session title
    print(f"\n6. Updating session title...")
    response = requests.put(f"{BASE_URL}/sessions/{session_id}", json={
        "title": "Updated Test Session"
    })
    
    if response.status_code == 200:
        updated_session = response.json()
        print(f"✅ Session updated: {updated_session['title']}")
    else:
        print(f"❌ Failed to update session: {response.status_code}")
    
    # Test 7: Test MongoDB connection
    print(f"\n7. Testing MongoDB connection...")
    response = requests.get(f"{BASE_URL}/test-mongodb")
    
    if response.status_code == 200:
        mongo_status = response.json()
        print(f"✅ MongoDB status: {mongo_status['connection_status']}")
        print(f"   Database: {mongo_status['database_status']}")
    else:
        print(f"❌ Failed to test MongoDB: {response.status_code}")
    
    print("\n" + "=" * 50)
    print("✅ Session management tests completed!")

if __name__ == "__main__":
    try:
        test_session_endpoints()
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to backend server at http://localhost:8000")
        print("   Make sure the backend is running with: python -m uvicorn api:app_api --reload")
    except Exception as e:
        print(f"❌ Test failed with error: {e}") 