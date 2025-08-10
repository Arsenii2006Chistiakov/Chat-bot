#!/usr/bin/env python3
"""
Test script for the MongoDB Chatbot API
Demonstrates how to use the unified /chat endpoint
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

# Test user ID
USER_ID = "test_user_123"

def test_health_check():
    """Test the health check endpoint"""
    print("🏥 Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_root_endpoint():
    """Test the root endpoint"""
    print("🏠 Testing root endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_general_chat():
    """Test general chat functionality"""
    print("💬 Testing general chat...")
    
    chat_data = {
        "message": "Hello! How are you?",
        "user_id": USER_ID
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=chat_data)
    print(f"Chat Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result['response']}")
        print(f"Category: {result['category']}")
    else:
        print(f"Error: {response.text}")
    print()

def test_help_command():
    """Test the help command"""
    print("❓ Testing help command...")
    
    chat_data = {
        "message": "/help",
        "user_id": USER_ID
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=chat_data)
    print(f"Help Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result['response'][:200]}...")  # Truncate for display
        print(f"Category: {result['category']}")
    else:
        print(f"Error: {response.text}")
    print()

def test_load_command_help():
    """Test the /load command help"""
    print("📁 Testing /load command help...")
    
    chat_data = {
        "message": "/load",
        "user_id": USER_ID
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=chat_data)
    print(f"Load Help Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result['response'][:200]}...")  # Truncate for display
        print(f"Category: {result['category']}")
    else:
        print(f"Error: {response.text}")
    print()

def test_load_text_only():
    """Test loading text only"""
    print("📝 Testing /load --text command...")
    
    chat_data = {
        "message": "/load --text 'This is a test song lyrics content for embedding'",
        "user_id": USER_ID
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=chat_data)
    print(f"Load Text Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result['response']}")
        print(f"Category: {result['category']}")
        print(f"Context Count: {result['context_count']}")
    else:
        print(f"Error: {response.text}")
    print()

def test_search_functionality():
    """Test search functionality through chat"""
    print("🔍 Testing search functionality...")
    
    search_data = {
        "message": "find me some popular songs",
        "user_id": USER_ID
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=search_data)
    print(f"Search Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result['response'][:300]}...")  # Truncate for display
        print(f"Category: {result['category']}")
        print(f"Context Count: {result['context_count']}")
        
        # Check if we got search results
        if result.get('search_results'):
            print(f"Search Results Count: {len(result['search_results'])}")
        else:
            print("No search results in response")
    else:
        print(f"Error: {response.text}")
    print()

def test_add_to_context():
    """Test adding a search result to context"""
    print("➕ Testing add to context...")
    
    add_data = {
        "message": "/add 0",
        "user_id": USER_ID
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=add_data)
    print(f"Add to Context Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result['response']}")
        print(f"Category: {result['category']}")
        print(f"Context Count: {result['context_count']}")
    else:
        print(f"Error: {response.text}")
    print()

def test_get_context():
    """Test getting user context"""
    print("📋 Testing get context...")
    
    response = requests.get(f"{BASE_URL}/context/{USER_ID}")
    print(f"Get Context Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Context Items Count: {len(result['context_songs'])}")
        for i, item in enumerate(result['context_songs']):
            if item.get('type') == 'audio_file':
                print(f"  {i+1}. Audio File: {item.get('file_path', 'N/A')}")
            elif item.get('type') == 'text_content':
                print(f"  {i+1}. Text Content: {item.get('text', 'N/A')[:50]}...")
            else:
                print(f"  {i+1}. Song: {item.get('title', 'N/A')} by {item.get('artist', 'N/A')}")
    else:
        print(f"Error: {response.text}")
    print()

def test_analysis_query():
    """Test analysis query with context"""
    print("🧠 Testing analysis query...")
    
    # First, make sure we have some context
    context_response = requests.get(f"{BASE_URL}/context/{USER_ID}")
    if context_response.status_code == 200:
        context_count = len(context_response.json().get('context_songs', []))
        if context_count == 0:
            print("No context available. Adding a search result first...")
            test_search_functionality()
            time.sleep(1)
            test_add_to_context()
            time.sleep(1)
    
    # Now test analysis
    analysis_data = {
        "message": "Explain why this trend became popular",
        "user_id": USER_ID
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=analysis_data)
    print(f"Analysis Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result['response'][:300]}...")  # Truncate for display
        print(f"Category: {result['category']}")
        print(f"Context Count: {result['context_count']}")
    else:
        print(f"Error: {response.text}")
    print()

def test_clear_context():
    """Test clearing context"""
    print("🧹 Testing clear context...")
    
    clear_data = {
        "message": "/clear",
        "user_id": USER_ID
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=clear_data)
    print(f"Clear Context Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result['response']}")
        print(f"Category: {result['category']}")
        print(f"Context Count: {result['context_count']}")
    else:
        print(f"Error: {response.text}")
    print()

def test_chat_history():
    """Test chat history command"""
    print("📚 Testing chat history...")
    
    history_data = {
        "message": "/history",
        "user_id": USER_ID
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=history_data)
    print(f"Chat History Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result['response'][:200]}...")  # Truncate for display
        print(f"Category: {result['category']}")
    else:
        print(f"Error: {response.text}")
    print()

def test_cleanup():
    """Test cleanup endpoints"""
    print("🗑️ Testing cleanup...")
    
    # Remove user
    response = requests.delete(f"{BASE_URL}/user/{USER_ID}")
    print(f"Remove User Status: {response.status_code}")
    print()

def main():
    """Run all tests"""
    print("🚀 Starting Unified Chat API Tests...")
    print("=" * 60)
    
    try:
        test_health_check()
        test_root_endpoint()
        test_general_chat()
        test_help_command()
        test_load_command_help()
        test_load_text_only()
        test_search_functionality()
        test_add_to_context()
        test_get_context()
        test_analysis_query()
        test_clear_context()
        test_chat_history()
        test_cleanup()
        
        print("✅ All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the API. Make sure the server is running on localhost:8000")
        print("💡 Start the server with: python app.py")
    except Exception as e:
        print(f"❌ Error during testing: {e}")

if __name__ == "__main__":
    main()
