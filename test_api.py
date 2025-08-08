#!/usr/bin/env python3
"""
Test script for the hackathon API endpoint
"""

import requests
import json
import sys

def test_hackathon_api(api_url, document_url, questions):
    """Test the /hackrx/run endpoint"""
    
    endpoint = f"{api_url}/hackrx/run"
    
    payload = {
        "documents": document_url,
        "questions": questions
    }
    
    print(f"Testing API endpoint: {endpoint}")
    print(f"Document: {document_url}")
    print(f"Questions: {questions}")
    print("=" * 50)
    
    try:
        response = requests.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=300  # 5 minute timeout
        )
        
        print(f"Response Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"Response: {json.dumps(result, indent=2)}")
            
            if "answers" in result and len(result["answers"]) == len(questions):
                print(f"\n✅ Got {len(result['answers'])} answers for {len(questions)} questions")
                for i, (question, answer) in enumerate(zip(questions, result["answers"]), 1):
                    print(f"\nQ{i}: {question}")
                    print(f"A{i}: {answer}")
                return True
            else:
                print("❌ Invalid response format or answer count mismatch")
                return False
                
        else:
            print("❌ API Error!")
            try:
                error_detail = response.json()
                print(f"Error details: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"Error text: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_health_endpoint(api_url):
    """Test the /health endpoint"""
    
    endpoint = f"{api_url}/health"
    print(f"\nTesting health endpoint: {endpoint}")
    
    try:
        response = requests.get(endpoint, timeout=10)
        print(f"Health Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Health Response: {json.dumps(result, indent=2)}")
            return True
        else:
            print(f"Health Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

if __name__ == "__main__":
    # Configuration
    API_URL = "https://your-app-name.onrender.com"  # Replace with your actual Render URL
    
    if len(sys.argv) > 1:
        API_URL = sys.argv[1]
    
    # Test document and questions
    DOCUMENT_URL = "https://drive.google.com/file/d/1gKYcPWqrC2TnfWRJvwpTB8uuQk0pypWw/view?usp=sharing"
    QUESTIONS = [
        "What is the name of the insurance policy?",
        "What are the key benefits covered?",
        "What is the maximum coverage amount?"
    ]
    
    print("Testing Hackathon API")
    print("=" * 50)
    
    # First test health endpoint
    health_ok = test_health_endpoint(API_URL)
    
    if health_ok:
        # Then test main API
        api_ok = test_hackathon_api(API_URL, DOCUMENT_URL, QUESTIONS)
        
        if api_ok:
            print("\n🎉 All tests passed! Your hackathon API is working correctly.")
        else:
            print("\n❌ API test failed.")
    else:
        print("\n❌ Health check failed - API may not be running.")
