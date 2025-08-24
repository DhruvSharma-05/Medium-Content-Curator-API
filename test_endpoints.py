#!/usr/bin/env python3
"""
Simple test script to verify the Medium AI Article Suggester API Server
Run this script while your Flask API is running on localhost:5000
"""

import requests
import json
import time
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:5000"

def print_test_header(test_name):
    """Print a formatted test header"""
    print(f"\n{'='*60}")
    print(f"🧪 Testing: {test_name}")
    print(f"{'='*60}")

def print_success(message):
    """Print success message"""
    print(f"✅ {message}")

def print_error(message):
    """Print error message"""
    print(f"❌ {message}")

def print_info(message):
    """Print info message"""
    print(f"ℹ️  {message}")

def test_health_endpoint():
    """Test the health check endpoint"""
    print_test_header("Health Check Endpoint")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Health check passed! Status: {data.get('status')}")
            print_info(f"Service: {data.get('service')}")
            print_info(f"Features: {len(data.get('features', []))} features available")
            return True
        else:
            print_error(f"Health check failed with status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_error("Could not connect to the API. Make sure it's running on localhost:5000")
        return False
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        return False

def test_topics_endpoint():
    """Test the topics endpoint"""
    print_test_header("Topics Endpoint")
    
    try:
        response = requests.get(f"{BASE_URL}/api/topics")
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Topics endpoint working! Found {data.get('total_count')} topics")
            print_info(f"Categories: {list(data.get('categories', {}).keys())}")
            return True
        else:
            print_error(f"Topics endpoint failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Error testing topics: {str(e)}")
        return False

def test_search_endpoint():
    """Test the search endpoint"""
    print_test_header("Search Endpoint")
    
    try:
        search_data = {
            "query": "artificial intelligence",
            "limit": 3
        }
        
        response = requests.post(
            f"{BASE_URL}/api/search",
            json=search_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Search endpoint working! Found {data.get('total_count')} articles")
            if data.get('search_results'):
                first_article = data['search_results'][0]
                print_info(f"First article: {first_article.get('title', 'No title')[:50]}...")
            return True
        else:
            print_error(f"Search endpoint failed with status {response.status_code}")
            if response.text:
                print_info(f"Response: {response.text[:200]}...")
            return False
            
    except Exception as e:
        print_error(f"Error testing search: {str(e)}")
        return False

def test_trending_endpoint():
    """Test the trending endpoint"""
    print_test_header("Trending Endpoint")
    
    try:
        response = requests.get(f"{BASE_URL}/api/trending?limit=3")
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Trending endpoint working! Found {data.get('total_count')} articles")
            if data.get('trending_articles'):
                first_article = data['trending_articles'][0]
                print_info(f"First trending: {first_article.get('title', 'No title')[:50]}...")
            return True
        else:
            print_error(f"Trending endpoint failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Error testing trending: {str(e)}")
        return False

def test_suggestions_endpoint():
    """Test the suggestions endpoint"""
    print_test_header("Suggestions Endpoint")
    
    try:
        suggestions_data = {
            "user_preferences": {
                "topics": ["AI", "Machine Learning"],
                "reading_time": "medium",
                "recency": "week",
                "difficulty": "intermediate"
            },
            "search_query": "deep learning",
            "limit": 3
        }
        
        response = requests.post(
            f"{BASE_URL}/api/suggestions",
            json=suggestions_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Suggestions endpoint working! Found {data.get('total_count')} suggestions")
            if data.get('suggestions'):
                first_suggestion = data['suggestions'][0]
                print_info(f"First suggestion: {first_suggestion.get('title', 'No title')[:50]}...")
            return True
        else:
            print_error(f"Suggestions endpoint failed with status {response.status_code}")
            if response.text:
                print_info(f"Response: {response.text[:200]}...")
            return False
            
    except Exception as e:
        print_error(f"Error testing suggestions: {str(e)}")
        return False

def test_user_preferences():
    """Test the user preferences endpoints"""
    print_test_header("User Preferences Endpoints")
    
    try:
        # Test saving preferences
        preferences_data = {
            "user_id": "test_user_123",
            "preferences": {
                "topics": ["Python", "Web Development"],
                "reading_time": "short",
                "recency": "month",
                "difficulty": "beginner"
            }
        }
        
        response = requests.post(
            f"{BASE_URL}/api/user/preferences",
            json=preferences_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print_success("Save preferences endpoint working!")
            
            # Test getting preferences
            get_response = requests.get(f"{BASE_URL}/api/user/preferences/test_user_123")
            
            if get_response.status_code == 200:
                print_success("Get preferences endpoint working!")
                return True
            else:
                print_error(f"Get preferences failed with status {get_response.status_code}")
                return False
        else:
            print_error(f"Save preferences failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Error testing user preferences: {str(e)}")
        return False

def test_stats_endpoint():
    """Test the stats endpoint"""
    print_test_header("Stats Endpoint")
    
    try:
        response = requests.get(f"{BASE_URL}/api/stats")
        
        if response.status_code == 200:
            data = response.json()
            print_success("Stats endpoint working!")
            stats = data.get('stats', {})
            print_info(f"Cached articles: {stats.get('cached_articles', 0)}")
            print_info(f"Active users: {stats.get('active_users', 0)}")
            return True
        else:
            print_error(f"Stats endpoint failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Error testing stats: {str(e)}")
        return False

def test_invalid_endpoint():
    """Test that invalid endpoints return proper 404 errors"""
    print_test_header("Invalid Endpoint Handling")
    
    try:
        response = requests.get(f"{BASE_URL}/invalid/endpoint")
        
        if response.status_code == 404:
            data = response.json()
            if data.get('error') == 'Endpoint not found' and 'available_endpoints' in data:
                print_success("404 error handling working correctly!")
                print_info(f"Available endpoints: {len(data['available_endpoints'])} endpoints listed")
                return True
            else:
                print_error("404 response format incorrect")
                return False
        else:
            print_error(f"Invalid endpoint should return 404, got {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Error testing invalid endpoint: {str(e)}")
        return False

def run_all_tests():
    """Run all API tests"""
    print("🚀 Starting Medium AI Article Suggester API Tests")
    print(f"📡 Testing API at: {BASE_URL}")
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Health Check", test_health_endpoint),
        ("Topics", test_topics_endpoint),
        ("Search", test_search_endpoint),
        ("Trending", test_trending_endpoint),
        ("Suggestions", test_suggestions_endpoint),
        ("User Preferences", test_user_preferences),
        ("Stats", test_stats_endpoint),
        ("Error Handling", test_invalid_endpoint)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
            time.sleep(0.5)  # Small delay between tests
        except Exception as e:
            print_error(f"Test {test_name} crashed: {str(e)}")
            results.append((test_name, False))
    
    # Print summary
    print_test_header("Test Results Summary")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"📊 Results: {passed}/{total} tests passed")
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} - {test_name}")
    
    if passed == total:
        print_success("🎉 All tests passed! Your API is working perfectly!")
    else:
        print_error(f"⚠️  {total - passed} tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = run_all_tests()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Testing interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {str(e)}")
        exit(1)
