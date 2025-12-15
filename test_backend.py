"""
Backend Test Script
Tests RAG engine pipeline and /chat endpoint.
"""
import requests
import json
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test health endpoint."""
    logger.info("\n" + "="*60)
    logger.info("TEST: Health Check")
    logger.info("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    logger.info(f"Status: {response.status_code}")
    logger.info(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    logger.info("✅ Health check passed\n")


def test_greeting():
    """Test greeting detection."""
    logger.info("\n" + "="*60)
    logger.info("TEST: Greeting Detection")
    logger.info("="*60)
    
    payload = {
        "chat_history": [
            {"role": "human", "content": "Hello"}
        ]
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    logger.info(f"Status: {response.status_code}")
    data = response.json()
    logger.info(f"Answer: {data['answer']}")
    logger.info(f"Type: {data['type']}")
    assert data['type'] == 'greeting'
    logger.info("✅ Greeting test passed\n")


def test_simple_query():
    """Test simple AI/ML query."""
    logger.info("\n" + "="*60)
    logger.info("TEST: Simple AI Query")
    logger.info("="*60)
    
    payload = {
        "chat_history": [
            {"role": "ai", "content": "👋 Hello! I'm AI Shine."},
            {"role": "human", "content": "What is artificial intelligence?"}
        ]
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    logger.info(f"Status: {response.status_code}")
    data = response.json()
    logger.info(f"Answer: {data['answer'][:200]}...")
    logger.info(f"Type: {data['type']}")
    assert response.status_code == 200
    assert data['type'] in ['structured', 'text']
    logger.info("✅ Simple query test passed\n")


def test_continuation():
    """Test continuation cue detection."""
    logger.info("\n" + "="*60)
    logger.info("TEST: Continuation Cue")
    logger.info("="*60)
    
    payload = {
        "chat_history": [
            {"role": "ai", "content": "👋 Hello! I'm AI Shine."},
            {"role": "human", "content": "What is machine learning?"},
            {"role": "ai", "content": "Machine learning is a subset of AI..."},
            {"role": "human", "content": "Tell me more about this"}
        ]
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    logger.info(f"Status: {response.status_code}")
    data = response.json()
    logger.info(f"Answer length: {len(data['answer'])} chars")
    logger.info(f"Type: {data['type']}")
    assert response.status_code == 200
    logger.info("✅ Continuation test passed\n")


def test_out_of_scope():
    """Test domain decline (non-AI/ML query)."""
    logger.info("\n" + "="*60)
    logger.info("TEST: Out-of-Scope Query")
    logger.info("="*60)
    
    payload = {
        "chat_history": [
            {"role": "ai", "content": "👋 Hello! I'm AI Shine."},
            {"role": "human", "content": "What is the best pasta recipe?"}
        ]
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    logger.info(f"Status: {response.status_code}")
    data = response.json()
    logger.info(f"Answer: {data['answer'][:200]}...")
    logger.info(f"Type: {data['type']}")
    assert response.status_code == 200
    logger.info("✅ Out-of-scope test passed\n")


def test_craft_framework():
    """Test specific KB content (CRAFT framework)."""
    logger.info("\n" + "="*60)
    logger.info("TEST: CRAFT Framework Query")
    logger.info("="*60)
    
    payload = {
        "chat_history": [
            {"role": "ai", "content": "👋 Hello! I'm AI Shine."},
            {"role": "human", "content": "What is the CRAFT prompting framework?"}
        ]
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    logger.info(f"Status: {response.status_code}")
    data = response.json()
    logger.info(f"Answer: {data['answer'][:300]}...")
    logger.info(f"Type: {data['type']}")
    assert response.status_code == 200
    logger.info("✅ CRAFT framework test passed\n")


def test_multi_turn_conversation():
    """Test multi-turn conversation flow."""
    logger.info("\n" + "="*60)
    logger.info("TEST: Multi-Turn Conversation")
    logger.info("="*60)
    
    conversation = [
        {"role": "ai", "content": "👋 Hello! I'm AI Shine."},
    ]
    
    queries = [
        "What is AI creativity?",
        "How can students use it?",
        "Tell me more",
        "What tools can help?"
    ]
    
    for i, query in enumerate(queries, 1):
        logger.info(f"\n--- Turn {i} ---")
        conversation.append({"role": "human", "content": query})
        
        payload = {"chat_history": conversation}
        response = requests.post(f"{BASE_URL}/chat", json=payload)
        data = response.json()
        
        logger.info(f"User: {query}")
        logger.info(f"AI ({data['type']}): {data['answer'][:150]}...")
        
        conversation.append({"role": "ai", "content": data['answer']})
        
        assert response.status_code == 200
    
    logger.info("\n✅ Multi-turn conversation test passed\n")


def run_all_tests():
    """Run all test cases."""
    logger.info("\n" + "="*60)
    logger.info("AI SHINE TUTOR - BACKEND TEST SUITE")
    logger.info("="*60)
    
    try:
        test_health_check()
        test_greeting()
        test_simple_query()
        test_continuation()
        test_out_of_scope()
        test_craft_framework()
        test_multi_turn_conversation()
        
        logger.info("\n" + "="*60)
        logger.info("✅ ALL TESTS PASSED")
        logger.info("="*60 + "\n")
    
    except AssertionError as e:
        logger.error(f"\n❌ TEST FAILED: {e}\n")
    except requests.exceptions.ConnectionError:
        logger.error("\n❌ Cannot connect to backend. Is the server running on port 8000?\n")
    except Exception as e:
        logger.error(f"\n❌ UNEXPECTED ERROR: {e}\n")


if __name__ == "__main__":
    run_all_tests()