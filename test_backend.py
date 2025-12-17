import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

@pytest.fixture
def mock_engine():
    """Fixture to mock the UnifiedFlashEngine."""
    with patch("main.unified_engine", new_callable=AsyncMock) as mock_engine_instance:
        mock_engine_instance.process_query.return_value = {
            "answer": "Mocked response",
            "type": "text"
        }
        yield

def test_health_check():
    """Test health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_greeting(mock_engine):
    """Test greeting intent."""
    response = client.post("/chat", json={
        "chat_history": [{"role": "human", "content": "Hello"}]
    })
    assert response.status_code == 200

def test_simple_query(mock_engine):
    """Test simple AI/ML query."""
    response = client.post("/chat", json={
        "chat_history": [{"role": "human", "content": "What is artificial intelligence?"}]
    })
    assert response.status_code == 200

def test_continuation(mock_engine):
    """Test continuation cue detection."""
    response = client.post("/chat", json={
        "chat_history": [
            {"role": "human", "content": "What is machine learning?"},
            {"role": "ai", "content": "Machine learning is a subset of AI..."},
            {"role": "human", "content": "Tell me more about this"}
        ]
    })
    assert response.status_code == 200

def test_out_of_scope(mock_engine):
    """Test domain decline (non-AI/ML query)."""
    response = client.post("/chat", json={
        "chat_history": [{"role": "human", "content": "What is the best pasta recipe?"}]
    })
    assert response.status_code == 200

def test_craft_framework(mock_engine):
    """Test specific KB content (CRAFT framework)."""
    response = client.post("/chat", json={
        "chat_history": [{"role": "human", "content": "What is the CRAFT prompting framework?"}]
    })
    assert response.status_code == 200

def test_multi_turn_conversation(mock_engine):
    """Test multi-turn conversation flow."""
    conversation = [
        {"role": "ai", "content": "👋 Hello! I'm AI Shine."},
    ]
    queries = [
        "What is AI creativity?",
        "How can students use it?",
        "Tell me more",
        "What tools can help?"
    ]
    for query in queries:
        conversation.append({"role": "human", "content": query})
        response = client.post("/chat", json={"chat_history": conversation})
        assert response.status_code == 200
        conversation.append({"role": "ai", "content": response.json()["answer"]})
