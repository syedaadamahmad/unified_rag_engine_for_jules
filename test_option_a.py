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

@pytest.mark.asyncio
async def test_greeting(mock_engine):
    """Test greeting intent."""
    response = client.post("/chat", json={
        "chat_history": [{"role": "human", "content": "Hello"}]
    })
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_simple_query(mock_engine):
    """Test simple AI/ML query."""
    response = client.post("/chat", json={
        "chat_history": [{"role": "human", "content": "What is artificial intelligence?"}]
    })
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_typo_handling(mock_engine):
    """Test typo handling."""
    response = client.post("/chat", json={
        "chat_history": [{"role": "human", "content": "What is deep lernin?"}]
    })
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_context_awareness(mock_engine):
    """Test context awareness."""
    response = client.post("/chat", json={
        "chat_history": [
            {"role": "human", "content": "What is machine learning?"},
            {"role": "ai", "content": "Machine learning is a subset of AI..."},
            {"role": "human", "content": "Tell me more about this"}
        ]
    })
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_out_of_scope(mock_engine):
    """Test out of scope query."""
    response = client.post("/chat", json={
        "chat_history": [{"role": "human", "content": "What is the best pasta recipe?"}]
    })
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_farewell(mock_engine):
    """Test farewell intent."""
    response = client.post("/chat", json={
        "chat_history": [{"role": "human", "content": "Goodbye"}]
    })
    assert response.status_code == 200
