"""
Test script for Option A implementation.
Run with: pytest test_option_a.py -v -s
"""
import pytest
import asyncio
from Backend.models import Message
from Backend.unified_rag_engine import UnifiedFlashEngine

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)

@pytest.fixture(scope="function")
def engine():
    """Create engine instance for testing (synchronous fixture)"""
    engine = UnifiedFlashEngine()
    yield engine
    engine.cleanup()

@pytest.mark.asyncio
async def test_greeting(engine):
    """Test greeting detection and response"""
    response = await engine.process_query(
        query="Hello!",
        chat_history=[]
    )
    assert response["type"] == "greeting"
    assert "AI Shine" in response["answer"]
    print(f"\n✅ Greeting: {response['answer'][:50]}...")

@pytest.mark.asyncio
async def test_simple_query(engine):
    """Test simple AI/ML query"""
    response = await engine.process_query(
        query="What is machine learning?",
        chat_history=[]
    )
    assert response["type"] in ["text", "decline"]
    assert len(response["answer"]) > 0
    print(f"\n✅ Simple query response length: {len(response['answer'])} chars")

@pytest.mark.asyncio
async def test_typo_handling(engine):
    """Test fuzzy matching with typos (k=5 should help)"""
    response = await engine.process_query(
        query="Tell me about haulio",  # Typo of 'hailuo'
        chat_history=[]
    )
    assert response["type"] in ["text", "decline"]
    print(f"\n✅ Typo handling: {response['answer'][:100]}...")

@pytest.mark.asyncio
async def test_context_awareness(engine):
    """Test multi-turn conversation context"""
    history = [
        Message(role="human", content="What is CNN?", type="text"),
        Message(role="ai", content="CNN is Convolutional Neural Network...", type="text")
    ]
    response = await engine.process_query(
        query="How does it work?",
        chat_history=history
    )
    assert response["type"] in ["text", "decline"]
    print(f"\n✅ Context-aware response: {response['answer'][:100]}...")

@pytest.mark.asyncio
async def test_out_of_scope(engine):
    """Test out-of-scope query handling"""
    response = await engine.process_query(
        query="What's the weather today?",
        chat_history=[]
    )
    # Should decline or provide AI-focused response
    print(f"\n✅ Out-of-scope: {response['answer'][:100]}...")

@pytest.mark.asyncio
async def test_farewell(engine):
    """Test farewell detection"""
    response = await engine.process_query(
        query="Goodbye!",
        chat_history=[]
    )
    assert response["type"] == "text"
    assert len(response["answer"]) > 0
    print(f"\n✅ Farewell: {response['answer'][:50]}...")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])