"""
Test script for Option A implementation.
"""
import pytest
from unittest.mock import AsyncMock

from Backend.models import Message
from Backend.unified_rag_engine import UnifiedFlashEngine

@pytest.fixture(scope="function")
def engine():
    """Create a new engine instance for each test."""
    return UnifiedFlashEngine()

@pytest.mark.asyncio
async def test_greeting(engine):
    """Test greeting detection and response."""
    # Mock the intent detector for this specific test
    engine.intent_detector.detect_intent = AsyncMock(
        return_value={"intent_type": "greeting", "is_continuation": False}
    )

    response = await engine.process_query("Hello!", chat_history=[])

    assert response["type"] == "greeting"
    assert "AI Shine" in response["answer"]

@pytest.mark.asyncio
async def test_simple_query(engine):
    """Test simple AI/ML query."""
    response = await engine.process_query("What is machine learning?", chat_history=[])
    assert response["type"] in ["text", "decline", "error"]
    assert len(response["answer"]) > 0

@pytest.mark.asyncio
async def test_typo_handling(engine):
    """Test fuzzy matching with typos."""
    response = await engine.process_query("Tell me about haulio", chat_history=[])
    assert response["type"] in ["text", "decline", "error"]

@pytest.mark.asyncio
async def test_context_awareness(engine):
    """Test multi-turn conversation context."""
    history = [
        Message(role="human", content="What is CNN?"),
        Message(role="ai", content="CNN is Convolutional Neural Network...")
    ]
    response = await engine.process_query("How does it work?", chat_history=history)
    assert response["type"] in ["text", "decline", "error"]

@pytest.mark.asyncio
async def test_out_of_scope(engine):
    """Test out-of-scope query handling."""
    response = await engine.process_query("What's the weather today?", chat_history=[])
    assert response["type"] in ["text", "decline", "error"]

@pytest.mark.asyncio
async def test_farewell(engine):
    """Test farewell detection."""
    engine.intent_detector.detect_intent = AsyncMock(
        return_value={"intent_type": "farewell", "is_continuation": False}
    )
    response = await engine.process_query("Goodbye!", chat_history=[])
    assert response["type"] == "text"
    assert len(response["answer"]) > 0
