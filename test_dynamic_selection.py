import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from Backend.unified_rag_engine import UnifiedFlashEngine

@pytest.mark.asyncio
async def test_dynamic_llm_selection_for_query():
    """
    Verify that the 'lite' LLM is used for simple queries.
    """
    engine = UnifiedFlashEngine()

    # Mock the intent detector to return a 'query' intent
    mock_intent_detector = AsyncMock()
    mock_intent_detector.detect_intent.return_value = {
        "intent_type": "query",
        "is_continuation": False
    }
    engine.intent_detector = mock_intent_detector

    # Mock the LLM clients to track which one is called
    engine.llm_lite = AsyncMock()
    engine.llm_flash = AsyncMock()

    # Set a valid return value for the ainvoke method
    engine.llm_lite.ainvoke.return_value = MagicMock(content="Mocked Lite LLM response")


    await engine.process_query("What is machine learning?", chat_history=[])

    # Assert that the 'lite' model was called and the 'flash' was not
    engine.llm_lite.ainvoke.assert_called_once()
    engine.llm_flash.ainvoke.assert_not_called()

@pytest.mark.asyncio
async def test_dynamic_llm_selection_for_continuation():
    """
    Verify that the 'flash' LLM is used for continuation queries.
    """
    engine = UnifiedFlashEngine()

    # Mock the intent detector to return a 'continuation' intent
    mock_intent_detector = AsyncMock()
    mock_intent_detector.detect_intent.return_value = {
        "intent_type": "continuation",
        "is_continuation": True
    }
    engine.intent_detector = mock_intent_detector

    # Mock the LLM clients to track which one is called
    engine.llm_lite = AsyncMock()
    engine.llm_flash = AsyncMock()

    # Set a valid return value for the ainvoke method
    engine.llm_flash.ainvoke.return_value = MagicMock(content="Mocked Flash LLM response")

    # The engine expects Pydantic Message objects, not dicts
    from Backend.models import Message
    chat_history = [
        Message(role="human", content="What is machine learning?"),
        Message(role="ai", content="It is a branch of AI."),
    ]
    await engine.process_query("Tell me more", chat_history=chat_history)

    # Assert that the 'flash' model was called and the 'lite' was not
    engine.llm_flash.ainvoke.assert_called_once()
    engine.llm_lite.ainvoke.assert_not_called()
