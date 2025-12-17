"""
Pytest Fixtures for Backend Tests
"""
import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture(scope="session", autouse=True)
def mock_llm_clients():
    """Mock Gemini clients to avoid actual API calls."""
    with patch('Backend.langchain_llm_client.create_langchain_gemini_client') as mock_flash, \
         patch('Backend.langchain_llm_client.create_langchain_gemini_lite_client') as mock_lite:

        # Mock the powerful model
        mock_flash_instance = MagicMock()
        mock_flash_instance.ainvoke.return_value.content = "This is a detailed answer."
        mock_flash.return_value = mock_flash_instance

        # Mock the fast model
        mock_lite_instance = MagicMock()
        mock_lite_instance.ainvoke.return_value.content = '{"intent": "question"}'
        mock_lite.return_value = mock_lite_instance

        yield mock_flash, mock_lite

@pytest.fixture
def mock_retriever():
    """Mock the retriever to return dummy documents."""
    with patch('Backend.langchain_retriever.LangChainMongoRetriever') as mock:
        instance = mock.return_value
        instance.get_relevant_documents.return_value = [MagicMock(page_content="Dummy document")]
        yield instance
