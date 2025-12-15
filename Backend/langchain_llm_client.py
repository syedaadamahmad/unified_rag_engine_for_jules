# unified
import os
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
from unittest.mock import AsyncMock, MagicMock

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_langchain_gemini_client() -> ChatGoogleGenerativeAI:
    """
    Create LangChain-compatible Gemini 2.5 Flash client.

    Configuration:
    - Model: gemini-2.5-flash (smart + fast)
    - Temperature: 0.7 (balanced creativity)
    - Safety: All blocks disabled (educational context)
    - Max tokens: 4096 (sufficient for detailed answers)

    Returns:
        Configured ChatGoogleGenerativeAI instance

    Raises:
        ValueError: If GOOGLE_API_KEY not set in environment
    """
    if os.getenv("USE_MOCK_LLM") == "true":
        logger.info("[LANGCHAIN_GEMINI] Using mock LLM for testing")
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(content="Mocked LLM response")
        return mock_llm

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("[LANGCHAIN_GEMINI] GOOGLE_API_KEY not set in environment")

    # Disable all safety filters for educational chatbot
    # Context: This is a supervised learning environment
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    }

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        max_output_tokens=4096,
        safety_settings=safety_settings,
    )

    logger.info("[LANGCHAIN_GEMINI] ✅ Gemini 2.5 Flash client initialized")
    return llm
