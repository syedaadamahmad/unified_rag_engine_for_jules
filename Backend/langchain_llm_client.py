import os
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_langchain_gemini_client() -> ChatGoogleGenerativeAI:
    """
    Create LangChain-compatible Gemini 2.5 Flash client.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("[LANGCHAIN_GEMINI] GOOGLE_API_KEY not set in environment")

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
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

def create_langchain_gemini_lite_client() -> ChatGoogleGenerativeAI:
    """
    Create LangChain-compatible Gemini 2.5 Flash Lite client for intent detection.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("[LANGCHAIN_GEMINI_LITE] GOOGLE_API_KEY not set in environment")

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=api_key,
        temperature=0.5,
        top_p=0.9,
        top_k=30,
        max_output_tokens=1024,
        safety_settings=safety_settings,
    )

    logger.info("[LANGCHAIN_GEMINI_LITE] ✅ Gemini 2.5 Flash Lite client initialized")
    return llm
