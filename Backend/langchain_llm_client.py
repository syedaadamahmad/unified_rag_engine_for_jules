# unified 
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























# """ has everything except for live token streaming
# LangChain Gemini Client
# Wraps Google Gemini 2.5 Flash and Flash-Lite with LangChain's ChatGoogleGenerativeAI interface.
# """
# import os
# import logging
# from langchain_google_genai import ChatGoogleGenerativeAI
# from google.generativeai.types import HarmCategory, HarmBlockThreshold
# from dotenv import load_dotenv

# load_dotenv()

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# def create_langchain_gemini_client() -> ChatGoogleGenerativeAI:
#     """
#     Create LangChain-compatible Gemini 2.5 Flash client.
    
#     Returns:
#         ChatGoogleGenerativeAI instance configured for educational content
#     """
#     api_key = os.getenv("GOOGLE_API_KEY")
#     if not api_key:
#         raise ValueError("[LANGCHAIN_GEMINI] GOOGLE_API_KEY not set")
    
#     # Configure safety settings (permissive for educational content)
#     safety_settings = {
#         HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
#         HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
#         HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
#         HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
#     }
    
#     # Create LangChain Gemini client
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.5-flash",
#         google_api_key=api_key,
#         temperature=0.7,
#         top_p=0.95,
#         top_k=40,
#         max_output_tokens=4096,
#         safety_settings=safety_settings,
#         convert_system_message_to_human=True
#     )
    
#     logger.info("[LANGCHAIN_GEMINI] ✅ Initialized Gemini 2.5 Flash")
#     logger.info("[LANGCHAIN_GEMINI] Config: temp=0.7, max_tokens=4096")
    
#     return llm


# def create_langchain_gemini_lite_client() -> ChatGoogleGenerativeAI:
#     """
#     Create LangChain-compatible Gemini 2.5 Flash-Lite client.
#     Fast and cost-effective for simple queries.
    
#     Returns:
#         ChatGoogleGenerativeAI instance optimized for brief responses
#     """
#     api_key = os.getenv("GOOGLE_API_KEY")
#     if not api_key:
#         raise ValueError("[LANGCHAIN_GEMINI_LITE] GOOGLE_API_KEY not set")
    
#     # Configure safety settings
#     safety_settings = {
#         HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
#         HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
#         HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
#         HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
#     }
    
#     # Create Flash-Lite client (faster, cheaper, 15 RPM limit)
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.5-flash-lite",
#         google_api_key=api_key,
#         temperature=0.5,  # Lower for more focused responses
#         top_p=0.9,
#         top_k=30,
#         max_output_tokens=1024,  # Shorter responses
#         safety_settings=safety_settings,
#         convert_system_message_to_human=True
#     )
    
#     logger.info("[LANGCHAIN_GEMINI_LITE] ✅ Initialized Gemini 2.5 Flash-Lite")
#     logger.info("[LANGCHAIN_GEMINI_LITE] Config: temp=0.5, max_tokens=1024")
    
#     return llm























# """ has everything except for live token streaming
# LangChain Gemini Client
# Wraps Google Gemini 2.5 Flash with LangChain's ChatGoogleGenerativeAI interface.
# """
# import os
# import logging
# from langchain_google_genai import ChatGoogleGenerativeAI
# from google.generativeai.types import HarmCategory, HarmBlockThreshold
# from dotenv import load_dotenv

# load_dotenv()

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# def create_langchain_gemini_client() -> ChatGoogleGenerativeAI:
#     """
#     Create LangChain-compatible Gemini 2.5 Flash client.
    
#     Returns:
#         ChatGoogleGenerativeAI instance configured for educational content
#     """
#     api_key = os.getenv("GOOGLE_API_KEY")
#     if not api_key:
#         raise ValueError("[LANGCHAIN_GEMINI] GOOGLE_API_KEY not set")
    
#     # Configure safety settings (permissive for educational content)
#     safety_settings = {
#         HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
#         HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
#         HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
#         HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
#     }
    
#     # Create LangChain Gemini client
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.5-flash",
#         google_api_key=api_key,
#         temperature=0.9,  # Balanced creativity and grounding
#         top_p=0.95,
#         top_k=40,
#         max_output_tokens=4096,  # Increased to reduce truncation
#         safety_settings=safety_settings,
#         convert_system_message_to_human=True  # Gemini requires system as first user message
#     )
    
#     logger.info("[LANGCHAIN_GEMINI] ✅ Initialized Gemini 2.5 Flash via LangChain")
#     logger.info("[LANGCHAIN_GEMINI] Config: temp=0.9, max_tokens=4096")
    
#     return llm