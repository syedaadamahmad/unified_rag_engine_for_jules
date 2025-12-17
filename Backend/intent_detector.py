import os
import logging
from typing import List, Dict, Any, Optional
from enum import Enum

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError

from Backend.langchain_llm_client import create_langchain_gemini_lite_client
from Backend.models import Message

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the intent classification schema
class Intent(Enum):
    """Enumeration of possible user intents."""
    GREETING = "greeting"
    FAREWELL = "farewell"
    CONTINUATION = "continuation"
    QUERY = "query"

class IntentClassification(BaseModel):
    """Structured output for intent classification."""
    intent: Intent = Field(..., description="The classified intent of the user's message.")

class LangChainIntentDetector:
    """
    Detects user intent using a LangChain classification chain.
    """

    def __init__(self):
        """Initializes the intent detection chain."""
        self.llm = create_langchain_gemini_lite_client()

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert at classifying user intent. "
                    "Classify the user's message into one of the following categories. "
                    "Consider the last few messages in the conversation history for context. "
                    "CONTINUATION intents are follow-up questions like 'tell me more', 'go on', or 'what else?'\n\n"
                    "You must respond with ONLY ONE of the following lowercase values:\n"
                    "- greeting\n"
                    "- farewell\n"
                    "- continuation\n"
                    "- query\n\n"
                    "Do not invent new labels. Do not explain. Do not return JSON outside the schema.",
                ),
                ("human", "Conversation History:\n{chat_history}\n\nUser Message: {user_message}"),
            ]
        )

        self.chain = prompt | self.llm.with_structured_output(IntentClassification)
        logger.info("[INTENT_DETECTOR] ✅ LangChain intent detector initialized.")

    def _format_history_for_prompt(self, chat_history: Optional[List[Message]]) -> str:
        """Formats the last few messages of chat history into a simple string."""
        if not chat_history:
            return "No history."

        recent_history = chat_history[-4:]
        formatted = []
        for msg in recent_history:
            role = "User" if msg.role == "human" else "Assistant"
            formatted.append(f"{role}: {msg.content}")
        return "\n".join(formatted)

    async def detect_intent(self, message: str, chat_history: Optional[List[Message]] = None) -> Dict[str, Any]:
        """
        Detects the intent of a user message using the LangChain classification chain.
        """
        if not message or not message.strip():
            return {"intent_type": "query", "is_continuation": False}

        # Short-circuit for simple greetings
        if len(message.split()) < 3 and message.lower().strip() in ["hi", "hello", "hey"]:
            return {"intent_type": "greeting", "is_continuation": False}

        formatted_history = self._format_history_for_prompt(chat_history)

        try:
            result = await self.chain.ainvoke({
                "chat_history": formatted_history,
                "user_message": message
            })
            intent_type = result.intent.value
            logger.info(f"[INTENT_DETECTOR] Detected intent: {intent_type}")
        except ValidationError as e:
            logger.warning(f"[INTENT_DETECTOR] ⚠️ Intent validation failed, defaulting to 'query'. Error: {e}")
            intent_type = "query"
        except Exception as e:
            logger.error(f"[INTENT_DETECTOR] ❌ Failed to detect intent: {e}", exc_info=True)
            intent_type = "query"

        return {
            "intent_type": intent_type,
            "is_continuation": intent_type == "continuation",
        }
