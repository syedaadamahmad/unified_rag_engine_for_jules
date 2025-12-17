import os
import logging
from typing import List, Dict, Any, Optional
from enum import Enum

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

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
    This provides a more robust and scalable alternative to regex-based detection.
    """

    def __init__(self):
        """Initializes the intent detection chain."""
        # Use the fast "lite" client for intent detection to minimize latency
        self.llm = create_langchain_gemini_lite_client()

        # The prompt template instructs the LLM on how to classify the user's message
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert at classifying user intent. "
                    "Classify the user's message into one of the following categories: "
                    "GREETING, FAREWELL, CONTINUATION, or QUERY. "
                    "Consider the last few messages in the conversation history for context. "
                    "CONTINUATION intents are follow-up questions like 'tell me more', 'go on', or 'what else?'",
                ),
                ("human", "Conversation History:\n{chat_history}\n\nUser Message: {user_message}"),
            ]
        )

        # Create a structured output chain that forces the LLM to return data in the `IntentClassification` schema
        self.chain = prompt | self.llm.with_structured_output(IntentClassification)
        logger.info("[INTENT_DETECTOR] ✅ LangChain intent detector initialized.")

    def _format_history_for_prompt(self, chat_history: Optional[List[Message]]) -> str:
        """Formats the last few messages of chat history into a simple string."""
        if not chat_history:
            return "No history."

        # Get the last 4 messages to keep the prompt concise
        recent_history = chat_history[-4:]
        formatted = []
        for msg in recent_history:
            role = "User" if msg.role == "human" else "Assistant"
            formatted.append(f"{role}: {msg.content}")
        return "\n".join(formatted)

    async def detect_intent(self, message: str, chat_history: Optional[List[Message]] = None) -> Dict[str, Any]:
        """
        Detects the intent of a user message using the LangChain classification chain.

        Returns a dictionary compatible with the `UnifiedFlashEngine`.
        """
        if not message or not message.strip():
            return {"intent_type": "query", "is_continuation": False}

        # Format history for the prompt
        formatted_history = self._format_history_for_prompt(chat_history)

        # Invoke the chain to get the structured classification
        try:
            result = await self.chain.ainvoke({
                "chat_history": formatted_history,
                "user_message": message
            })
            intent_type = result.intent.value
            logger.info(f"[INTENT_DETECTOR] Detected intent: {intent_type}")
        except Exception as e:
            logger.error(f"[INTENT_DETECTOR] ❌ Failed to detect intent: {e}", exc_info=True)
            # Fallback to "query" on failure
            intent_type = "query"

        # Map the classification result to the dictionary format expected by the engine
        return {
            "intent_type": intent_type,
            "is_continuation": intent_type == "continuation",
        }
