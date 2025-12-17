import logging
import asyncio
import json
import time
from typing import List, Dict, Any, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import SystemMessage, HumanMessage

from Backend.models import Message
from Backend.langchain_retriever import LangChainMongoRetriever
from Backend.langchain_llm_client import create_langchain_gemini_client, create_langchain_gemini_lite_client
from Backend.prompt_builder import PromptBuilder
from Backend.intent_detector import LangChainIntentDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedFlashEngine:
    """
    Unified RAG Engine using Gemini 2.5 Flash with streaming support.
    """

    def __init__(self):
        self.llm_flash = create_langchain_gemini_client()
        self.llm_lite = create_langchain_gemini_lite_client()
        self.retriever = LangChainMongoRetriever(max_results=5)
        self.prompt_builder = PromptBuilder()
        self.intent_detector = LangChainIntentDetector()
        self.executor = ThreadPoolExecutor(max_workers=4)

        logger.info("[UNIFIED_ENGINE] ✅ Gemini 2.5 Flash Engine Ready (Streaming Enabled)")

    async def _get_docs_async(self, query: str):
        """
        Run blocking retrieval operation in thread pool.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            self.retriever.get_relevant_documents,
            query
        )

    async def process_query(self, query: str, chat_history: List[Message]) -> Dict[str, Any]:
        """
        Non-streaming query processing.
        """
        start_time = time.time()

        try:
            # 1. Intent Detection
            intent = await self.intent_detector.detect_intent(query, chat_history)

            if intent["intent_type"] == "greeting":
                logger.info("[INTENT] Greeting detected")
                return {
                    "answer": self.prompt_builder.build_greeting_response(),
                    "type": "greeting"
                }

            if intent["intent_type"] == "farewell":
                logger.info("[INTENT] Farewell detected")
                return {
                    "answer": self.prompt_builder.build_farewell_response(),
                    "type": "text"
                }

            retrieval_query = query
            if intent.get("is_continuation"):
                for msg in reversed(chat_history):
                    if msg.role == "human" and not self.prompt_builder.continuation_regex.search(msg.content):
                        retrieval_query = msg.content
                        break

            # 2. Retrieval
            docs = await self._get_docs_async(retrieval_query)
            context_chunks = [doc.page_content for doc in docs]
            module_names = [doc.metadata.get("topic", "Unknown") for doc in docs]

            # 3. Build Prompts
            system_instruction = self.prompt_builder.build_system_prompt(
                intent=intent,
                has_context=bool(context_chunks)
            )

            final_user_prompt = self.prompt_builder.build_user_prompt(
                query=query,
                context_chunks=context_chunks,
                intent=intent,
                chat_history=chat_history,
                show_module_citation=True,
                module_names=module_names
            )

            # 4. Generate Answer
            messages = [
                SystemMessage(content=system_instruction),
                HumanMessage(content=final_user_prompt)
            ]

            llm = self.llm_flash if intent.get("is_continuation") else self.llm_lite
            response = await llm.ainvoke(messages)
            answer = response.content.strip()

            # 5. Determine Response Type
            response_type = "text"
            if "I specialize in AI" in answer or "⚠️" in answer:
                response_type = "decline"

            return {
                "answer": answer,
                "type": response_type
            }

        except Exception as e:
            logger.error(f"[UNIFIED_ERR] {type(e).__name__}: {str(e)}", exc_info=True)
            return {
                "answer": "⚠️ I encountered an error processing your request. Please try again.",
                "type": "error"
            }

    async def process_query_stream(self, query: str, chat_history: List[Message]) -> AsyncGenerator[str, None]:
        """
        Streaming query processing.
        """
        start_time = time.time()

        try:
            # 1. Intent Detection
            intent = await self.intent_detector.detect_intent(query, chat_history)

            if intent["intent_type"] in ["greeting", "farewell"]:
                response_builders = {
                    "greeting": self.prompt_builder.build_greeting_response,
                    "farewell": self.prompt_builder.build_farewell_response
                }
                response = {
                    "answer": response_builders[intent["intent_type"]](),
                    "type": intent["intent_type"] if intent["intent_type"] == "greeting" else "text"
                }
                yield json.dumps(response) + "\n"
                return

            retrieval_query = query
            if intent.get("is_continuation"):
                for msg in reversed(chat_history):
                    if msg.role == "human" and not self.prompt_builder.continuation_regex.search(msg.content):
                        retrieval_query = msg.content
                        break

            # 2. Retrieval
            docs = await self._get_docs_async(retrieval_query)
            context_chunks = [doc.page_content for doc in docs]
            module_names = [doc.metadata.get("topic", "Unknown") for doc in docs]

            # 3. Build Prompts
            system_instruction = self.prompt_builder.build_system_prompt(
                intent=intent,
                has_context=bool(context_chunks)
            )

            final_user_prompt = self.prompt_builder.build_user_prompt(
                query=query,
                context_chunks=context_chunks,
                intent=intent,
                chat_history=chat_history,
                show_module_citation=True,
                module_names=module_names
            )

            messages = [
                SystemMessage(content=system_instruction),
                HumanMessage(content=final_user_prompt)
            ]

            yield json.dumps({"type": "text", "status": "generating"}) + "\n"

            llm = self.llm_flash if intent.get("is_continuation") else self.llm_lite

            async for chunk in llm.astream(messages):
                if chunk.content:
                    payload = {"answer_chunk": chunk.content}
                    yield json.dumps(payload) + "\n"
                    await asyncio.sleep(0)

        except Exception as e:
            logger.error(f"[STREAM_ERR] {type(e).__name__}: {str(e)}", exc_info=True)
            yield json.dumps({
                "answer": "⚠️ I encountered an error processing your request. Please try again.",
                "type": "error"
            }) + "\n"

    def cleanup(self):
        """Graceful shutdown."""
        if self.executor:
            self.executor.shutdown(wait=True)
