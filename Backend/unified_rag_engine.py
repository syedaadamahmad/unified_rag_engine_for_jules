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

    Key Features:
    - Single LLM call per query (no rephrasing step)
    - Non-blocking async retrieval via ThreadPoolExecutor
    - Token-by-token streaming for instant perceived response
    - Higher K retrieval (k=5) for typo/fuzzy match handling
    - Maintains ChatResponse schema for non-streaming endpoint
    - NDJSON streaming for /chat_stream endpoint
    - Uses PromptBuilder for strict formatting and knowledge locking
    - Dynamic LLM selection for speed and quality
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
        Prevents MongoDB + AWS Bedrock from blocking event loop.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            self.retriever.get_relevant_documents,
            query
        )

    async def process_query(self, query: str, chat_history: List[Message]) -> Dict[str, Any]:
        """
        Non-streaming query processing (for backward compatibility).
        Used by /chat endpoint.

        Returns:
            Dict with 'answer' (str) and 'type' (str) keys
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

            # If continuing, find the last substantive user query for retrieval
            retrieval_query = query
            if intent.get("is_continuation"):
                for msg in reversed(chat_history):
                    if msg.role == "human" and not self.prompt_builder.continuation_regex.search(msg.content):
                        retrieval_query = msg.content
                        logger.info(f"[CONTINUATION] Using last substantive query for retrieval: '{retrieval_query[:50]}...'")
                        break

            # 2. Retrieval (Non-blocking)
            logger.info(f"[RETRIEVAL] Fetching docs for: '{retrieval_query[:50]}...'")
            docs = await self._get_docs_async(retrieval_query)

            context_chunks = [doc.page_content for doc in docs]
            module_names = [doc.metadata.get("topic", "Unknown") for doc in docs]

            logger.info(f"[RETRIEVAL] Found {len(context_chunks)} documents")

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

            # Dynamic LLM Selection
            llm = self.llm_flash if intent.get("is_continuation") else self.llm_lite
            logger.info(f"[LLM_SELECTION] Using {'Flash' if llm == self.llm_flash else 'Lite'} model.")

            response = await llm.ainvoke(messages)

            answer = response.content.strip()

            total_time = time.time() - start_time
            logger.info(f"[PERF] Total query time: {total_time:.2f}s")

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
        Streaming query processing with NDJSON output.
        Used by /chat_stream endpoint.

        Yields:
            JSON strings (one per line) with:
            - {"type": "text", "status": "generating"} - Initial signal
            - {"answer_chunk": "text"} - Token fragments
            - {"answer": "full text", "type": "greeting"} - Complete responses (greetings/farewells)
            - {"answer": "error", "type": "error"} - Error responses
        """
        start_time = time.time()

        try:
            # 1. Intent Detection (Fast)
            intent = await self.intent_detector.detect_intent(query, chat_history)

            if intent["intent_type"] == "greeting":
                # Send full greeting immediately
                response = {
                    "answer": self.prompt_builder.build_greeting_response(),
                    "type": "greeting"
                }
                yield json.dumps(response) + "\n"
                logger.info("[STREAM] Greeting sent")
                return

            if intent["intent_type"] == "farewell":
                response = {
                    "answer": self.prompt_builder.build_farewell_response(),
                    "type": "text"
                }
                yield json.dumps(response) + "\n"
                logger.info("[STREAM] Farewell sent")
                return

            # If continuing, find the last substantive user query for retrieval
            retrieval_query = query
            if intent.get("is_continuation"):
                for msg in reversed(chat_history):
                    if msg.role == "human" and not self.prompt_builder.continuation_regex.search(msg.content):
                        retrieval_query = msg.content
                        logger.info(f"[CONTINUATION] Using last substantive query for retrieval: '{retrieval_query[:50]}...'")
                        break

            # 2. Retrieval (Non-blocking)
            logger.info(f"[STREAM] Fetching docs for: '{retrieval_query[:50]}...'")
            docs = await self._get_docs_async(retrieval_query)

            context_chunks = [doc.page_content for doc in docs]
            module_names = [doc.metadata.get("topic", "Unknown") for doc in docs]

            retrieval_time = time.time() - start_time
            logger.info(f"[STREAM] Retrieval took {retrieval_time:.2f}s, found {len(context_chunks)} docs")

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

            # 4. Signal streaming start (kills frontend spinner immediately)
            yield json.dumps({"type": "text", "status": "generating"}) + "\n"

            # Dynamic LLM Selection
            llm = self.llm_flash if intent.get("is_continuation") else self.llm_lite
            logger.info(f"[LLM_SELECTION] Using {'Flash' if llm == self.llm_flash else 'Lite'} model for streaming.")

            logger.info(f"[STREAM] Starting LLM stream at {time.time() - start_time:.2f}s")

            # 5. Stream tokens from LLM
            async for chunk in llm.astream(messages):
                if chunk.content:
                    # Send each token/fragment immediately
                    payload = {"answer_chunk": chunk.content}
                    yield json.dumps(payload) + "\n"

                    # Force async context switch to ensure immediate delivery
                    await asyncio.sleep(0)

            total_time = time.time() - start_time
            logger.info(f"[STREAM] ✅ Complete stream in {total_time:.2f}s")

        except Exception as e:
            logger.error(f"[STREAM_ERR] {type(e).__name__}: {str(e)}", exc_info=True)
            yield json.dumps({
                "answer": "⚠️ I encountered an error processing your request. Please try again.",
                "type": "error"
            }) + "\n"

    def cleanup(self):
        """Graceful shutdown - close thread pool"""
        logger.info("[CLEANUP] Shutting down ThreadPoolExecutor...")
        self.executor.shutdown(wait=True)
        logger.info("[CLEANUP] ✅ Cleanup complete")
