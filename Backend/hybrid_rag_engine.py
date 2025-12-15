"""
Hybrid RAG Engine
Intelligent routing between Flash-Lite (fast/brief) and Flash (detailed) based on query complexity.
Optimized to minimize API calls.
Synced with PromptBuilder formatting rules.
"""
import logging
import re
from typing import List, Dict, Any
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from Backend.models import Message
from Backend.langchain_retriever import LangChainMongoRetriever
from Backend.langchain_llm_client import create_langchain_gemini_client, create_langchain_gemini_lite_client
from Backend.prompt_builder import PromptBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridRAGEngine:
    """
    Hybrid RAG engine with intelligent routing:
    - Flash-Lite: Fast, brief responses for initial queries
    - Flash: Detailed, comprehensive responses for deep-dive requests
    
    Optimized for minimal API calls:
    - ConversationBufferMemory (no summarization calls)
    - Lite chain skips condense step (1 API call only)
    - Full chain uses condense when needed (2 API calls)
    
    Synced with PromptBuilder formatting rules for consistency.
    """
    
    # Patterns for detecting deep-dive requests
    DETAILED_PATTERNS = [
        r'\bdetails?\b',  # Matches "detail" or "details"
        r'\btell\s+me\s+more\b',
        r'\bexpand\b',
        r'\bcontinue\b',
        r'\bgo\s+deeper\b',
        r'\bin\s+detail\b',
        r'\belaborate\b',
        r'\bexplain\s+further\b',
        r'\bmore\s+information\b',
        r'\bcomprehensive\b',
    ]
    
    # Patterns for brief responses
    BRIEF_PATTERNS = [
        r'\bbriefly\b',
        r'\bquick\b',
        r'\bshort\b',
        r'\bsummarize\b',
        r'\bin\s+short\b',
    ]
    
    def __init__(self):
        """Initialize hybrid RAG components."""
        try:
            logger.info("[HYBRID_RAG_ENGINE] 🔧 Initializing with API call optimization...")
            
            # Initialize retrievers (same underlying MongoDB, different top_k)
            self.lite_retriever = LangChainMongoRetriever(max_results=1)  # Top 1 for lite
            self.full_retriever = LangChainMongoRetriever(max_results=3)  # Top 3 for full
            
            # Initialize LLMs
            self.flash_llm = create_langchain_gemini_client()  # Full power
            self.lite_llm = create_langchain_gemini_lite_client()  # Fast & cheap
            
            # ConversationBufferMemory - NO SUMMARIZATION API CALLS
            # Stores full history in memory (uses more tokens but zero API calls)
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
                input_key="question"
            )
            
            logger.info("[HYBRID_RAG_ENGINE] ✅ Using ConversationBufferMemory (0 API calls for memory)")
            
            # Prompt builder for greeting/farewell
            self.prompt_builder = PromptBuilder()
            
            # Compile regex patterns
            self.detailed_regex = re.compile('|'.join(self.DETAILED_PATTERNS), re.IGNORECASE)
            self.brief_regex = re.compile('|'.join(self.BRIEF_PATTERNS), re.IGNORECASE)
            
            # Build chains
            self.lite_chain = self._build_lite_chain()
            self.full_chain = self._build_full_chain()
            
            logger.info("[HYBRID_RAG_ENGINE] ✅ All components initialized")
            logger.info("[HYBRID_RAG_ENGINE] 📊 API Call Breakdown:")
            logger.info("[HYBRID_RAG_ENGINE]    Lite query: 1 API call (answer only)")
            logger.info("[HYBRID_RAG_ENGINE]    Full query: 2 API calls (condense + answer)")
            logger.info("[HYBRID_RAG_ENGINE]    Memory: 0 API calls (buffer storage)")
            
        except Exception as e:
            logger.error(f"[HYBRID_RAG_ENGINE_ERR] Initialization failed: {e}")
            raise
    
    def _build_lite_chain(self) -> ConversationalRetrievalChain:
        """Build lite chain for brief, fast responses. OPTIMIZED: No condense step."""
        
        # Lite QA prompt - SYNCED WITH PROMPTBUILDER
        lite_qa_template = """You are AI Shine, an expert educational assistant specializing in Artificial Intelligence, Machine Learning, and related technologies.

Context from knowledge base:
{context}

Chat History:
{chat_history}

Question: {question}

SCOPE (strict):
- Only answer topics in: Artificial Intelligence, Machine Learning, Deep Learning, Data Science, Natural Language Processing, Computer Vision, AI Applications and Ethics, AI-powered Education, Programming for AI/ML (e.g., Python, TensorFlow, PyTorch).
- If out-of-scope, respond exactly (plain text, no HTML):
"⚠️ I specialize in AI and Machine Learning topics. I'd be happy to help with questions about [suggest 2-3 related AI/ML topics]."
- Start that out-of-scope line with the ⚠️ emoji.

KNOWLEDGE-LOCK (critical):
- Use ONLY the provided educational content for this turn. Do NOT add roles, items, examples, claims, steps, or statistics that are not present in the content.
- HALLUCINATION GUARDRAIL: If the KB content mentions a concept (e.g., "machine learning") but doesn't provide a complete definition, structure your answer as follows:
  [1] Lead with a clear, concise definition using your AI/ML training knowledge
  [2] Then naturally integrate KB examples and applications (e.g., "It's used in Netflix recommendations for...")
  [3] Do NOT start with phrases like "The provided content doesn't define..." or "From a general AI perspective..." - just provide the definition directly and weave in KB examples
- If the KB has NO information on a domain-specific AI/ML topic, you may provide a brief, accurate definition from your training knowledge, but acknowledge that specific applications aren't in the provided materials.
- NEVER invent examples, statistics, specific claims, roles, or applications that are not present in the KB content. Only provide general definitions when absolutely necessary.
- If details are missing, structure as: "[Clear definition]. Based on the content, [integrate KB examples]."
- Do NOT switch to prompting frameworks unless explicitly asked.

FUZZY MATCHING (critical):
- If the query uses similar-sounding names with minor spelling variations (e.g., "haulio" vs "hailuo", "photomat" vs "photomath"), treat them as the same entity.
- Focus on phonetic similarity and semantic context, not exact character matching.
- If KB content mentions a tool/concept that sounds nearly identical to the query, assume they refer to the same thing.

OUTPUT FORMAT (HTML only):
- Use <p> for paragraphs and insert a single blank line between consecutive <p> blocks.
- Use <strong> for key terms (2–4 words). Use <ul><li> for key points.
- Insert a single blank line between each <li> item in lists for better readability.
- CRITICAL: ALL content must be wrapped in proper HTML tags. Never output plain text lines without tags.
- For any bulleted or numbered content, ALWAYS use <ul><li> or <ol><li> tags.
- Never use plain text bullets (•, -, *) without wrapping in <li> tags.
- Never output raw text paragraphs - always wrap in <p> tags.
- No markdown; no plain text outside HTML (except the out-of-scope line).

BREVITY FOR LITE MODE:
- Respond in 2-3 concise sentences maximum wrapped in <p> tags.
- Avoid lists unless absolutely necessary.
- Be direct and educational. No meta-commentary.

EXAMPLE OF CORRECT FORMATTING:
<p>Machine learning enables computers to learn from data without explicit programming.</p>

<p>It powers applications like Netflix recommendations and email spam filters.</p>

Answer:"""
        
        # OPTIMIZATION: condense_question_llm=None skips expensive condense step
        # Lite queries are simple, don't need conversation history rephrasing
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.lite_llm,
            retriever=self.lite_retriever,
            memory=self.memory,
            condense_question_llm=None,  # Skip condense = saves 1 API call
            combine_docs_chain_kwargs={
                "prompt": PromptTemplate(
                    template=lite_qa_template,
                    input_variables=["context", "chat_history", "question"]
                )
            },
            return_source_documents=False,
            verbose=False
        )
        
        logger.info("[HYBRID_RAG_ENGINE] ✅ Lite chain built (1 API call per query)")
        return chain
    
    def _build_full_chain(self) -> ConversationalRetrievalChain:
        """Build full chain for detailed, comprehensive responses."""
        
        condense_template = """Given this conversation, rephrase the follow-up as a standalone question.

Chat History:
{chat_history}

Follow Up: {question}
Standalone question:"""
        
        # Full QA prompt - SYNCED WITH PROMPTBUILDER
        full_qa_template = """You are AI Shine, an expert educational assistant specializing in Artificial Intelligence, Machine Learning, and related technologies.

Context from knowledge base:
{context}

Chat History:
{chat_history}

Question: {question}

SCOPE (strict):
- Only answer topics in: Artificial Intelligence, Machine Learning, Deep Learning, Data Science, Natural Language Processing, Computer Vision, AI Applications and Ethics, AI-powered Education, Programming for AI/ML (e.g., Python, TensorFlow, PyTorch).
- If out-of-scope, respond exactly (plain text, no HTML):
"⚠️ I specialize in AI and Machine Learning topics. I'd be happy to help with questions about [suggest 2-3 related AI/ML topics]."
- Start that out-of-scope line with the ⚠️ emoji.

KNOWLEDGE-LOCK (critical):
- Use ONLY the provided educational content for this turn. Do NOT add roles, items, examples, claims, steps, or statistics that are not present in the content.
- HALLUCINATION GUARDRAIL: If the KB content mentions a concept (e.g., "machine learning") but doesn't provide a complete definition, structure your answer as follows:
  [1] Lead with a clear, concise definition using your AI/ML training knowledge
  [2] Then naturally integrate KB examples and applications (e.g., "It's used in Netflix recommendations for...")
  [3] Do NOT start with phrases like "The provided content doesn't define..." or "From a general AI perspective..." - just provide the definition directly and weave in KB examples
- If the KB has NO information on a domain-specific AI/ML topic, you may provide a brief, accurate definition from your training knowledge, but acknowledge that specific applications aren't in the provided materials.
- NEVER invent examples, statistics, specific claims, roles, or applications that are not present in the KB content. Only provide general definitions when absolutely necessary.
- If details are missing, structure as: "[Clear definition]. Based on the content, [integrate KB examples]."
- Do NOT switch to prompting frameworks unless explicitly asked.

OUTPUT FORMAT (HTML only):
- Use <p> for paragraphs and insert a single blank line between consecutive <p> blocks.
- Use <strong> for key terms (2–4 words). Use <ul><li> for key points.
- Insert a single blank line between each <li> item in lists for better readability.
- CRITICAL: ALL content must be wrapped in proper HTML tags. Never output plain text lines without tags.
- For any bulleted or numbered content, ALWAYS use <ul><li> or <ol><li> tags.
- Never use plain text bullets (•, -, *) without wrapping in <li> tags.
- Never output raw text paragraphs - always wrap in <p> tags.
- No markdown; no plain text outside HTML (except the out-of-scope line).

EXAMPLE OF CORRECT FORMATTING:
<p>AI significantly aids in visualizing complex topics by transforming abstract information into intelligent, digestible visuals.</p>

<p>Here are some ways AI helps:</p>

<ul>
<li>AI can create labeled diagrams for scientific concepts (e.g., human heart, digestive system)</li>

<li>It can animate processes like mitosis or the water cycle</li>

<li>AI excels at converting tables into colorful charts (pie, bar, line)</li>
</ul>

ENUMERATION POLICY (complete in all modes):
- If the KB content has a numbered/bulleted list, ensure EVERY item appears.
- Show items 1–3 with full descriptions.
- Show items 4..N with full descriptions as well (do NOT collapse or use "Learn more" buttons).
- Insert a single blank line between each <li> item for readability.
- Do NOT add, merge, rename, or omit items. Use ONLY KB-provided examples/details.

DETAILED/CONTINUE BEHAVIOR:
- For "continue", "more detail", "elaborate", "go deeper", etc.:
  - Do NOT restate earlier content; extend using ONLY details present in the KB.
  - If a list began earlier, resume at the next item and complete all remaining KB items before adding extra depth.
  - Aim for 4–6 <p> paragraphs.

SYNTHESIS APPROACH:
- When KB has comprehensive content: Synthesize ALL relevant details into flowing explanations
- Use ALL relevant details from the KB - don't just extract one sentence
- Explain HOW things work, WHAT they do, and WHY they matter
- Lead with definitions when KB mentions concepts without explaining them
- Paraphrase KB content in your own words - avoid direct quotes
- Only state "I don't have detailed information" if KB truly lacks substance
- NEVER invent specific: tools, statistics, company names, features, or applications not in KB

KB-FIRST WITH MINIMAL APPEND:
- Only after fully using the KB content may you append minimal, generic AI knowledge to fill gaps; keep such additions short and aligned with the KB.
- Never add or replace list items beyond those in the KB.

CHUNKING AND CONTINUATION MARKER:
- Write in logically complete sections. If you must stop before completing all sections or list items, end with: <p><em>Write 'continue' to keep generating...</em></p>
- On the next turn, resume exactly where you stopped without repeating earlier content.

TONE:
- Teach clearly and naturally with a warm, encouraging, professional tone. Keep small talk minimal.
- Direct and educational. NO meta-commentary.

Answer:"""
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.flash_llm,
            retriever=self.full_retriever,
            memory=self.memory,
            condense_question_prompt=PromptTemplate(
                template=condense_template,
                input_variables=["chat_history", "question"]
            ),
            combine_docs_chain_kwargs={
                "prompt": PromptTemplate(
                    template=full_qa_template,
                    input_variables=["context", "chat_history", "question"]
                )
            },
            return_source_documents=False,
            verbose=False
        )
        
        logger.info("[HYBRID_RAG_ENGINE] ✅ Full chain built (2 API calls per query)")
        return chain
    
    def _route_query(self, query: str, chat_history: List[Message]) -> str:
        """
        Route query to appropriate chain based on complexity signals.
        
        Args:
            query: User query
            chat_history: Conversation history
        
        Returns:
            "lite" or "full"
        """
        query_lower = query.lower().strip()
        
        # Explicit brief request → lite
        if self.brief_regex.search(query_lower):
            logger.info("[ROUTER] Brief signal detected → Lite chain")
            return "lite"
        
        # Explicit detailed request → full
        if self.detailed_regex.search(query_lower):
            logger.info("[ROUTER] Detailed signal detected → Full chain")
            return "full"
        
        # Default: lite (fast & cheap for first queries on any topic)
        logger.info("[ROUTER] Default routing → Lite chain")
        return "lite"
    
    def process_query(
        self,
        query: str,
        chat_history: List[Message]
    ) -> Dict[str, Any]:
        """
        Process user query with intelligent routing.
        
        Args:
            query: Current user query
            chat_history: Full conversation history
        
        Returns:
            Dict with 'answer' (str) and 'type' (str)
        """
        import uuid
        request_id = str(uuid.uuid4())[:8]  # Short request ID for tracking
        
        try:
            logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 📝 NEW REQUEST: '{query[:100]}...'")
            logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 📊 Chat history length: {len(chat_history)} messages")
            
            # Handle greetings
            if self.prompt_builder.greeting_regex.match(query.strip()):
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 👋 GREETING DETECTED")
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ✅ Response via regex pattern (0 API calls)")
                self.memory.clear()
                return {
                    "answer": self.prompt_builder.build_greeting_response(),
                    "type": "greeting"
                }
            
            # Handle farewells
            if self.prompt_builder.farewell_regex.match(query.strip()):
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 👋 FAREWELL DETECTED")
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ✅ Response via regex pattern (0 API calls)")
                self.memory.clear()
                return {
                    "answer": self.prompt_builder.build_farewell_response(),
                    "type": "text"
                }
            
            # Route to appropriate chain
            route = self._route_query(query, chat_history)
            logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 🎯 ROUTING DECISION: {route.upper()}")
            
            if route == "lite":
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 🚀 LITE CHAIN SELECTED")
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 💡 Reason: Default/first query OR brief signal detected")
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 📊 Expected API calls: 1")
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ⚡ API Call #1 START: Answer generation (Flash-Lite, 15 RPM)")
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}]    Purpose: Generate brief response from KB context")
                
                response = self.lite_chain.invoke({"question": query})
                
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ✅ API Call #1 COMPLETE: Answer generation")
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 📈 Total API calls for this request: 1")
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 🎯 Model used: gemini-2.5-flash-lite")
            else:
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 🔥 FULL CHAIN SELECTED")
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 💡 Reason: Detailed signal detected (tell me more/expand/in detail)")
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 📊 Expected API calls: 2")
                
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ⚡ API Call #1 START: Condense question (Flash, 1500 RPM)")
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}]    Purpose: Rephrase '{query}' with chat history context")
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}]    Input: '{query}' + {len(chat_history)} previous messages")
                
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ⚡ API Call #2 START: Answer generation (Flash, 1500 RPM)")
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}]    Purpose: Generate detailed response with HTML formatting")
                
                response = self.full_chain.invoke({"question": query})
                
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ✅ API Call #1 COMPLETE: Condense question")
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ✅ API Call #2 COMPLETE: Answer generation")
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 📈 Total API calls for this request: 2")
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 🎯 Model used: gemini-2.5-flash")
            
            answer = response.get("answer", "")
            
            # Clean response
            answer = self._clean_response(answer)
            
            # Classify response type
            response_type = self._classify_response(answer)
            
            logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ✅ RESPONSE COMPLETE")
            logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 📝 Response length: {len(answer)} characters")
            logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 🏷️  Response type: {response_type}")
            logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 🚦 Route used: {route}")
            logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 💾 Memory stored in buffer (0 API calls)")
            logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 📊 FINAL COUNT - API calls made: {1 if route == 'lite' else 2}")
            logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ════════════════════════════════════════")
            
            return {
                "answer": answer,
                "type": response_type,
                "route": route,
                "request_id": request_id  # For frontend debugging
            }
            
        except Exception as e:
            logger.error(f"[HYBRID_RAG_ENGINE] [{request_id}] ❌ PIPELINE FAILURE: {e}", exc_info=True)
            return {
                "answer": "⚠️ An unexpected error occurred. Please try your question again.",
                "type": "text"
            }
    
    def _clean_response(self, response: str) -> str:
        """Clean and format LLM response."""
        import re
        
        # Convert markdown bold to HTML
        response = re.sub(r'\*\*([^\*]+)\*\*', r'<strong>\1</strong>', response)
        
        # Remove any stray asterisk bullets
        response = response.replace('* ', '• ')
        
        return response.strip()
    
    def _classify_response(self, response: str) -> str:
        """Classify response type for frontend rendering."""
        if response.startswith("⚠") or "I specialize in AI and Machine Learning topics" in response:
            return "decline"
        
        if "I don't have" in response or "I don't know" in response:
            return "decline"
        
        return "text"
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            self.memory.clear()
            logger.info("[HYBRID_RAG_ENGINE] ✅ Cleanup complete")
        except Exception as e:
            logger.error(f"[HYBRID_RAG_ENGINE] Cleanup error: {e}")




















# initial hybrid pipeline, full chain uses its own internal prompting isntead of prompt builder
# """
# Hybrid RAG Engine
# Intelligent routing between Flash-Lite (fast/brief) and Flash (detailed) based on query complexity.
# Optimized to minimize API calls.
# """
# import logging
# import re
# from typing import List, Dict, Any
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# from langchain_core.prompts import PromptTemplate
# from Backend.models import Message
# from Backend.langchain_retriever import LangChainMongoRetriever
# from Backend.langchain_llm_client import create_langchain_gemini_client, create_langchain_gemini_lite_client
# from Backend.prompt_builder import PromptBuilder

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class HybridRAGEngine:
#     """
#     Hybrid RAG engine with intelligent routing:
#     - Flash-Lite: Fast, brief responses for initial queries
#     - Flash: Detailed, comprehensive responses for deep-dive requests
    
#     Optimized for minimal API calls:
#     - ConversationBufferMemory (no summarization calls)
#     - Lite chain skips condense step (1 API call only)
#     - Full chain uses condense when needed (2 API calls)
#     """
    
#     # Patterns for detecting deep-dive requests
#     DETAILED_PATTERNS = [
#         r'\btell\s+me\s+more\b',
#         r'\bexpand\b',
#         r'\bcontinue\b',
#         r'\bgo\s+deeper\b',
#         r'\bin\s+detail\b',
#         r'\belaborate\b',
#         r'\bexplain\s+further\b',
#         r'\bmore\s+information\b',
#         r'\bcomprehensive\b',
#         r'\btell\s+me\s+more\b',
#     ]
    
#     # Patterns for brief responses
#     BRIEF_PATTERNS = [
#         r'\bbriefly\b',
#         r'\bquick\b',
#         r'\bshort\b',
#         r'\bsummarize\b',
#         r'\bin\s+short\b',
#     ]
    
#     def __init__(self):
#         """Initialize hybrid RAG components."""
#         try:
#             logger.info("[HYBRID_RAG_ENGINE] 🔧 Initializing with API call optimization...")
            
#             # Initialize retrievers (same underlying MongoDB, different top_k)
#             self.lite_retriever = LangChainMongoRetriever(max_results=1)  # Top 1 for lite
#             self.full_retriever = LangChainMongoRetriever(max_results=3)  # Top 3 for full
            
#             # Initialize LLMs
#             self.flash_llm = create_langchain_gemini_client()  # Full power
#             self.lite_llm = create_langchain_gemini_lite_client()  # Fast & cheap
            
#             # ConversationBufferMemory - NO SUMMARIZATION API CALLS
#             # Stores full history in memory (uses more tokens but zero API calls)
#             self.memory = ConversationBufferMemory(
#                 memory_key="chat_history",
#                 return_messages=True,
#                 output_key="answer",
#                 input_key="question"
#             )
            
#             logger.info("[HYBRID_RAG_ENGINE] ✅ Using ConversationBufferMemory (0 API calls for memory)")
            
#             # Prompt builder for greeting/farewell
#             self.prompt_builder = PromptBuilder()
            
#             # Compile regex patterns
#             self.detailed_regex = re.compile('|'.join(self.DETAILED_PATTERNS), re.IGNORECASE)
#             self.brief_regex = re.compile('|'.join(self.BRIEF_PATTERNS), re.IGNORECASE)
            
#             # Build chains
#             self.lite_chain = self._build_lite_chain()
#             self.full_chain = self._build_full_chain()
            
#             logger.info("[HYBRID_RAG_ENGINE] ✅ All components initialized")
#             logger.info("[HYBRID_RAG_ENGINE] 📊 API Call Breakdown:")
#             logger.info("[HYBRID_RAG_ENGINE]    Lite query: 1 API call (answer only)")
#             logger.info("[HYBRID_RAG_ENGINE]    Full query: 2 API calls (condense + answer)")
#             logger.info("[HYBRID_RAG_ENGINE]    Memory: 0 API calls (buffer storage)")
            
#         except Exception as e:
#             logger.error(f"[HYBRID_RAG_ENGINE_ERR] Initialization failed: {e}")
#             raise
    
#     def _build_lite_chain(self) -> ConversationalRetrievalChain:
#         """Build lite chain for brief, fast responses. OPTIMIZED: No condense step."""
        
#         # Lite QA prompt - emphasizes brevity
#         lite_qa_template = """You are AI Shine, an AI/ML educational assistant.

# Context from knowledge base:
# {context}

# Chat History:
# {chat_history}

# Question: {question}

# RULES:
# 1. SCOPE: AI, ML, Deep Learning, Data Science, NLP, Computer Vision, AI Applications only
# 2. BREVITY: Respond in 2-3 concise sentences maximum. No lists unless absolutely necessary.
# 3. SYNTHESIS: Lead with definition if KB lacks it, then add one KB example
# 4. TONE: Direct and educational. No meta-commentary.
# 5. FORMAT: Use <p> tags only. No lists for lite responses.
# 6. OUT OF SCOPE: "⚠️ I specialize in AI and Machine Learning topics."

# Answer:"""
        
#         # OPTIMIZATION: condense_question_llm=None skips expensive condense step
#         # Lite queries are simple, don't need conversation history rephrasing
#         chain = ConversationalRetrievalChain.from_llm(
#             llm=self.lite_llm,
#             retriever=self.lite_retriever,
#             memory=self.memory,
#             condense_question_llm=None,  # Skip condense = saves 1 API call
#             combine_docs_chain_kwargs={
#                 "prompt": PromptTemplate(
#                     template=lite_qa_template,
#                     input_variables=["context", "chat_history", "question"]
#                 )
#             },
#             return_source_documents=False,
#             verbose=False
#         )
        
#         logger.info("[HYBRID_RAG_ENGINE] ✅ Lite chain built (1 API call per query)")
#         return chain
    
#     def _build_full_chain(self) -> ConversationalRetrievalChain:
#         """Build full chain for detailed, comprehensive responses."""
        
#         condense_template = """Given this conversation, rephrase the follow-up as a standalone question.

# Chat History:
# {chat_history}

# Follow Up: {question}
# Standalone question:"""
        
#         # Full QA prompt - emphasizes detail
#         full_qa_template = """You are AI Shine, an AI/ML educational assistant.

# Context from knowledge base:
# {context}

# Chat History:
# {chat_history}

# Question: {question}

# RULES:
# 1. SCOPE: AI, ML, Deep Learning, Data Science, NLP, Computer Vision, AI Applications only
# 2. SYNTHESIS:
#    - If KB mentions concept without defining: lead with definition, then add KB examples
#    - Paraphrase all KB content naturally - never use direct quotes or preserve quotation marks
#    - NEVER invent examples, tools, or statistics not in KB
#    - Only provide general definitions when needed
# 3. TONE: Direct and educational. NO meta-commentary
# 4. FORMAT:
#    - <p> for paragraphs (blank line between)
#    - <ul><li> for lists (blank line between items)
#    - <strong> for key terms (2-4 words)
#    - If long: end with <p><em>Write 'continue' to keep generating...</em></p>
# 5. OUT OF SCOPE: "⚠️ I specialize in AI and Machine Learning topics. I'd be happy to help with questions about [suggest 2-3 AI/ML topics]."

# Answer:"""
        
#         chain = ConversationalRetrievalChain.from_llm(
#             llm=self.flash_llm,
#             retriever=self.full_retriever,
#             memory=self.memory,
#             condense_question_prompt=PromptTemplate(
#                 template=condense_template,
#                 input_variables=["chat_history", "question"]
#             ),
#             combine_docs_chain_kwargs={
#                 "prompt": PromptTemplate(
#                     template=full_qa_template,
#                     input_variables=["context", "chat_history", "question"]
#                 )
#             },
#             return_source_documents=False,
#             verbose=False
#         )
        
#         logger.info("[HYBRID_RAG_ENGINE] ✅ Full chain built (2 API calls per query)")
#         return chain
    
#     def _route_query(self, query: str, chat_history: List[Message]) -> str:
#         """
#         Route query to appropriate chain based on complexity signals.
        
#         Args:
#             query: User query
#             chat_history: Conversation history
        
#         Returns:
#             "lite" or "full"
#         """
#         query_lower = query.lower().strip()
        
#         # Explicit brief request → lite
#         if self.brief_regex.search(query_lower):
#             logger.info("[ROUTER] Brief signal detected → Lite chain")
#             return "lite"
        
#         # Explicit detailed request → full
#         if self.detailed_regex.search(query_lower):
#             logger.info("[ROUTER] Detailed signal detected → Full chain")
#             return "full"
        
#         # Default: lite (fast & cheap for first queries on any topic)
#         logger.info("[ROUTER] Default routing → Lite chain")
#         return "lite"
    
#     def process_query(
#         self,
#         query: str,
#         chat_history: List[Message]
#     ) -> Dict[str, Any]:
#         """
#         Process user query with intelligent routing.
        
#         Args:
#             query: Current user query
#             chat_history: Full conversation history
        
#         Returns:
#             Dict with 'answer' (str) and 'type' (str)
#         """
#         import uuid
#         request_id = str(uuid.uuid4())[:8]  # Short request ID for tracking
        
#         try:
#             logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 📝 NEW REQUEST: '{query[:100]}...'")
#             logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 📊 Chat history length: {len(chat_history)} messages")
            
#             # Handle greetings
#             if self.prompt_builder.greeting_regex.match(query.strip()):
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 👋 GREETING DETECTED")
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ✅ Response via regex pattern (0 API calls)")
#                 self.memory.clear()
#                 return {
#                     "answer": self.prompt_builder.build_greeting_response(),
#                     "type": "greeting"
#                 }
            
#             # Handle farewells
#             if self.prompt_builder.farewell_regex.match(query.strip()):
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 👋 FAREWELL DETECTED")
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ✅ Response via regex pattern (0 API calls)")
#                 self.memory.clear()
#                 return {
#                     "answer": self.prompt_builder.build_farewell_response(),
#                     "type": "text"
#                 }
            
#             # Route to appropriate chain
#             route = self._route_query(query, chat_history)
#             logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 🎯 ROUTING DECISION: {route.upper()}")
            
#             if route == "lite":
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 🚀 LITE CHAIN SELECTED")
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 💡 Reason: Default/first query OR brief signal detected")
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 📊 Expected API calls: 1")
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ⚡ API Call #1 START: Answer generation (Flash-Lite, 15 RPM)")
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}]    Purpose: Generate brief response from KB context")
                
#                 response = self.lite_chain.invoke({"question": query})
                
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ✅ API Call #1 COMPLETE: Answer generation")
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 📈 Total API calls for this request: 1")
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 🎯 Model used: gemini-2.5-flash-lite")
#             else:
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 🔥 FULL CHAIN SELECTED")
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 💡 Reason: Detailed signal detected (tell me more/expand/in detail)")
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 📊 Expected API calls: 2")
                
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ⚡ API Call #1 START: Condense question (Flash, 1500 RPM)")
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}]    Purpose: Rephrase '{query}' with chat history context")
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}]    Input: '{query}' + {len(chat_history)} previous messages")
                
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ⚡ API Call #2 START: Answer generation (Flash, 1500 RPM)")
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}]    Purpose: Generate detailed response with HTML formatting")
                
#                 response = self.full_chain.invoke({"question": query})
                
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ✅ API Call #1 COMPLETE: Condense question")
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ✅ API Call #2 COMPLETE: Answer generation")
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 📈 Total API calls for this request: 2")
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 🎯 Model used: gemini-2.5-flash")
            
#             answer = response.get("answer", "")
            
#             # Clean response
#             answer = self._clean_response(answer)
            
#             # Classify response type
#             response_type = self._classify_response(answer)
            
#             logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ✅ RESPONSE COMPLETE")
#             logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 📝 Response length: {len(answer)} characters")
#             logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 🏷️  Response type: {response_type}")
#             logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 🚦 Route used: {route}")
#             logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 💾 Memory stored in buffer (0 API calls)")
#             logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 📊 FINAL COUNT - API calls made: {1 if route == 'lite' else 2}")
#             logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ════════════════════════════════════════")
            
#             return {
#                 "answer": answer,
#                 "type": response_type,
#                 "route": route,
#                 "request_id": request_id  # For frontend debugging
#             }
            
#         except Exception as e:
#             logger.error(f"[HYBRID_RAG_ENGINE] [{request_id}] ❌ PIPELINE FAILURE: {e}", exc_info=True)
#             return {
#                 "answer": "⚠️ An unexpected error occurred. Please try your question again.",
#                 "type": "text"
#             }
    
#     def _clean_response(self, response: str) -> str:
#         """Clean and format LLM response."""
#         import re
        
#         # Convert markdown bold to HTML
#         response = re.sub(r'\*\*([^\*]+)\*\*', r'<strong>\1</strong>', response)
        
#         # Remove any stray asterisk bullets
#         response = response.replace('* ', '• ')
        
#         return response.strip()
    
#     def _classify_response(self, response: str) -> str:
#         """Classify response type for frontend rendering."""
#         if response.startswith("⚠") or "I specialize in AI and Machine Learning topics" in response:
#             return "decline"
        
#         if "I don't have" in response or "I don't know" in response:
#             return "decline"
        
#         return "text"
    
#     def cleanup(self):
#         """Cleanup resources."""
#         try:
#             self.memory.clear()
#             logger.info("[HYBRID_RAG_ENGINE] ✅ Cleanup complete")
#         except Exception as e:
#             logger.error(f"[HYBRID_RAG_ENGINE] Cleanup error: {e}")