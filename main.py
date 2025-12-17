import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from Backend.models import ChatRequest, ChatResponse
from Backend.unified_rag_engine import UnifiedFlashEngine

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

unified_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager"""
    global unified_engine
    
    logger.info("[STARTUP] Initializing Unified Flash Engine...")
    try:
        if os.getenv("PYTEST_RUNNING"):
            from unittest.mock import MagicMock, AsyncMock
            unified_engine = MagicMock()
            async def mock_process_query(query, *args, **kwargs):
                if "hello" in query.lower():
                    return {"answer": "Hello! How can I help you?", "type": "greeting"}
                return {"answer": "Mocked answer", "type": "text"}
            unified_engine.process_query = AsyncMock(side_effect=mock_process_query)
            logger.info("[STARTUP] ✅ Mock Unified Flash Engine ready")
        else:
            unified_engine = UnifiedFlashEngine()
            logger.info("[STARTUP] ✅ Unified Flash Engine ready")
    except Exception as e:
        logger.error(f"[STARTUP] ❌ Failed to initialize engine: {e}", exc_info=True)
        raise
    
    yield
    
    if unified_engine and not os.getenv("PYTEST_RUNNING"):
        logger.info("[SHUTDOWN] Cleaning up engine...")
        unified_engine.cleanup()
        logger.info("[SHUTDOWN] ✅ Shutdown complete")

app = FastAPI(
    title="AI Shine API",
    description="Educational AI/ML chatbot with RAG and streaming",
    version="2.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AI Shine API",
        "version": "2.1.0",
        "engine": "Unified Flash (Streaming)",
        "endpoints": {
            "chat": "/chat (non-streaming)",
            "chat_stream": "/chat_stream (streaming)"
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Non-streaming chat endpoint (backward compatible).
    """
    if not unified_engine:
        logger.error("[CHAT] Engine not initialized")
        raise HTTPException(status_code=503, detail="Engine unavailable")
    
    try:
        current_query = None
        for msg in reversed(request.chat_history):
            if msg.role == "human":
                current_query = msg.content
                break
        
        if not current_query:
            logger.warning("[CHAT] No user message in chat_history")
            raise HTTPException(status_code=400, detail="No user message found in chat history")

        logger.info(f"[CHAT] Processing query: '{current_query[:50]}...'")

        response = await unified_engine.process_query(
            query=current_query,
            chat_history=request.chat_history
        )
        
        logger.info(f"[CHAT] ✅ Response generated (type={response['type']})")
        
        return ChatResponse(
            answer=response["answer"],
            type=response["type"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CHAT] ❌ Unexpected error: {type(e).__name__}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/chat_stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint.
    """
    if not unified_engine:
        logger.error("[CHAT_STREAM] Engine not initialized")
        raise HTTPException(status_code=503, detail="Engine unavailable")

    try:
        current_query = None
        for msg in reversed(request.chat_history):
            if msg.role == "human":
                current_query = msg.content
                break
        
        if not current_query:
            logger.warning("[CHAT_STREAM] No user message in chat_history")
            raise HTTPException(status_code=400, detail="No user message found in chat history")

        logger.info(f"[CHAT_STREAM] Streaming request: '{current_query[:50]}...'")

        return StreamingResponse(
            unified_engine.process_query_stream(
                query=current_query,
                chat_history=request.chat_history
            ),
            media_type="application/x-ndjson"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CHAT_STREAM] ❌ Unexpected error: {type(e).__name__}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy" if unified_engine else "degraded",
        "engine_initialized": unified_engine is not None,
        "model": "gemini-2.5-flash",
        "retriever": "MongoDB Atlas Vector Search",
        "embeddings": "AWS Bedrock Titan v2",
        "streaming": "enabled"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
