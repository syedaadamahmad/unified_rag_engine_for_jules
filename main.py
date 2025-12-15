import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from Backend.models import ChatRequest, ChatResponse
from Backend.unified_rag_engine import UnifiedFlashEngine

load_dotenv()

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
        unified_engine = UnifiedFlashEngine()
        logger.info("[STARTUP] ✅ Unified Flash Engine ready")
    except Exception as e:
        logger.error(f"[STARTUP] ❌ Failed to initialize engine: {e}", exc_info=True)
        raise
    
    yield
    
    if unified_engine:
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
    
    Returns complete response after generation finishes.
    Use this for testing or simple integrations.
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
    Streaming chat endpoint (recommended for production).
    
    Returns NDJSON stream (Newline Delimited JSON).
    Each line is a JSON object with:
    - {"type": "text", "status": "generating"} - Initial signal
    - {"answer_chunk": "text"} - Token fragments to append
    - {"answer": "full", "type": "greeting"} - Complete responses
    
    Frontend should:
    1. Read stream line-by-line
    2. Parse each line as JSON
    3. Append answer_chunk to display
    4. Handle complete messages (greetings, errors)
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