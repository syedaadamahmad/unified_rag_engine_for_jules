"""
MongoDB Atlas Vector Search Client
Render-compatible with retry logic, connection pooling, and graceful error handling.
"""
import os
import logging
import time
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure, ServerSelectionTimeoutError
from dotenv import load_dotenv
import certifi

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MongoDBClient:
    """MongoDB Atlas Vector Search client with Render cloud compatibility."""
    
    def __init__(self, max_retries: int = 3, retry_delay: int = 2):
        """
        Initialize MongoDB client with retry logic.
        
        Args:
            max_retries: Number of connection retry attempts
            retry_delay: Seconds to wait between retries
        """
        load_dotenv(override=True)
        
        self.uri = os.getenv("MONGO_DB_URI")
        self.db_name = os.getenv("DB_NAME")
        self.collection_name = "module_vectors"
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Validate required env vars
        if not self.uri:
            raise ValueError("[MONGO_ERR] MONGO_DB_URI not set in environment")
        
        if not self.db_name:
            raise ValueError("[MONGO_ERR] DB_NAME not set in environment")
        
        logger.info(f"[MONGO_INIT] Connecting to database: {self.db_name}")
        
        self.client = None
        self.db = None
        self.collection = None
        
        self._connect_with_retry()
    
    def _connect_with_retry(self):
        """Establish MongoDB connection with retry logic for cloud environments."""
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"[MONGO] Connection attempt {attempt}/{self.max_retries}")
                
                self.client = MongoClient(
                    self.uri,
                    tls=True,
                    tlsCAFile=certifi.where(),
                    tlsAllowInvalidCertificates=True,
                    tlsAllowInvalidHostnames=True,
                    retryWrites=True,
                    retryReads=True,
                    serverSelectionTimeoutMS=20000,
                    connectTimeoutMS=20000,
                    socketTimeoutMS=30000,
                    maxPoolSize=10,
                    minPoolSize=2,
                )
                # Test connection
                self.client.admin.command('ping')
                
                self.db = self.client[self.db_name]
                self.collection = self.db[self.collection_name]
                
                logger.info(f"[MONGO_OK] Connected to {self.db_name}.{self.collection_name}")
                return
                
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                logger.warning(f"[MONGO] Attempt {attempt} failed: {e}")
                
                if attempt < self.max_retries:
                    logger.info(f"[MONGO] Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"[MONGO_ERR] All {self.max_retries} connection attempts failed")
                    raise ConnectionFailure(f"Failed to connect after {self.max_retries} attempts: {e}")
    
    def ensure_connection(self):
        """Verify connection is alive, reconnect if needed."""
        try:
            self.client.admin.command('ping')
        except Exception as e:
            logger.warning(f"[MONGO] Connection lost, reconnecting: {e}")
            self._connect_with_retry()
    
    def vector_search(
        self,
        query_embedding: List[float],
        limit: int = 5,
        similarity_threshold: float = 0.55,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search with optional metadata filtering.
        
        Args:
            query_embedding: 1024-dim embedding vector
            limit: Max results to return
            similarity_threshold: Minimum cosine similarity score
            metadata_filters: Optional dict of metadata filters
        
        Returns:
            List of documents with score >= threshold, sorted by relevance
        """
        try:
            self.ensure_connection()
            
            # Build aggregation pipeline
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": limit * 20,
                        "limit": limit
                    }
                },
                {
                    "$addFields": {
                        "score": {"$meta": "vectorSearchScore"}
                    }
                },
                {
                    "$match": {
                        "score": {"$gte": similarity_threshold}
                    }
                }
            ]
            
            # Add metadata filters if provided
            if metadata_filters:
                pipeline.append({"$match": metadata_filters})
            
            # Project fields
            pipeline.append({
                "$project": {
                    "_id": 1,
                    "topic": 1,
                    "category": 1,
                    "level": 1,
                    "summary": 1,
                    "content": 1,
                    "keywords": 1,
                    "module_name": 1,
                    "source": 1,
                    "presentation_data": 1,
                    "score": 1
                }
            })
            
            results = list(self.collection.aggregate(pipeline, maxTimeMS=30000))
            
            logger.info(f"[VECTOR_SEARCH] Retrieved {len(results)} chunks above threshold {similarity_threshold}")
            if results:
                logger.info(f"[VECTOR_SEARCH_DEBUG] Top: {results[0].get('topic', 'N/A')} ({results[0].get('score', 0):.3f})")
                logger.info(f"[VECTOR_SEARCH_DEBUG] Source: {results[0].get('source', 'N/A')}")
            
            return results
            
        except OperationFailure as e:
            logger.error(f"[VECTOR_SEARCH_ERR] Operation failed: {e}")
            return []
        except Exception as e:
            logger.error(f"[VECTOR_SEARCH_ERR] Unexpected error: {e}", exc_info=True)
            return []
    
    def insert_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Bulk insert documents with embeddings."""
        try:
            if not documents:
                logger.warning("[INSERT] No documents to insert")
                return False
            
            self.ensure_connection()
            
            result = self.collection.insert_many(documents, ordered=False)
            logger.info(f"[INSERT_OK] Inserted {len(result.inserted_ids)} documents")
            return True
            
        except Exception as e:
            logger.error(f"[INSERT_ERR] {e}")
            return False
    
    def close(self):
        """Close MongoDB connection."""
        if self.client:
            try:
                self.client.close()
                logger.info("[MONGO_CLOSE] Connection closed")
            except Exception as e:
                logger.warning(f"[MONGO_CLOSE] Error during close: {e}")