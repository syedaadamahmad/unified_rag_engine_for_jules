"""
LangChain MongoDB Retriever unified
"""
import os
import logging
from typing import List, Dict, Any, Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_aws import BedrockEmbeddings
from pymongo import MongoClient
from dotenv import load_dotenv
import certifi

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LangChainMongoRetriever(BaseRetriever):
    """
    LangChain-compatible retriever for MongoDB Atlas vector search.
    """
    
    # Pydantic fields
    vector_search: Any = None
    similarity_threshold: float = 0.55
    max_results: int = 3 # Default upped to 5 for better recall
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, max_results: Optional[int] = None, **kwargs):
        """Initialize MongoDB vector search."""
        # Get MongoDB connection details
        mongo_uri = os.getenv("MONGO_DB_URI")
        db_name = os.getenv("DB_NAME")
        collection_name = "module_vectors"
        
        if not mongo_uri or not db_name:
            raise ValueError("[LANGCHAIN_RETRIEVER] MONGO_DB_URI or DB_NAME not set")
        
        # Initialize MongoDB client
        client = MongoClient(
            mongo_uri,
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
        
        collection = client[db_name][collection_name]
        
        # Initialize Bedrock embeddings
        embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
            region_name=os.getenv("AWS_DEFAULT_REGION", "ap-south-1"),
            credentials_profile_name=None
        )
        
        # Initialize Vector Search
        vector_search_instance = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name="vector_index",
            text_key="content",
            embedding_key="embedding"
        )
        
        # Handle max_results override if provided in init
        config = kwargs
        if max_results is not None:
            config["max_results"] = max_results

        # Initialize parent
        super().__init__(vector_search=vector_search_instance, **config)
        
        logger.info(f"[LANGCHAIN_RETRIEVER] Initialized. Max Results: {self.max_results}")
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Retrieve relevant documents."""
        try:
            # Primary search
            results = self.vector_search.similarity_search_with_score(
                query=query,
                k=self.max_results,
                pre_filter={"source": "knowledge_base"}
            )
            
            # Filter by threshold
            filtered_results = [
                (doc, score) for doc, score in results 
                if score >= self.similarity_threshold
            ]
            
            if filtered_results:
                return [doc for doc, _ in filtered_results]
            
            # Fallback for lower threshold
            logger.info(f"[LANGCHAIN_RETRIEVER] No results > {self.similarity_threshold}, trying 0.45")
            
            results_lower = self.vector_search.similarity_search_with_score(
                query=query,
                k=self.max_results,
                pre_filter={"source": "knowledge_base"}
            )
            
            filtered_lower = [
                (doc, score) for doc, score in results_lower 
                if score >= 0.45
            ]
            
            if filtered_lower:
                return [doc for doc, _ in filtered_lower]
            
            logger.warning("[LANGCHAIN_RETRIEVER] ❌ No results found")
            return []
            
        except Exception as e:
            logger.error(f"[LANGCHAIN_RETRIEVER] Error: {e}", exc_info=True)
            return []
    
    async def _aget_relevant_documents(self, query: str, *, run_manager=None):
        return self._get_relevant_documents(query, run_manager=run_manager)





























# """has everything except for live token streaming
# LangChain MongoDB Retriever
# Wraps MongoDB Atlas vector search with LangChain's BaseRetriever interface.
# """
# import os
# import logging
# from typing import List, Dict, Any
# from langchain_core.retrievers import BaseRetriever
# from langchain_core.documents import Document
# from langchain_core.callbacks import CallbackManagerForRetrieverRun
# from langchain_mongodb import MongoDBAtlasVectorSearch
# from langchain_aws import BedrockEmbeddings
# from pymongo import MongoClient
# from dotenv import load_dotenv
# import certifi

# load_dotenv()

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class LangChainMongoRetriever(BaseRetriever):
#     """
#     LangChain-compatible retriever for MongoDB Atlas vector search.
#     Uses native LangChain components for better integration.
#     """
    
#     # Pydantic fields - must be class attributes
#     vector_search: Any = None
#     similarity_threshold: float = 0.55
#     max_results: int = 3
    
#     class Config:
#         arbitrary_types_allowed = True
    
#     def __init__(self, **kwargs):
#         """Initialize MongoDB vector search with LangChain components."""
#         # Get MongoDB connection details
#         mongo_uri = os.getenv("MONGO_DB_URI")
#         db_name = os.getenv("DB_NAME")
#         collection_name = "module_vectors"
        
#         if not mongo_uri or not db_name:
#             raise ValueError("[LANGCHAIN_RETRIEVER] MONGO_DB_URI or DB_NAME not set")
        
#         # Initialize MongoDB client
#         client = MongoClient(
#             mongo_uri,
#             tls=True,
#             tlsCAFile=certifi.where(),
#             tlsAllowInvalidCertificates=True,
#             tlsAllowInvalidHostnames=True,
#             retryWrites=True,
#             retryReads=True,
#             serverSelectionTimeoutMS=20000,
#             connectTimeoutMS=20000,
#             socketTimeoutMS=30000,
#             maxPoolSize=10,
#             minPoolSize=2,
#         )
        
#         # Test connection
#         client.admin.command('ping')
#         logger.info(f"[LANGCHAIN_RETRIEVER] Connected to {db_name}.{collection_name}")
        
#         collection = client[db_name][collection_name]
        
#         # Initialize Bedrock embeddings with LangChain wrapper
#         embeddings = BedrockEmbeddings(
#             model_id="amazon.titan-embed-text-v2:0",
#             region_name=os.getenv("AWS_DEFAULT_REGION", "ap-south-1"),
#             credentials_profile_name=None  # Uses env vars
#         )
        
#         # Initialize LangChain's MongoDB vector search
#         vector_search_instance = MongoDBAtlasVectorSearch(
#             collection=collection,
#             embedding=embeddings,
#             index_name="vector_index",
#             text_key="content",
#             embedding_key="embedding"
#         )
        
#         # Initialize parent with vector_search set
#         super().__init__(vector_search=vector_search_instance, **kwargs)
        
#         logger.info(f"[LANGCHAIN_RETRIEVER] Initialized with threshold={self.similarity_threshold}")
    
#     def _get_relevant_documents(
#         self,
#         query: str,
#         *,
#         run_manager: CallbackManagerForRetrieverRun = None
#     ) -> List[Document]:
#         """
#         Retrieve relevant documents for a query.
        
#         Args:
#             query: User query string
#             run_manager: Callback manager (unused)
        
#         Returns:
#             List of LangChain Document objects
#         """
#         try:
#             logger.info(f"[LANGCHAIN_RETRIEVER] Query: {query}")
            
#             # Primary search with threshold 0.55
#             results = self.vector_search.similarity_search_with_score(
#                 query=query,
#                 k=self.max_results,
#                 pre_filter={"source": "knowledge_base"}
#             )
            
#             # Filter by similarity threshold
#             filtered_results = [
#                 (doc, score) for doc, score in results 
#                 if score >= self.similarity_threshold
#             ]
            
#             if filtered_results:
#                 logger.info(f"[LANGCHAIN_RETRIEVER] ✅ Found {len(filtered_results)} documents")
#                 documents = [doc for doc, _ in filtered_results]
                
#                 # Log top result for debugging
#                 if documents:
#                     top_metadata = documents[0].metadata
#                     logger.info(f"[LANGCHAIN_RETRIEVER] Top: {top_metadata.get('topic', 'N/A')}")
                
#                 return documents
            
#             # HALLUCINATION GUARDRAIL: Fallback to lower threshold (0.45)
#             # This allows LLM to synthesize from lower-scoring but relevant chunks
#             # rather than inventing content when no high-confidence matches exist
#             logger.info(f"[LANGCHAIN_RETRIEVER] No results above {self.similarity_threshold}, trying lower threshold")
            
#             results_lower = self.vector_search.similarity_search_with_score(
#                 query=query,
#                 k=self.max_results,
#                 pre_filter={"source": "knowledge_base"}
#             )
            
#             filtered_lower = [
#                 (doc, score) for doc, score in results_lower 
#                 if score >= 0.45
#             ]
            
#             if filtered_lower:
#                 logger.info(f"[LANGCHAIN_RETRIEVER] ✅ Found {len(filtered_lower)} documents with lower threshold")
#                 documents = [doc for doc, _ in filtered_lower]
#                 return documents
            
#             logger.warning("[LANGCHAIN_RETRIEVER] ❌ No results found even with lower threshold")
#             return []
            
#         except Exception as e:
#             logger.error(f"[LANGCHAIN_RETRIEVER] Error: {e}", exc_info=True)
#             return []
    
#     async def _aget_relevant_documents(
#         self,
#         query: str,
#         *,
#         run_manager: CallbackManagerForRetrieverRun = None
#     ) -> List[Document]:
#         """
#         Async version of _get_relevant_documents.
#         Falls back to sync for now (MongoDB doesn't support async natively).
#         """
#         return self._get_relevant_documents(query, run_manager=run_manager)