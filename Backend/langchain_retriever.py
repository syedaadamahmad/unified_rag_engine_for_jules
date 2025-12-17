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

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, max_results: Optional[int] = None, **kwargs):
        """Initialize MongoDB vector search."""
        # Get MongoDB connection details
        mongo_uri = os.getenv("MONGO_DB_URI")
        db_name = os.getenv("DB_NAME")
        collection_name = "module_vectors"

        if not mongo_uri or not db_name:
            raise ValueError("[LANGCHAIN_RETRIEVER] MONGO_DB_URI or DB_NAME not set")

        # Initialize MongoDB client based on environment
        if os.getenv("PYTEST_RUNNING"):
            client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        else:
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
        """
        Retrieve relevant documents with a single, optimized database call.

        This method performs one similarity search and then filters the results
        in memory to prioritize higher-confidence documents, avoiding a second network call.
        """
        try:
            # Single database call with a lower threshold to get a candidate set
            results = self.vector_search.similarity_search_with_score(
                query=query,
                k=self.max_results,
                pre_filter={"source": "knowledge_base"}
            )

            # In-memory filtering for high-confidence results
            high_confidence_docs = [
                doc for doc, score in results
                if score >= self.similarity_threshold
            ]

            if high_confidence_docs:
                logger.info(f"[LANGCHAIN_RETRIEVER] ✅ Found {len(high_confidence_docs)} high-confidence documents")
                return high_confidence_docs

            # Fallback to lower-confidence results from the same call
            lower_confidence_docs = [
                doc for doc, score in results
                if score >= 0.45
            ]

            if lower_confidence_docs:
                logger.info(f"[LANGCHAIN_RETRIEVER] ✅ Found {len(lower_confidence_docs)} lower-confidence documents")
                return lower_confidence_docs

            logger.warning("[LANGCHAIN_RETRIEVER] ❌ No results found matching thresholds")
            return []

        except Exception as e:
            logger.error(f"[LANGCHAIN_RETRIEVER] Error: {e}", exc_info=True)
            return []

    async def _aget_relevant_documents(self, query: str, *, run_manager=None):
        return self._get_relevant_documents(query, run_manager=run_manager)
