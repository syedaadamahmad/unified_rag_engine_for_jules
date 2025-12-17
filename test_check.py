import os
import pytest
from unittest.mock import patch, MagicMock

# Set mock environment variables before importing other modules
os.environ["USE_MOCK_EMBEDDINGS"] = "true"
os.environ["MONGO_DB_URI"] = "mongodb://localhost:27017/"
os.environ["DB_NAME"] = "test_db"

from Backend.mongodb_client import MongoDBClient

def test_mongodb_client_connection():
    """
    Test MongoDB client connection.
    This test is skipped if USE_MOCK_EMBEDDINGS is true.
    """
    if os.getenv("USE_MOCK_EMBEDDINGS") == "true":
        pytest.skip("Skipping database connection test in mock environment")

    # This part of the test will only run if USE_MOCK_EMBEDDINGS is not "true"
    client = MongoDBClient()
    assert client is not None
    assert client.client is not None
    # Test a simple command to ensure the connection is live
    client.client.admin.command('ping')

def test_mongodb_client_singleton():
    """Test that the MongoDB client is a singleton."""
    if os.getenv("USE_MOCK_EMBEDDINGS") == "true":
        pytest.skip("Skipping database connection test in mock environment")

    client1 = MongoDBClient()
    client2 = MongoDBClient()
    assert client1 is client2
