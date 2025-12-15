from unittest.mock import MagicMock

def get_mock_bedrock_embeddings():
    """Return a mock BedrockEmbeddings object."""
    mock_embeddings = MagicMock()
    # Simulate the embed_query method to return a list of floats
    mock_embeddings.embed_query.return_value = [0.1] * 768
    return mock_embeddings
