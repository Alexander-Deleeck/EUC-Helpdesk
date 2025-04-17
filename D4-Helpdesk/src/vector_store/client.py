"""
Handles the setup and connection to the ChromaDB vector store.

Provides functions to get a persistent client, initialize the embedding function,
and get or create a specific collection based on configuration.
"""

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Import configuration from the parent src directory
from .. import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_chroma_client(vector_store_path: Path) -> chromadb.PersistentClient:
    """
    Initializes and returns a persistent ChromaDB client for the specified path.

    Ensures the directory exists before initializing the client.

    Args:
        vector_store_path: The directory path for the persistent ChromaDB storage.
                           This should come from the active dataset config.

    Returns:
        An initialized chromadb.PersistentClient instance.

    Raises:
        IOError: If the directory cannot be created.
        Exception: For other ChromaDB client initialization errors.
    """
    try:
        logger.info(f"Ensuring vector store directory exists: {vector_store_path}")
        vector_store_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initializing ChromaDB persistent client at: {vector_store_path}")
        client = chromadb.PersistentClient(path=str(vector_store_path))
        logger.info("ChromaDB client initialized successfully.")
        return client
    except OSError as e:
        logger.error(f"Failed to create directory for ChromaDB: {vector_store_path} - {e}", exc_info=True)
        raise IOError(f"Could not create ChromaDB directory: {e}") from e
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB client at {vector_store_path}: {e}", exc_info=True)
        raise


def get_openai_embedding_function(api_key: str = config.AZURE_OPENAI_API_KEY,
                                    api_base: str = config.AZURE_OPENAI_ENDPOINT,
                                    api_version: str = config.OPENAI_API_VERSION,
                                    model_name: str = config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT) -> Optional[embedding_functions.OpenAIEmbeddingFunction]:
    """
    Initializes and returns the OpenAIEmbeddingFunction for ChromaDB using Azure credentials.

    Reads configuration directly from the imported config module.

    Args:
        api_key: Azure OpenAI API Key.
        api_base: Azure OpenAI Endpoint URL.
        api_version: Azure OpenAI API Version.
        model_name: Azure OpenAI Embedding Deployment Name.

    Returns:
        An initialized OpenAIEmbeddingFunction instance, or None if configuration is missing.

    Raises:
        ValueError: If any required configuration value is missing.
    """
    required_vars = {
        "API Key": api_key,
        "Endpoint": api_base,
        "API Version": api_version,
        "Embedding Deployment": model_name
    }
    missing_vars = [name for name, value in required_vars.items() if not value]
    if missing_vars:
        error_msg = f"Missing required Azure OpenAI configuration for embedding function: {', '.join(missing_vars)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        logger.info(f"Initializing OpenAIEmbeddingFunction with model: {model_name}")
        embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            api_base=api_base,
            api_type="azure",
            api_version=api_version,
            model_name=model_name, # Use the deployment name here for Azure
        )
        logger.info("OpenAIEmbeddingFunction initialized successfully.")
        return embedding_function
    except Exception as e:
        logger.error(f"Failed to initialize OpenAIEmbeddingFunction: {e}", exc_info=True)
        raise


def get_or_create_collection(
    client: chromadb.Client,
    collection_name: str,
    embedding_function: Optional[embedding_functions.EmbeddingFunction] = None,
    collection_metadata: Optional[Dict[str, Any]] = None
) -> chromadb.Collection:
    """
    Gets or creates a ChromaDB collection with the specified name and configuration.

    Args:
        client: An initialized ChromaDB client (e.g., PersistentClient).
        collection_name: The name for the collection (from dataset config).
        embedding_function: The embedding function to use for the collection.
                             If None, attempts to get the default OpenAI function.
        collection_metadata: Optional metadata to associate with the collection upon creation.

    Returns:
        The retrieved or newly created chromadb.Collection instance.

    Raises:
        ValueError: If collection_name is empty or embedding_function is not provided and cannot be defaulted.
        Exception: For errors during collection retrieval or creation.
    """
    if not collection_name:
        raise ValueError("Collection name cannot be empty.")

    if embedding_function is None:
        logger.info("Embedding function not provided, attempting to get default OpenAI function.")
        try:
            embedding_function = get_openai_embedding_function()
        except ValueError as e:
            logger.error(f"Failed to get default embedding function: {e}")
            raise ValueError("Embedding function is required but could not be defaulted.") from e

    try:
        logger.info(f"Getting or creating collection: '{collection_name}'")
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata=collection_metadata # Add metadata if provided
        )
        logger.info(f"Successfully obtained collection: '{collection.name}' (Count: {collection.count()})")
        return collection
    except Exception as e:
        logger.error(f"Failed to get or create collection '{collection_name}': {e}", exc_info=True)
        raise


# Example Usage (for testing within the module)
# if __name__ == "__main__":
#     # This assumes your config.py and .env are set up correctly
#     # And that you have a dataset config like 'helpdesk' defined
#     try:
#         print("--- Testing Vector Store Client --- ")

#         # 1. Get active dataset config (replace 'helpdesk' if needed)
#         active_config = config.get_active_config("helpdesk")
#         vector_store_path = active_config["vector_store_path"]
#         collection_name = active_config["vector_store_collection_name"]

#         print(f"Using Vector Store Path: {vector_store_path}")
#         print(f"Using Collection Name: {collection_name}")

#         # 2. Get Chroma Client
#         chroma_client = get_chroma_client(vector_store_path)
#         print(f"Chroma client obtained: {type(chroma_client)}")

#         # 3. Get Embedding Function (implicitly tests config loading)
#         emb_func = get_openai_embedding_function()
#         print(f"Embedding function obtained: {type(emb_func)}")

#         # 4. Get or Create Collection
#         collection_meta = {"dataset": active_config['name'], "created_by": "test_script"}
#         collection = get_or_create_collection(
#             client=chroma_client,
#             collection_name=collection_name,
#             embedding_function=emb_func,
#             collection_metadata=collection_meta
#         )
#         print(f"Collection obtained: {collection.name}")
#         print(f"Current item count: {collection.count()}")
#         print(f"Collection metadata: {collection.metadata}")

#         print("\n--- Vector Store Client tests passed successfully --- ")

#     except ValueError as e:
#         print(f"\nConfiguration or Value Error: {e}")
#     except IOError as e:
#         print(f"\nIO Error (usually directory creation): {e}")
#     except Exception as e:
#         print(f"\nAn unexpected error occurred: {e}") 