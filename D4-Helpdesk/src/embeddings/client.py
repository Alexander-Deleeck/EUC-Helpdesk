"""
Initializes and provides the Azure OpenAI client for embeddings.
"""

from openai import AzureOpenAI
import logging

# Import necessary configuration variables from the central config module
# Assumes config.py is in the parent directory (src/)
from .. import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_azure_openai_client() -> AzureOpenAI:
    """
    Initializes and returns an AzureOpenAI client using credentials
    and endpoints defined in the project's configuration.

    Raises:
        ValueError: If any required configuration variable (API key, endpoint,
                    deployment name, API version) is missing.

    Returns:
        An initialized AzureOpenAI client instance.
    """
    # Validate that necessary config variables are loaded
    required_vars = {
        "API Key": config.AZURE_OPENAI_API_KEY,
        "Endpoint": config.AZURE_OPENAI_ENDPOINT,
        "Embedding Deployment": config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        "API Version": config.OPENAI_API_VERSION
    }

    missing_vars = [name for name, value in required_vars.items() if not value]
    if missing_vars:
        error_msg = f"Missing required Azure OpenAI configuration variable(s) in .env or config: {', '.join(missing_vars)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        client = AzureOpenAI(
            api_key=config.AZURE_OPENAI_API_KEY,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            azure_deployment=config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT, # Use the specific embedding deployment
            api_version=config.OPENAI_API_VERSION
        )
        logger.info("Azure OpenAI client initialized successfully.")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Azure OpenAI client: {e}", exc_info=True)
        raise


# Example usage (for testing within the module)
# if __name__ == "__main__":
#     try:
#         client = get_azure_openai_client()
#         print("Successfully obtained Azure OpenAI client instance.")
#         # You could potentially make a test call here if needed, e.g., list models
#         # models = client.models.list()
#         # print("Available models:", list(models))
#     except ValueError as e:
#         print(f"Configuration Error: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}") 