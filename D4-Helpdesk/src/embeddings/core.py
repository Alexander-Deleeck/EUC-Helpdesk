"""
Core logic for generating text embeddings using the configured Azure OpenAI client.
"""

import time
import logging
from typing import List, Dict, Optional
from openai import AzureOpenAI

# Assuming config is available one level up
from .. import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_embeddings_batch(
    texts: List[str],
    client: AzureOpenAI,
    embedding_model_name: str = config.AZURE_OPENAI_EMBEDDING_MODEL_NAME,
    batch_size: int = 16, # Adjust based on API limits and performance
    retry_attempts: int = 3,
    initial_retry_delay: float = 1.0
) -> Optional[List[List[float]]]:
    """
    Generates embeddings for a list of texts in batches using the provided Azure OpenAI client.

    Handles basic retries on failure.

    Args:
        texts: A list of text strings to embed.
        client: An initialized AzureOpenAI client instance.
        embedding_model_name: The deployment name or model name for embeddings.
                              Defaults to the value in config.
        batch_size: Number of texts to process in each API call.
        retry_attempts: Number of times to retry a failed batch.
        initial_retry_delay: Initial delay in seconds before retrying.

    Returns:
        A list of embedding vectors (each vector is a list of floats),
        or None if embedding generation fails completely after retries.
        The order matches the input text list.
    """
    if not texts:
        logger.info("No texts provided for embedding. Returning empty list.")
        return []

    if not embedding_model_name:
        logger.error("Embedding model name/deployment ID is not configured.")
        raise ValueError("Embedding model name/deployment ID must be provided.")

    all_embeddings = []
    total_texts = len(texts)
    logger.info(f"Starting embedding generation for {total_texts} texts using model '{embedding_model_name}'...")

    for i in range(0, total_texts, batch_size):
        batch_texts = texts[i : min(i + batch_size, total_texts)]
        batch_num = (i // batch_size) + 1
        total_batches = (total_texts + batch_size - 1) // batch_size
        logger.debug(f"Processing batch {batch_num}/{total_batches} (size: {len(batch_texts)})... ")

        current_attempt = 0
        while current_attempt < retry_attempts:
            try:
                response = client.embeddings.create(
                    input=batch_texts,
                    model=embedding_model_name
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                logger.debug(f"Batch {batch_num}/{total_batches} completed successfully.")
                break # Success, exit retry loop

            except Exception as e:
                current_attempt += 1
                logger.warning(
                    f"Error embedding batch {batch_num}/{total_batches} (Attempt {current_attempt}/{retry_attempts}): {e}"
                )
                if current_attempt >= retry_attempts:
                    logger.error(
                        f"Failed to embed batch {batch_num}/{total_batches} after {retry_attempts} attempts. Aborting."
                    )
                    return None # Indicate complete failure
                else:
                    delay = initial_retry_delay * (2 ** (current_attempt - 1)) # Exponential backoff
                    logger.info(f"Retrying batch {batch_num} in {delay:.2f} seconds...")
                    time.sleep(delay)

        # Optional delay between successful batches to respect rate limits if needed
        # time.sleep(0.1)

    if len(all_embeddings) == total_texts:
        logger.info(f"Successfully generated embeddings for all {total_texts} texts.")
        return all_embeddings
    else:
        # This case should ideally be caught by the retry logic returning None
        logger.error("Embedding generation failed for some texts.")
        return None


# --- Note on the original `generate_embeddings` function --- #
# The original function in utils_cleaning.py took a dictionary `to_embed = {id: text}`
# and modified it in place. The refactored approach (`generate_embeddings_batch`)
# takes a list of texts and returns a list of embeddings, decoupling it from the
# specific data structure (like dict or DataFrame).
# The calling script (`2_embed_data.py`) will be responsible for:
# 1. Extracting the list of texts from the DataFrame based on `config['text_column_for_embedding']`.
# 2. Calling `generate_embeddings_batch` with this list.
# 3. Associating the returned embeddings with the corresponding IDs (from `config['id_column']`)
#    and metadata before adding them to the vector store.


# Example usage (for testing within the module)
# if __name__ == "__main__":
#     from .client import get_azure_openai_client # Relative import for testing

#     example_docs = [
#         "This is document one for testing.",
#         "Here is the second document.",
#         "A third piece of text.",
#         "Finally, the fourth document."
#     ]

#     try:
#         print("Initializing Azure OpenAI client...")
#         test_client = get_azure_openai_client()

#         print("\nGenerating embeddings...")
#         embeddings = generate_embeddings_batch(example_docs, test_client, batch_size=2)

#         if embeddings:
#             print(f"\nSuccessfully generated {len(embeddings)} embeddings.")
#             for i, emb in enumerate(embeddings):
#                 print(f"Embedding {i+1} (dimension: {len(emb)}): {emb[:5]}..." ) # Print first 5 dims
#         else:
#             print("\nEmbedding generation failed.")

#     except ValueError as e:
#         print(f"Configuration Error: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}") 