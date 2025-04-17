"""
Utilities for token counting and cost estimation related to embeddings.
"""

import tiktoken
import logging
from typing import List, Union, Tuple
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache the tokenizer loading
_tokenizer = None

def _get_tokenizer():
    """Lazy loads and returns the tiktoken tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        try:
            _tokenizer = tiktoken.get_encoding("cl100k_base")
            logger.info("Tokenizer 'cl100k_base' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load tokenizer 'cl100k_base': {e}", exc_info=True)
            raise
    return _tokenizer


def count_tokens(text: str) -> int:
    """
    Count tokens in a single text string using the cl100k_base tokenizer
    (commonly used by models like text-embedding-ada-002).

    Args:
        text: The text string to tokenize.

    Returns:
        The number of tokens in the text.
        Returns 0 if the input is not a string or is empty.
    """
    if not isinstance(text, str) or not text:
        return 0
    try:
        tokenizer = _get_tokenizer()
        return len(tokenizer.encode(text))
    except Exception as e:
        # Log the error but maybe return 0 or re-raise depending on desired strictness
        logger.error(f"Error tokenizing text: {e}", exc_info=True)
        # Depending on requirements, you might want to return 0 or raise an error
        # Returning 0 allows processing to continue but might skew stats.
        return 0


def count_tokens_in_list(texts: List[str]) -> Tuple[int, List[int]]:
    """
    Count tokens for a list of text strings efficiently.

    Args:
        texts: A list of text strings.

    Returns:
        A tuple containing:
        - total_tokens (int): The sum of tokens across all texts.
        - individual_token_counts (List[int]): A list with token counts for each text.
    """
    if not texts:
        return 0, []

    tokenizer = _get_tokenizer()
    try:
        # Use encode_batch for potential speedup if available and applicable
        # For simplicity, using encode repeatedly here.
        # Consider optimizing further if performance on large lists is critical.
        individual_counts = [len(tokenizer.encode(text)) if isinstance(text, str) else 0 for text in texts]
        total_tokens = sum(individual_counts)
        return total_tokens, individual_counts
    except Exception as e:
        logger.error(f"Error tokenizing list of texts: {e}", exc_info=True)
        # Decide error handling: return (0, []) or raise
        return 0, []


# --- Cost Calculation ---

# Define default pricing (consider moving to config if it changes often)
# Example price for text-embedding-ada-002 as of late 2023/early 2024
DEFAULT_EMBEDDING_PRICE_PER_1K_TOKENS_EUR = 0.0001 # Adjust based on actual Azure pricing


def calculate_embedding_cost(total_tokens: int, price_per_1k_tokens: float = DEFAULT_EMBEDDING_PRICE_PER_1K_TOKENS_EUR) -> float:
    """
    Calculate the estimated cost for embedding a given number of tokens.

    Args:
        total_tokens: The total number of tokens to be embedded.
        price_per_1k_tokens: The price in your currency (e.g., EUR) per 1000 tokens.
                           Defaults to a predefined value.

    Returns:
        The estimated cost.
    """
    if total_tokens < 0:
        logger.warning(f"Total tokens cannot be negative: {total_tokens}. Returning cost 0.")
        return 0.0
    if price_per_1k_tokens < 0:
        logger.warning(f"Price per 1k tokens is negative: {price_per_1k_tokens}. Using absolute value.")
        price_per_1k_tokens = abs(price_per_1k_tokens)

    return (total_tokens / 1000.0) * price_per_1k_tokens


# --- Combined Reporting Function ---

def get_token_stats(texts: List[str], price_per_1k: float = DEFAULT_EMBEDDING_PRICE_PER_1K_TOKENS_EUR) -> dict:
    """
    Calculates token counts, statistics, and estimated cost for a list of texts.

    Args:
        texts: List of text documents.
        price_per_1k: Cost per 1000 tokens for embedding.

    Returns:
        A dictionary containing statistics:
        {'total_documents', 'total_tokens', 'estimated_cost_eur',
         'average_tokens_per_doc', 'min_tokens', 'max_tokens', 'median_tokens'}
    """
    total_docs = len(texts)
    if total_docs == 0:
        return {
            'total_documents': 0,
            'total_tokens': 0,
            'estimated_cost_eur': 0.0,
            'average_tokens_per_doc': 0.0,
            'min_tokens': 0,
            'max_tokens': 0,
            'median_tokens': 0
        }

    total_tokens, individual_counts = count_tokens_in_list(texts)
    estimated_cost = calculate_embedding_cost(total_tokens, price_per_1k)

    stats = {
        'total_documents': total_docs,
        'total_tokens': total_tokens,
        'estimated_cost_eur': estimated_cost,
        'average_tokens_per_doc': np.mean(individual_counts) if individual_counts else 0,
        'min_tokens': np.min(individual_counts) if individual_counts else 0,
        'max_tokens': np.max(individual_counts) if individual_counts else 0,
        'median_tokens': int(np.median(individual_counts)) if individual_counts else 0 # Median as int
    }
    return stats


# Example usage:
# if __name__ == "__main__":
#     example_texts = [
#         "This is the first document.",
#         "This document is the second document.",
#         "And this is the third one.",
#         "Is this the first document?"
#     ]

#     # Test individual counting
#     for i, text in enumerate(example_texts):
#         tokens = count_tokens(text)
#         print(f"Document {i+1}: '{text}' -> {tokens} tokens")

#     # Test list counting
#     total, individuals = count_tokens_in_list(example_texts)
#     print(f"\nList total tokens: {total}")
#     print(f"List individual counts: {individuals}")

#     # Test cost calculation
#     cost = calculate_embedding_cost(total)
#     print(f"Estimated cost for {total} tokens: €{cost:.6f}")

#     # Test stats generation
#     stats = get_token_stats(example_texts)
#     print("\nToken Statistics:")
#     for key, value in stats.items():
#         print(f"- {key}: {value}")
