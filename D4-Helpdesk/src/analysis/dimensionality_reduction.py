"""
Provides functions for dimensionality reduction, primarily using UMAP.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Union, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- UMAP Availability Check ---
try:
    import umap
    UMAP_AVAILABLE = True
    logger.info("UMAP library loaded successfully.")
except ImportError:
    UMAP_AVAILABLE = False
    logger.warning("UMAP library not found. Dimensionality reduction with UMAP will not be available.")
    umap = None # Set to None to avoid NameErrors later


def reduce_dimensions_umap(
    embeddings: Union[np.ndarray, List[List[float]]],
    umap_params: Dict[str, Any],
    random_state: Optional[int] = None
) -> Optional[np.ndarray]:
    """
    Reduces the dimensionality of embeddings using the UMAP algorithm.

    Args:
        embeddings: A numpy array or list of lists containing the high-dimensional embeddings.
        umap_params: A dictionary containing parameters for UMAP initialization
                     (e.g., {'n_neighbors': 15, 'n_components': 5, 'min_dist': 0.0, 'metric': 'cosine'}).
        random_state: Optional random state for reproducibility.

    Returns:
        A numpy array containing the reduced embeddings, or None if UMAP is not available
        or if an error occurs during reduction.
    """
    if not UMAP_AVAILABLE:
        logger.error("UMAP is not installed. Cannot perform dimensionality reduction.")
        return None

    if embeddings is None or len(embeddings) == 0:
        logger.warning("No embeddings provided for dimensionality reduction.")
        return None

    # Convert list of lists to numpy array if necessary
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)

    if not isinstance(embeddings, np.ndarray):
         logger.error(f"Embeddings must be a numpy array or list of lists, got {type(embeddings)}")
         return None

    logger.info(f"Starting UMAP dimensionality reduction on {embeddings.shape[0]} embeddings...")
    logger.info(f"UMAP parameters: {umap_params}")
    if random_state is not None:
        logger.info(f"Using random state: {random_state}")
        umap_params['random_state'] = random_state # Add/overwrite random state in params

    try:
        reducer = umap.UMAP(**umap_params)
        embeddings_reduced = reducer.fit_transform(embeddings)
        logger.info(f"UMAP reduction complete. Reduced shape: {embeddings_reduced.shape}")
        return embeddings_reduced

    except Exception as e:
        logger.error(f"Error during UMAP dimensionality reduction: {e}", exc_info=True)
        return None

# Example Usage (for testing)
# if __name__ == "__main__":
#     print("--- Testing Dimensionality Reduction --- ")
#     if not UMAP_AVAILABLE:
#         print("UMAP not available, skipping test.")
#     else:
#         # Create dummy high-dimensional data
#         dummy_embeddings = np.random.rand(100, 768) # Example: 100 embeddings, 768 dimensions

#         # Define UMAP parameters (similar to what might be in config)
#         test_umap_params = {
#             'n_neighbors': 10,
#             'n_components': 2, # Reduce to 2D for easy plotting
#             'min_dist': 0.1,
#             'metric': 'euclidean' # Or cosine, etc.
#         }
#         test_random_state = 42

#         print(f"Input shape: {dummy_embeddings.shape}")
#         print(f"UMAP Params: {test_umap_params}")
#         print(f"Random State: {test_random_state}")

#         reduced_embeddings = reduce_dimensions_umap(
#             dummy_embeddings,
#             test_umap_params,
#             test_random_state
#         )

#         if reduced_embeddings is not None:
#             print(f"\nReduced embeddings shape: {reduced_embeddings.shape}")
#             # print("Sample reduced data:")
#             # print(reduced_embeddings[:5])
#             print("\nUMAP reduction test successful.")
#         else:
#             print("\nUMAP reduction test failed.") 