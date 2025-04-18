"""
Provides functions for clustering embeddings, primarily using HDBSCAN.
Includes scoring functions for optimization.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Union, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- HDBSCAN Availability Check ---
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
    logger.info("HDBSCAN library loaded successfully.")
except ImportError:
    HDBSCAN_AVAILABLE = False
    logger.warning("HDBSCAN library not found. Clustering functionality will not be available.")
    hdbscan = None # Set to None to avoid NameErrors later


def cluster_hdbscan(
    embeddings: Union[np.ndarray, List[List[float]]],
    hdbscan_params: Dict[str, Any],
    verbose: bool = False # Added verbose option
) -> Optional[np.ndarray]:
    """
    Performs clustering on embeddings using the HDBSCAN algorithm.

    Args:
        embeddings: A numpy array or list of lists containing the embeddings
                    (can be original or dimensionally reduced).
        hdbscan_params: A dictionary containing parameters for HDBSCAN initialization
                        (e.g., {'min_cluster_size': 15, 'metric': 'euclidean', ...}).
        verbose: If True, prints more detailed output during clustering.

    Returns:
        A numpy array containing the cluster labels for each embedding (-1 for noise),
        or None if HDBSCAN is not available or an error occurs.
    """
    if not HDBSCAN_AVAILABLE:
        logger.error("HDBSCAN is not installed. Cannot perform clustering.")
        return None

    if embeddings is None or len(embeddings) == 0:
        logger.warning("No embeddings provided for clustering.")
        return None

    # Convert list of lists to numpy array if necessary
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)

    if not isinstance(embeddings, np.ndarray):
         logger.error(f"Embeddings must be a numpy array or list of lists, got {type(embeddings)}")
         return None

    logger.info(f"Starting HDBSCAN clustering on {embeddings.shape[0]} points...")
    logger.info(f"HDBSCAN parameters: {hdbscan_params}")

    try:
        # Ensure core_dist_n_jobs is explicitly set if multi-processing is desired
        # Set prediction_data=True if you need outlier scores or membership vectors later
        clusterer = hdbscan.HDBSCAN(**hdbscan_params, gen_min_span_tree=False, prediction_data=True, core_dist_n_jobs=-1)

        if verbose:
            print("Fitting HDBSCAN clusterer...") # Use print for direct verbose output

        cluster_labels = clusterer.fit_predict(embeddings)

        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = np.sum(cluster_labels == -1)

        logger.info(f"HDBSCAN clustering complete.")
        logger.info(f"  Number of clusters found: {n_clusters}")
        logger.info(f"  Number of noise points: {n_noise} ({n_noise / len(cluster_labels):.1%})")

        # Return the clusterer object as well if needed for probabilities, etc.
        # return cluster_labels, clusterer
        return cluster_labels # Just return labels for now

    except Exception as e:
        logger.error(f"Error during HDBSCAN clustering: {e}", exc_info=True)
        return None


def score_clusters(
    clusterer: hdbscan.HDBSCAN, # Accepts the fitted clusterer object
    prob_threshold: float = 0.05,
    label_lower_bound: Optional[int] = None,
    label_upper_bound: Optional[int] = None
) -> float:
    """
    Scores the clustering based on the proportion of noise/low-probability points
    and optionally penalizes if the cluster count is outside defined bounds.

    Lower score is better (consistent with minimization objective in hyperopt).

    Args:
        clusterer: The fitted HDBSCAN clusterer object (must have been fitted with
                   prediction_data=True to access probabilities).
        prob_threshold: Minimum probability for a point to be considered 'well clustered'.
        label_lower_bound: Optional minimum acceptable number of clusters (exclusive of noise).
        label_upper_bound: Optional maximum acceptable number of clusters (exclusive of noise).

    Returns:
        A float representing the clustering score (lower is better).
        Returns a high score (e.g., 1.0 or 2.0) if inputs are invalid or bounds are violated.
    """
    if not HDBSCAN_AVAILABLE:
         logger.error("Cannot score clusters, HDBSCAN is not available.")
         return 2.0 # Return high score

    if not isinstance(clusterer, hdbscan.HDBSCAN) or not hasattr(clusterer, 'labels_'):
         logger.error("Invalid HDBSCAN clusterer object provided for scoring.")
         return 2.0 # Return high score

    try:
        cluster_labels = clusterer.labels_
        label_count = len(np.unique(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        total_points = len(cluster_labels)

        if total_points == 0:
            logger.warning("Cannot score clustering with zero points.")
            return 2.0 # Return high score for empty input

        # Using clusterer.probabilities_ requires prediction_data=True during fit
        if hasattr(clusterer, 'probabilities_'):
            probabilities = clusterer.probabilities_
            well_clustered_points = np.count_nonzero(probabilities >= prob_threshold)
            effective_noise = total_points - well_clustered_points
            noise_fraction = float(effective_noise) / total_points
            logger.debug(f"Scoring based on probabilities >= {prob_threshold}. Noise fraction: {noise_fraction:.3f}")
        else:
            # Fallback if probabilities are not available (e.g., prediction_data=False)
            logger.warning("HDBSCAN probabilities not available. Scoring based on noise label (-1). Ensure prediction_data=True was used.")
            noise_points = np.count_nonzero(cluster_labels == -1)
            noise_fraction = float(noise_points) / total_points
            logger.debug(f"Scoring based on noise label count. Noise fraction: {noise_fraction:.3f}")

        score = noise_fraction

        # Penalize if the number of clusters is outside the desired range
        penalty = 0.0
        if label_lower_bound is not None and label_count < label_lower_bound:
            penalty = 1.0
            logger.debug(f"Adding penalty: Cluster count {label_count} < lower bound {label_lower_bound}")
        elif label_upper_bound is not None and label_count > label_upper_bound:
            penalty = 1.0
            logger.debug(f"Adding penalty: Cluster count {label_count} > upper bound {label_upper_bound}")
        
        return label_count, noise_fraction
        #final_score = score + penalty
        #logger.debug(f"Final score: {final_score:.4f} (Noise Fraction: {score:.4f}, Penalty: {penalty:.1f}, Labels: {label_count})")
        #return final_score

    except Exception as e:
        logger.error(f"Error scoring clusters: {e}", exc_info=True)
        return 2.0 # Return high score on error

# Example Usage (for testing)
# if __name__ == "__main__":
#     print("--- Testing Clustering --- ")
#     if not HDBSCAN_AVAILABLE:
#         print("HDBSCAN not available, skipping test.")
#     else:
#         # Create dummy embedding data (e.g., already reduced)
#         np.random.seed(42)
#         dummy_embeddings = np.random.rand(150, 5) # Example: 150 points, 5 dimensions
#         # Add some structure for potential clusters
#         dummy_embeddings[50:100, :] += 0.5
#         dummy_embeddings[100:150, :] -= 0.5

#         # Define HDBSCAN parameters
#         test_hdbscan_params = {
#             'min_cluster_size': 10,
#             'metric': 'euclidean',
#             'cluster_selection_method': 'eom' # Example
#             # 'allow_single_cluster': True # Example parameter
#         }

#         print(f"Input shape: {dummy_embeddings.shape}")
#         print(f"HDBSCAN Params: {test_hdbscan_params}")

#         cluster_labels = cluster_hdbscan(dummy_embeddings, test_hdbscan_params, verbose=True)

#         if cluster_labels is not None:
#             print(f"\nCluster labels generated (first 20): {cluster_labels[:20]}")
#             print(f"Unique labels: {np.unique(cluster_labels)}")
#             print("\nHDBSCAN clustering test successful.")

#             # --- Test Scoring (Requires fitting again to get clusterer object) ---
#             print("\n--- Testing Scoring --- ")
#             try:
#                 # Re-fit to get the clusterer object with prediction_data=True
#                 test_clusterer = hdbscan.HDBSCAN(**test_hdbscan_params, prediction_data=True).fit(dummy_embeddings)

#                 score_no_bounds = score_clusters(test_clusterer, prob_threshold=0.1)
#                 print(f"Score (no bounds, p_thresh=0.1): {score_no_bounds:.4f}")

#                 # Test with bounds (e.g., expecting 2-4 clusters)
#                 score_with_bounds = score_clusters(test_clusterer, prob_threshold=0.1, label_lower_bound=2, label_upper_bound=4)
#                 print(f"Score (bounds [2, 4], p_thresh=0.1): {score_with_bounds:.4f}")

#                 # Test with bounds that will likely fail
#                 score_failed_bounds = score_clusters(test_clusterer, prob_threshold=0.1, label_lower_bound=5, label_upper_bound=10)
#                 print(f"Score (bounds [5, 10], p_thresh=0.1): {score_failed_bounds:.4f}")

#                 print("\nScoring test successful.")
#             except Exception as score_e:
#                 print(f"\nError during scoring test: {score_e}")
#         else:
#             print("\nHDBSCAN clustering test failed.") 