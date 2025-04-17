"""
Functions for hyperparameter optimization of UMAP and HDBSCAN.
Supports random search and Bayesian optimization using Hyperopt.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, Tuple
from functools import partial
import random
import time

# Import necessary components from other analysis modules
from .dimensionality_reduction import reduce_dimensions_umap, UMAP_AVAILABLE
from .clustering import cluster_hdbscan, score_clusters, HDBSCAN_AVAILABLE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Hyperopt Availability Check ---
try:
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
    HYPEROPT_AVAILABLE = True
    logger.info("Hyperopt library loaded successfully.")
except ImportError:
    HYPEROPT_AVAILABLE = False
    logger.warning("Hyperopt library not found. Bayesian optimization will not be available.")
    # Define dummy functions/classes if needed for type hinting or preventing NameErrors
    def fmin(*args, **kwargs):
        raise ImportError("hyperopt not installed")
    def tpe(*args, **kwargs):
        raise ImportError("hyperopt not installed")
    hp = None
    STATUS_OK = 'ok' # Define STATUS_OK even if hyperopt is not present
    class Trials:
         pass

# --- TQDM Availability Check ---
try:
    from tqdm import trange, tqdm
    TQDM_AVAILABLE = True
    logger.info("TQDM library loaded successfully.")
except ImportError:
    TQDM_AVAILABLE = False
    logger.warning("TQDM library not found. Progress bars will be disabled.")
    # Define dummy functions if tqdm is not available
    def trange(x, *args, **kwargs):
        return range(x)
    def tqdm(x, *args, **kwargs):
        return x


def _generate_clusters_for_opt(
    embeddings: np.ndarray,
    umap_params: Dict[str, Any],
    hdbscan_params: Dict[str, Any],
    random_state: Optional[int] = None
) -> Optional[Tuple[np.ndarray, hdbscan.HDBSCAN]]:
    """
    Internal helper function to perform UMAP reduction and HDBSCAN clustering sequentially.
    Used specifically within the optimization loop.

    Args:
        embeddings: Original high-dimensional embeddings.
        umap_params: Dictionary of parameters for UMAP.
        hdbscan_params: Dictionary of parameters for HDBSCAN.
        random_state: Optional random state for UMAP reproducibility.

    Returns:
        A tuple (cluster_labels, clusterer_object) if successful, otherwise None.
        Returns None if UMAP or HDBSCAN are unavailable or if an error occurs.
    """
    if not UMAP_AVAILABLE or not HDBSCAN_AVAILABLE:
        logger.error("UMAP or HDBSCAN not available, cannot generate clusters for optimization.")
        return None

    # 1. Dimensionality Reduction (using the function from the other module)
    logger.debug(f"Running UMAP with params: {umap_params}, random_state: {random_state}")
    embeddings_reduced = reduce_dimensions_umap(embeddings, umap_params, random_state)

    if embeddings_reduced is None:
        logger.warning("UMAP reduction failed, cannot proceed with clustering.")
        return None

    # 2. Clustering (using the function from the other module)
    # We need the clusterer object back for scoring based on probabilities
    logger.debug(f"Running HDBSCAN with params: {hdbscan_params}")
    try:
        # Fit HDBSCAN directly here to get the clusterer object
        # Ensure prediction_data=True for probability scoring
        hdbscan_params_with_pred = hdbscan_params.copy()
        hdbscan_params_with_pred['prediction_data'] = True
        hdbscan_params_with_pred['core_dist_n_jobs'] = -1 # Use all cores

        clusterer = hdbscan.HDBSCAN(**hdbscan_params_with_pred).fit(embeddings_reduced)
        cluster_labels = clusterer.labels_
        logger.debug("HDBSCAN fitting completed.")
        return cluster_labels, clusterer
    except Exception as e:
        logger.error(f"Error during HDBSCAN clustering step in optimization: {e}", exc_info=True)
        return None


def _objective_function(
    params: Dict[str, Any],
    embeddings: np.ndarray,
    score_config: Dict[str, Any],
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Objective function for hyperparameter optimization (minimization).

    Performs UMAP + HDBSCAN with given parameters and returns a score.

    Args:
        params: Dictionary containing the hyperparameters to test
                (e.g., {'n_neighbors', 'n_components', 'min_cluster_size'}).
        embeddings: The high-dimensional embeddings.
        score_config: Dictionary with parameters for the scoring function
                      (e.g., {'prob_threshold', 'label_lower_bound', 'label_upper_bound'}).
        random_state: Fixed random state for UMAP reproducibility during optimization.

    Returns:
        A dictionary compatible with Hyperopt:
        {'loss': score, 'status': STATUS_OK, 'params': params, 'label_count': count}
    """
    start_time = time.time()

    # Separate UMAP and HDBSCAN parameters
    umap_p = {
        'n_neighbors': params['n_neighbors'],
        'n_components': params['n_components'],
        'metric': params.get('umap_metric', 'cosine'), # Allow overriding metric
        'min_dist': params.get('umap_min_dist', 0.0)   # Allow overriding min_dist
    }
    hdbscan_p = {
        'min_cluster_size': params['min_cluster_size'],
        'metric': params.get('hdbscan_metric', 'euclidean'), # Allow overriding metric
        'cluster_selection_method': params.get('hdbscan_selection', 'eom')
        # Add other HDBSCAN params from space if needed
    }

    clustering_result = _generate_clusters_for_opt(embeddings, umap_p, hdbscan_p, random_state)

    if clustering_result is None:
        # Handle failure: return a high loss
        logger.warning(f"Clustering failed for params: {params}. Assigning high loss.")
        return {'loss': 2.0, 'status': 'fail', 'params': params, 'label_count': -1, 'eval_time': time.time() - start_time}

    cluster_labels, clusterer = clustering_result

    # Score the clustering result
    score = score_clusters(clusterer,
                             prob_threshold=score_config.get('prob_threshold', 0.05),
                             label_lower_bound=score_config.get('label_lower_bound'),
                             label_upper_bound=score_config.get('label_upper_bound'))

    label_count = len(np.unique(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    eval_time = time.time() - start_time

    logger.debug(f"Objective Eval | Params: {params} | Score: {score:.4f} | Labels: {label_count} | Time: {eval_time:.2f}s")

    return {
        'loss': score,
        'status': STATUS_OK,
        'params': params, # Store the actual params used
        'label_count': label_count, # Store label count for analysis
        'eval_time': eval_time # Store evaluation time
    }


def run_hyperparameter_search(
    embeddings: np.ndarray,
    search_space: Dict[str, Any], # Can be simple dict for random, or hyperopt space
    optimization_params: Dict[str, Any],
    score_config: Dict[str, Any],
    tuning_method: str = 'bayesian' # 'bayesian' or 'random'
) -> Union[Trials, pd.DataFrame, None]:
    """
    Runs hyperparameter search using either Random Search or Bayesian Optimization (TPE).

    Args:
        embeddings: High-dimensional embeddings (numpy array).
        search_space: Dictionary defining the hyperparameter search space.
                      For Bayesian, values should be hyperopt distribution objects (e.g., hp.choice).
                      For Random, values can be lists or ranges to sample from.
        optimization_params: Dictionary with optimization settings
                             (e.g., {'max_evals': 100, 'random_state': 42}).
        score_config: Dictionary with parameters for the scoring function
                      (e.g., {'prob_threshold', 'label_lower_bound', 'label_upper_bound'}).
        tuning_method: 'bayesian' or 'random'.

    Returns:
        - If 'bayesian': A hyperopt Trials object containing search results.
        - If 'random': A pandas DataFrame containing search results.
        - None if prerequisites are not met or an error occurs.
    """
    max_evals = optimization_params.get('max_evals', 50)
    random_state = optimization_params.get('random_state', int(time.time())) # Use time if not set
    np.random.seed(random_state) # Seed numpy for reproducibility if needed elsewhere
    random.seed(random_state)    # Seed python random

    logger.info(f"Starting hyperparameter search using method: '{tuning_method}'")
    logger.info(f"Max evaluations: {max_evals}")
    logger.info(f"Optimization Random State: {random_state}")
    logger.info(f"Scoring Config: {score_config}")

    # --- Prepare objective function with fixed arguments --- #
    objective_fn_partial = partial(
        _objective_function,
        embeddings=embeddings,
        score_config=score_config,
        random_state=random_state # Fix UMAP random state for comparable runs
    )

    if tuning_method == 'bayesian':
        if not HYPEROPT_AVAILABLE:
            logger.error("Cannot run Bayesian search: hyperopt library is not installed.")
            return None

        trials = Trials()
        try:
            logger.info("Running Bayesian optimization (TPE)...")
            # Ensure search_space uses hp objects (e.g., hp.choice, hp.uniform)
            best = fmin(
                fn=objective_fn_partial,
                space=search_space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials,
                rstate=np.random.default_rng(random_state), # Seed hyperopt's internal rng
                show_progressbar=TQDM_AVAILABLE # Show progress if tqdm available
            )
            logger.info("Bayesian optimization finished.")
            # `best` contains the best *indices* if hp.choice was used.
            # The `trials` object holds the detailed results for each evaluation.
            best_params_evaluated = space_eval(search_space, best)
            logger.info(f"Best parameters found (evaluated): {best_params_evaluated}")
            logger.info(f"Best score found: {min(trials.losses()) if trials.losses() else 'N/A'}")
            return trials # Return the full Trials object for saving

        except Exception as e:
            logger.error(f"Error during Bayesian optimization: {e}", exc_info=True)
            return trials # Return trials object even if error occurred mid-way

    elif tuning_method == 'random':
        logger.info("Running Random Search...")
        results_list = []

        # Use tqdm for progress bar if available
        eval_range = trange(max_evals, desc="Random Search Evaluations") if TQDM_AVAILABLE else range(max_evals)

        for i in eval_range:
            # Sample parameters randomly from the search space definition
            params = {}
            for key, value_options in search_space.items():
                # Simple sampling assuming lists/ranges - adapt if more complex spaces used
                if isinstance(value_options, (list, range)):
                    params[key] = random.choice(value_options)
                # Add elif for other types if needed (e.g., uniform range)
                else:
                    logger.warning(f"Unsupported space type for random sampling: {type(value_options)} for key '{key}'. Using first option if possible.")
                    try: params[key] = value_options[0]
                    except: params[key] = None # Fallback

            logger.debug(f"Random Search Eval {i+1}/{max_evals} | Testing params: {params}")
            result = objective_fn_partial(params)

            # Store results including parameters and score
            results_list.append({
                'run_id': i + 1,
                'n_neighbors': params.get('n_neighbors'),
                'n_components': params.get('n_components'),
                'min_cluster_size': params.get('min_cluster_size'),
                # Add other params as needed
                'label_count': result.get('label_count'),
                'loss': result.get('loss'),
                'status': result.get('status'),
                'eval_time': result.get('eval_time')
            })

        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values(by="loss", ascending=True)
        logger.info("Random search finished.")
        if not results_df.empty:
             best_run = results_df.iloc[0]
             logger.info(f"Best parameters found (run {best_run['run_id']}):")
             logger.info(f"  n_neighbors: {best_run['n_neighbors']}, n_components: {best_run['n_components']}, min_cluster_size: {best_run['min_cluster_size']}")
             logger.info(f"  Best score: {best_run['loss']:.4f}")
        return results_df
    else:
        logger.error(f"Invalid tuning_method: '{tuning_method}'. Choose 'bayesian' or 'random'.")
        return None


# Example Usage (for testing)
# if __name__ == "__main__":
#     print("--- Testing Optimization --- ")
#     if not UMAP_AVAILABLE or not HDBSCAN_AVAILABLE:
#         print("UMAP or HDBSCAN not available, skipping optimization test.")
#     else:
#         # Create dummy high-dimensional data
#         np.random.seed(42)
#         dummy_embeddings = np.random.rand(300, 64) # 300 points, 64 dimensions

#         # Define Search Space (using simple lists for random, could use hp for bayesian)
#         test_search_space = {
#             'n_neighbors': [5, 10, 15], # hp.choice('nn', [5, 10, 15]) for hyperopt
#             'n_components': [3, 5],     # hp.choice('nc', [3, 5])
#             'min_cluster_size': [5, 10] # hp.choice('mcs', [5, 10])
#         }
#         if HYPEROPT_AVAILABLE:
#              test_search_space_bayes = {
#                  'n_neighbors': hp.choice('nn', [5, 10, 15]),
#                  'n_components': hp.choice('nc', [3, 5]),
#                  'min_cluster_size': hp.choice('mcs', [5, 10])
#              }
#         else:
#              test_search_space_bayes = None

#         # Define Optimization Params
#         test_opt_params = {
#             'max_evals': 5, # Keep low for testing
#             'random_state': 42
#         }

#         # Define Scoring Config (e.g., aiming for 2-4 clusters)
#         test_score_config = {
#             'prob_threshold': 0.05,
#             'label_lower_bound': 2,
#             'label_upper_bound': 4
#         }

#         print("\n--- Testing Random Search --- ")
#         random_results = run_hyperparameter_search(
#             dummy_embeddings,
#             test_search_space,
#             test_opt_params,
#             test_score_config,
#             tuning_method='random'
#         )
#         if random_results is not None:
#             print("Random Search Results (DataFrame):")
#             print(random_results.head())
#         else:
#             print("Random Search failed or returned None.")

#         if HYPEROPT_AVAILABLE and test_search_space_bayes:
#             print("\n--- Testing Bayesian Search --- ")
#             bayes_results = run_hyperparameter_search(
#                 dummy_embeddings,
#                 test_search_space_bayes,
#                 test_opt_params,
#                 test_score_config,
#                 tuning_method='bayesian'
#             )
#             if bayes_results is not None:
#                 print("Bayesian Search Results (Trials object - showing best loss):")
#                 try:
#                     print(f"Best loss found: {bayes_results.best_trial['result']['loss']:.4f}")
#                     print(f"Best params: {bayes_results.best_trial['result']['params']}")
#                 except Exception as e:
#                     print(f"Could not extract best trial info: {e}")
#                 # For full results, you'd iterate through bayes_results.trials
#             else:
#                 print("Bayesian Search failed or returned None.")
#         else:
#             print("\nSkipping Bayesian Search test (hyperopt not installed or space not defined).")

#     print("\n--- Optimization tests finished --- ") 