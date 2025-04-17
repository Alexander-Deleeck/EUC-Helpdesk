"""
Workflow Script 4: Hyperparameter Optimization for Clustering

Loads data (embeddings) from the vector store, runs hyperparameter optimization
(Random Search or Bayesian TPE) for UMAP & HDBSCAN based on configuration,
and saves the results.

Usage:
  python scripts/4_optimize_clusters.py --dataset <dataset_name> [--method <random|bayesian>] [--evals <int>]

Example:
  python scripts/4_optimize_clusters.py --dataset helpdesk --method bayesian --evals 100
"""

import argparse
import logging
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import random
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure the src directory is in the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Import necessary functions from the src library
try:
    from src import config
    from src.vector_store import client as vs_client, ops as vs_ops
    from src.analysis import optimization as optim
    from src.utils import helpers
except ImportError as e:
    logger.error(f"Error importing src modules: {e}. Make sure the script is run from the project root or src is in PYTHONPATH.")
    sys.exit(1)

def main(args):
    """Main function to execute the hyperparameter optimization workflow."""
    dataset_name = args.dataset
    tuning_method = args.method
    max_evals_override = args.evals

    logger.info(f"--- Starting Hyperparameter Optimization for dataset: '{dataset_name}' ---")
    logger.info(f"Tuning method: {tuning_method}")

    # 1. Load Active Configuration
    try:
        active_config = config.get_active_config(dataset_name)
        logger.info(f"Loaded configuration: {active_config['name']}")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # 2. Setup Vector Store Client and Get Collection
    vs_path = active_config.get("vector_store_path")
    collection_name = active_config.get("vector_store_collection_name")
    if not vs_path or not collection_name:
        logger.error("Config missing 'vector_store_path' or 'vector_store_collection_name'.")
        sys.exit(1)

    try:
        chroma_client = vs_client.get_chroma_client(vs_path)
        collection = vs_client.get_or_create_collection(chroma_client, collection_name)
        if collection.count() == 0:
            logger.error(f"Vector store collection '{collection_name}' is empty. Cannot optimize.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to setup ChromaDB client or collection: {e}", exc_info=True)
        sys.exit(1)

    # 3. Load Embeddings from Vector Store
    logger.info("Loading embeddings from vector store...")
    # Only need embeddings for optimization
    data_df = vs_ops.get_all_data(collection, include=["embeddings"])

    if data_df is None or data_df.empty or 'embedding' not in data_df.columns:
        logger.error("Failed to load embeddings or no embeddings found.")
        sys.exit(1)

    embeddings = np.array(data_df['embedding'].tolist())
    logger.info(f"Loaded {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}.")

    # 4. Prepare Optimization Configuration
    # Use defaults from config.py, allow overrides from active_config and CLI args
    default_opt_params = config.DEFAULT_OPTIMIZATION_PARAMS.copy()
    dataset_opt_params = active_config.get('optimization_params', {})
    opt_params = {**default_opt_params, **dataset_opt_params}

    if max_evals_override is not None:
        opt_params['max_evals'] = max_evals_override
        logger.info(f"Overriding max_evals with CLI argument: {max_evals_override}")
    elif tuning_method == 'bayesian' and 'max_evals_bayesian' in opt_params:
         opt_params['max_evals'] = opt_params['max_evals_bayesian']
    elif tuning_method == 'random' and 'max_evals_random' in opt_params:
         opt_params['max_evals'] = opt_params['max_evals_random']
    # else use default max_evals if specific ones aren't set

    # --- Define Search Space --- #
    # This should ideally be part of the config, but defining here for now
    # Make sure to use hp.choice etc. if method is bayesian
    search_space = {}
    if tuning_method == 'bayesian':
        if not optim.HYPEROPT_AVAILABLE:
            logger.error("Hyperopt library is required for Bayesian search but not installed.")
            sys.exit(1)
        # Example Bayesian space (adjust ranges/choices as needed)
        search_space = {
            'n_neighbors': optim.hp.choice('n_neighbors', range(5, 51, 5)),
            'n_components': optim.hp.choice('n_components', range(3, 16, 2)),
            'min_cluster_size': optim.hp.choice('min_cluster_size', range(5, 31, 5))
            # Add other params like metrics if desired
            # 'umap_metric': optim.hp.choice('umap_metric', ['cosine', 'euclidean'])
        }
    elif tuning_method == 'random':
        # Example Random space (simple lists/ranges)
        search_space = {
            'n_neighbors': list(range(5, 51, 5)),
            'n_components': list(range(3, 16, 2)),
            'min_cluster_size': list(range(5, 31, 5))
            # 'umap_metric': ['cosine', 'euclidean']
        }
    else:
        logger.error(f"Invalid tuning_method: '{tuning_method}'. Choose 'bayesian' or 'random'.")
        sys.exit(1)
    logger.info(f"Using search space: {search_space}")

    # --- Define Score Configuration --- #
    # Get bounds from optimization_params in config
    score_config = {
        'prob_threshold': opt_params.get('score_prob_threshold', 0.05),
        'label_lower_bound': opt_params.get('score_label_lower_bound'),
        'label_upper_bound': opt_params.get('score_label_upper_bound')
    }

    # 5. Run Optimization
    results = optim.run_hyperparameter_search(
        embeddings=embeddings,
        search_space=search_space,
        optimization_params=opt_params, # Contains max_evals, random_state
        score_config=score_config,
        tuning_method=tuning_method
    )

    # 6. Save Results
    hyperopt_path = active_config.get("hyperopt_path")
    if not hyperopt_path:
        logger.error("Config key 'hyperopt_path' is missing. Cannot save optimization results.")
    elif results is not None:
        saved_path = helpers.save_hyperopt_results(
            results=results,
            tuning_method=tuning_method,
            num_evals=opt_params.get('max_evals', 0),
            output_dir=hyperopt_path,
            filename_prefix=f"{active_config['name']}_optimization"
        )
        if saved_path:
             logger.info(f"Optimization results saved to: {saved_path}")
        else:
             logger.error("Failed to save optimization results.")
    else:
        logger.warning("Optimization did not produce results to save.")

    logger.info(f"--- Hyperparameter Optimization for '{dataset_name}' Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hyperparameter optimization for UMAP and HDBSCAN.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset configuration to use."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="bayesian",
        choices=["random", "bayesian"],
        help="Optimization method to use: 'random' or 'bayesian'. Default: bayesian"
    )
    parser.add_argument(
        "--evals",
        type=int,
        default=None,
        help="Number of evaluations to run (overrides config default)."
    )

    args = parser.parse_args()
    main(args) 