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
import hdbscan

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
    from src.analysis import dimensionality_reduction as dr
    from src.analysis import clustering as cl
    from src.utils import helpers
    from src.utils.helpers import get_best_params_from_results
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

    # 3. Load Embeddings and IDs from Vector Store
    logger.info("Loading embeddings and IDs from vector store...")
    data_df = vs_ops.get_all_data(collection, include=["metadatas", "documents", "embeddings"])

    logger.info(f"\n\n{'-'*100}\n\nData DataFrame: {data_df.head()}\n\n{'-'*100}\n\n")
    logger.info(f"\n\nData DataFrame: {data_df.columns}\n\n{'-'*100}\n")
    
    if data_df is None or data_df.empty or 'embedding' not in data_df.columns or 'id' not in data_df.columns:
        logger.error("Failed to load embeddings/IDs or required columns missing.")
        sys.exit(1)

    embeddings = np.array(data_df['embedding'].tolist())
    data_ids = data_df['id'].tolist()
    logger.info(f"Loaded {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}.")
    logger.info(f"Loaded {len(data_ids)} corresponding data IDs.")

    # 4. Prepare Optimization Configuration
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

    # Get search space configuration
    search_space_config = opt_params.get('search_space', config.DEFAULT_SEARCH_SPACE)

    # --- Define Search Space --- #
    search_space = {}
    if tuning_method == 'bayesian':
        if not optim.HYPEROPT_AVAILABLE:
            logger.error("Hyperopt library is required for Bayesian search but not installed.")
            sys.exit(1)
        from hyperopt import hp
        # Convert configuration to hyperopt space
        for param_name, param_config in search_space_config.items():
            p_type = param_config.get('type', 'choice')
            p_range = param_config.get('range', [])

            if p_type == 'choice' and len(p_range) == 3:
                start, end, step = p_range
                options = list(range(start, end, step))
                search_space[param_name] = hp.choice(param_name, options)
            elif p_type == 'choice' and isinstance(p_range, list):
                search_space[param_name] = hp.choice(param_name, p_range)
            elif p_type == 'uniform' and len(p_range) == 2:
                low, high = p_range
                search_space[param_name] = hp.uniform(param_name, low, high)
            else:
                logger.warning(f"Unsupported or invalid search space config for '{param_name}'. Skipping.")

    elif tuning_method == 'random':
        # Convert configuration to lists for random.choice
        for param_name, param_config in search_space_config.items():
            p_type = param_config.get('type', 'choice')
            p_range = param_config.get('range', [])

            if (p_type == 'choice' or p_type == 'uniform') and len(p_range) == 3:
                start, end, step = p_range
                search_space[param_name] = list(range(start, end, step))
            elif (p_type == 'choice' or p_type == 'uniform') and len(p_range) == 2:
                start, end = p_range
                step = 1 if all(isinstance(x, int) for x in p_range) else (p_range[1] - p_range[0]) / 10
                if isinstance(step, int) and step > 0:
                    search_space[param_name] = list(range(start, end, step))
                else:
                    search_space[param_name] = list(np.linspace(start, end, num=10))
            elif p_type == 'choice' and isinstance(p_range, list):
                search_space[param_name] = p_range
            else:
                logger.warning(f"Unsupported or invalid search space config for random search: '{param_name}'. Skipping.")

    else:
        logger.error(f"Invalid tuning_method: '{tuning_method}'. Choose 'bayesian' or 'random'.")
        sys.exit(1)

    logger.info(f"Using search space configuration: {search_space_config}")
    logger.info(f"Constructed search space for '{tuning_method}': {search_space}")

    # --- Define Score Configuration --- #
    score_config = {
        'prob_threshold': opt_params.get('prob_threshold', 0.05),
        'label_lower_bound': opt_params.get('label_lower_bound', 10),
        'label_upper_bound': opt_params.get('label_upper_bound', 75)
    }
    opt_random_state = opt_params.get('random_state', int(time.time()))

    # 5. Run Optimization
    results = optim.run_hyperparameter_search(
        embeddings=embeddings,
        search_space=search_space,
        optimization_params=opt_params, # Contains max_evals, random_state
        score_config=score_config,
        tuning_method=tuning_method
    )

    # 6. Save Optimization Results
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

    # 7. Generate and Save Final Cluster Assignments
    cluster_results_path = active_config.get("cluster_results_path")
    if not cluster_results_path:
        logger.error("Config key 'cluster_results_path' is missing. Cannot save final cluster assignments.")
    elif results is not None:
        logger.info("Extracting best parameters to generate final cluster assignments...")
        best_params = get_best_params_from_results(results, tuning_method, search_space)

        if best_params:
            logger.info(f"Running final clustering with best parameters: {best_params}")

            # Separate UMAP and HDBSCAN parameters
            default_umap_params = config.DEFAULT_UMAP_PARAMS.copy()
            default_hdbscan_params = config.DEFAULT_HDBSCAN_PARAMS.copy()

            umap_p = {
                'n_neighbors': best_params['n_neighbors'],
                'n_components': best_params['n_components'],
                'metric': best_params.get('umap_metric', default_umap_params.get('metric', 'cosine')),
                'min_dist': best_params.get('umap_min_dist', default_umap_params.get('min_dist', 0.1))
            }
            hdbscan_p = {
                'min_cluster_size': best_params['min_cluster_size'],
                'metric': best_params.get('hdbscan_metric', default_hdbscan_params.get('metric', 'euclidean')),
                'cluster_selection_method': best_params.get('hdbscan_selection', default_hdbscan_params.get('cluster_selection_method', 'eom')),
                'min_samples': best_params.get('min_samples', default_hdbscan_params.get('min_samples')),
                'cluster_selection_epsilon': best_params.get('cluster_selection_epsilon', default_hdbscan_params.get('cluster_selection_epsilon', 0.0))
            }
            hdbscan_p = {k: v for k, v in hdbscan_p.items() if v is not None}

            umap_random_state = opt_random_state
            logger.info(f"Using UMAP random state: {umap_random_state}")

            # Re-run UMAP
            embeddings_reduced_final = dr.reduce_dimensions_umap(
                embeddings,
                umap_p,
                random_state=umap_random_state
            )

            if embeddings_reduced_final is not None:
                # Re-run HDBSCAN
                final_cluster_labels = cl.cluster_hdbscan(
                    embeddings_reduced_final,
                    hdbscan_p,
                    verbose=True
                )

                if final_cluster_labels is not None:
                    if len(final_cluster_labels) == len(data_ids):
                        # Create DataFrame for saving
                        assignments_df = pd.DataFrame({
                            'data_id': data_ids,
                            'cluster_label': final_cluster_labels
                        })

                        # Save the assignments
                        saved_assignments_path = helpers.save_cluster_assignments(
                            assignments_df=assignments_df,
                            output_dir=cluster_results_path,
                            filename_prefix=f"{active_config['name']}_assignments"
                        )
                        if saved_assignments_path:
                            logger.info(f"Final cluster assignments saved to: {saved_assignments_path}")
                        else:
                            logger.error("Failed to save final cluster assignments.")
                    else:
                        logger.error(f"Mismatch between number of labels ({len(final_cluster_labels)}) and data IDs ({len(data_ids)}). Cannot save assignments.")
                else:
                    logger.error("Final HDBSCAN clustering failed.")
            else:
                logger.error("Final UMAP reduction failed.")
        else:
            logger.error("Could not determine best parameters. Skipping final cluster assignment generation.")
    else:
        logger.warning("Optimization did not produce results. Skipping final cluster assignment generation.")

    logger.info(f"--- Hyperparameter Optimization Workflow for '{dataset_name}' Finished ---")


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