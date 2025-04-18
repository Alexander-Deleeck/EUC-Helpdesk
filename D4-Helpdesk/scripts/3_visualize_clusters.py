"""
Workflow Script 3: Apply Clustering and Visualize Results

Loads data (embeddings, IDs) from the vector store, applies dimensionality
reduction (UMAP) and clustering (HDBSCAN) using specified or default parameters,
saves the cluster assignments along with IDs and reduced coordinates, and
optionally generates an interactive visualization.

Usage:
  python scripts/3_visualize_clusters.py --dataset <dataset_name> [--umap_neighbors <int>] [--umap_components <int>] [--hdbscan_min_cluster_size <int>] [--no_save] [--no_plot]

Example:
  # Apply clustering with specific params, save results, and plot
  python scripts/3_visualize_clusters.py --dataset helpdesk --umap_neighbors 30 --hdbscan_min_cluster_size 20

  # Apply clustering with defaults, save results, skip plotting
  python scripts/3_visualize_clusters.py --dataset helpdesk --no_plot
"""

import argparse
import logging
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import time
import hdbscan
from datetime import datetime
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure the src directory is in the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Import necessary functions
try:
    from src import config
    from src.vector_store import client as vs_client, ops as vs_ops
    from src.analysis import dimensionality_reduction as dr, clustering as cl
    from src.visualization import plots
    from src.utils import helpers # Assuming a function to save dataframes exists
except ImportError as e:
    logger.error(f"Error importing src modules: {e}. Ensure the script is run from the project root or src is in PYTHONPATH.")
    sys.exit(1)

# *** NEW: Function to save cluster assignments ***
def save_cluster_assignments(df_assignments: pd.DataFrame, output_path: Path):
    """Saves the DataFrame with cluster assignments to a CSV file."""
    try:
        helpers.ensure_dir(output_path.parent) # Ensure directory exists
        df_assignments.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"Cluster assignments saved successfully to: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save cluster assignments to {output_path}: {e}", exc_info=True)
        return False
# ************************************************

def main(args):
    """Main function to execute the clustering and visualization workflow."""
    dataset_name = args.dataset
    logger.info(f"--- Starting Clustering & Visualization for dataset: '{dataset_name}' ---")

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
        # *** Get the correct embedding function if needed for collection access ***
        # embedding_function = vs_client.get_embedding_function() # Might be needed depending on Chroma version/setup
        # collection = chroma_client.get_collection(name=collection_name, embedding_function=embedding_function)
        collection = chroma_client.get_collection(name=collection_name) # Simpler if EF not needed for get
        collection_count = collection.count()
        if collection_count == 0:
            logger.error(f"Vector store collection '{collection_name}' is empty.")
            sys.exit(1)
        logger.info(f"Accessed collection '{collection_name}' with {collection_count} items.")
    except Exception as e:
        logger.error(f"Failed to setup ChromaDB client or collection: {e}", exc_info=True)
        sys.exit(1)

    # 3. Load Embeddings and IDs from Vector Store
    logger.info("Loading embeddings and IDs from vector store...")
    # Need IDs to map labels back to data points
    data_df = vs_ops.get_all_data(collection, include=["embeddings", "metadatas"]) # Get metadatas to extract ID

    if data_df is None or data_df.empty or 'embedding' not in data_df.columns or 'id' not in data_df.columns:
        logger.error("Failed to load embeddings/IDs or required columns missing.")
        sys.exit(1)

    embeddings = np.array(data_df['embedding'].tolist())
    doc_ids = data_df['id'].tolist() # Get the document IDs
    logger.info(f"Loaded {embeddings.shape[0]} embeddings and corresponding IDs.")

    # 4. Determine Clustering Parameters
    # Priority: CLI args > dataset_config > default_config
    umap_params = active_config.get('umap_params', config.DEFAULT_UMAP_PARAMS).copy()
    hdbscan_params = active_config.get('hdbscan_params', config.DEFAULT_HDBSCAN_PARAMS).copy()

    if args.umap_neighbors is not None: umap_params['n_neighbors'] = args.umap_neighbors
    if args.umap_components is not None: umap_params['n_components'] = args.umap_components
    if args.hdbscan_min_cluster_size is not None: hdbscan_params['min_cluster_size'] = args.hdbscan_min_cluster_size
    # Add CLI args for other params like min_dist if needed

    logger.info(f"Using UMAP parameters: {umap_params}")
    logger.info(f"Using HDBSCAN parameters: {hdbscan_params}")

    # --- Define random state ---
    # Use a fixed state for reproducibility or None for variability
    # Could also be sourced from config or CLI
    clustering_random_state = active_config.get("clustering_random_state", 42)
    logger.info(f"Using random state for UMAP: {clustering_random_state}")


    # 5. Perform Dimensionality Reduction (UMAP)
    if not dr.UMAP_AVAILABLE:
        logger.error("UMAP library not found. Cannot proceed.")
        sys.exit(1)
    embeddings_reduced = dr.reduce_dimensions_umap(embeddings, umap_params, random_state=clustering_random_state)
    if embeddings_reduced is None:
        logger.error("Dimensionality reduction failed.")
        sys.exit(1)

    # 6. Perform Clustering (HDBSCAN)
    if not cl.HDBSCAN_AVAILABLE:
        logger.error("HDBSCAN library not found. Cannot proceed.")
        sys.exit(1)

    # *** Fit HDBSCAN here to get the clusterer object for potentially saving probabilities ***
    try:
        hdbscan_clusterer = hdbscan.HDBSCAN(**hdbscan_params, prediction_data=True, core_dist_n_jobs=-1).fit(embeddings_reduced)
        cluster_labels = hdbscan_clusterer.labels_
        probabilities = hdbscan_clusterer.probabilities_ # Get probabilities
        n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        logger.info(f"Clustering applied successfully. Found {n_clusters_found} clusters.")
    except Exception as e:
        logger.error(f"HDBSCAN clustering failed: {e}", exc_info=True)
        sys.exit(1)
    # ************************************************************************************

    # 7. Prepare Results DataFrame
    logger.info("Preparing results DataFrame...")
    results_data = {
        'doc_id': doc_ids,
        'cluster_label': cluster_labels,
        'probability': probabilities # Include probabilities
    }
    # Add UMAP coordinates
    umap_dims = embeddings_reduced.shape[1]
    for i in range(umap_dims):
        results_data[f'umap_{chr(120+i)}'] = embeddings_reduced[:, i] # umap_x, umap_y, umap_z

    results_df = pd.DataFrame(results_data)

    # 8. Save Cluster Assignments *** (Conditional) ***
    if not args.no_save:
        # Construct output path from config
        processed_dir = Path(active_config.get("processed_path_template", "").format("dummy").parent) # Get dir from template
        if not processed_dir.name: # Basic check if path seems valid
             logger.error("Could not determine processed directory from 'processed_path_template' in config.")
        else:
            # Define filename (consider adding params or timestamp if needed)
            output_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{active_config['name']}_cluster_assignments.csv"
            output_path = processed_dir / output_filename
            save_cluster_assignments(results_df, output_path)
    else:
        logger.info("Skipping saving cluster assignments (--no_save specified).")

    # 9. Visualize Results *** (Conditional) ***
    if not args.no_plot:
        logger.info("Generating visualization...")
        if not plots.PLOTLY_AVAILABLE: # Check if plotting library is available
             logger.warning("Plotly library not found. Cannot generate plot.")
        else:
            # We need metadata for plotting tooltips/colors
            # Reload full data or use the metadata part from initial load
            # For simplicity, let's assume `data_df` from step 3 contains necessary metadata
            # Ensure 'id' column matches 'doc_id' in results_df for merging
            if 'metadata' in data_df.columns:
                 # Expand metadata dicts into columns - CAREFUL: might overwrite existing columns if names clash
                 metadata_df = pd.json_normalize(data_df['metadata'])
                 # Merge with results_df on the ID
                 plot_df = pd.merge(results_df, metadata_df, left_on='doc_id', right_on=active_config['id_column'], how='left')
                 # Determine color key - default to cluster_label or ask user/use config
                 color_key = 'cluster_label'
                 # Generate title
                 plot_title = f"{umap_dims}D UMAP of {active_config['name']} colored by {color_key}"
                 plots.visualize_embeddings_plotly(
                     embeddings_reduced=embeddings_reduced, # Pass reduced embeddings separately
                     plot_df=plot_df, # Pass the merged DataFrame with labels and metadata
                     color_by_key=color_key,
                     title=plot_title,
                     filter_outliers=True, # Or configure this
                     outlier_quantile=0.01 # Or configure this
                 )
            else:
                 logger.warning("Metadata not loaded or available in expected format for plotting.")

    else:
        logger.info("Skipping plot generation (--no_plot specified).")


    logger.info(f"--- Clustering & Visualization for '{dataset_name}' Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply UMAP & HDBSCAN clustering and visualize.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset configuration to use.")
    # Add args to override specific parameters if needed
    parser.add_argument("--umap_neighbors", type=int, help="Override UMAP n_neighbors.")
    parser.add_argument("--umap_components", type=int, help="Override UMAP n_components.")
    parser.add_argument("--hdbscan_min_cluster_size", type=int, help="Override HDBSCAN min_cluster_size.")
    parser.add_argument("--no_save", action="store_true", help="Do not save the cluster assignment results CSV.")
    parser.add_argument("--no_plot", action="store_true", help="Do not generate the interactive plot.")

    args = parser.parse_args()
    main(args)