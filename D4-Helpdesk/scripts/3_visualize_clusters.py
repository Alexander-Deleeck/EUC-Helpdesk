"""
Workflow Script 3: Cluster Visualization

Loads data (embeddings and metadata) from the vector store for a specified dataset,
performs dimensionality reduction (UMAP) and clustering (HDBSCAN) using configured
or default parameters, and generates an interactive visualization.

Usage:
  python scripts/3_visualize_clusters.py --dataset <dataset_name> [--umap_neighbors <int>] [--hdbscan_min_cluster <int>] [--color_by <col_name>]

Example:
  python scripts/3_visualize_clusters.py --dataset helpdesk --color_by cluster_label --umap_neighbors 20
"""

import argparse
import logging
from pathlib import Path
import sys
import pandas as pd
import numpy as np

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
    from src.analysis import dimensionality_reduction as dr, clustering
    from src.visualization import plots
    from src.utils import helpers # For saving the final DataFrame maybe
except ImportError as e:
    logger.error(f"Error importing src modules: {e}. Make sure the script is run from the project root or src is in PYTHONPATH.")
    sys.exit(1)

def main(args):
    """Main function to execute the cluster visualization workflow."""
    dataset_name = args.dataset
    logger.info(f"--- Starting Cluster Visualization Workflow for dataset: '{dataset_name}' ---")

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
            logger.error(f"Vector store collection '{collection_name}' is empty. Cannot visualize.")
            logger.error("Please run the embedding script (2_embed_data.py) first.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to setup ChromaDB client or collection: {e}", exc_info=True)
        sys.exit(1)

    # 3. Load Data from Vector Store
    logger.info("Loading data (embeddings, metadata, documents) from vector store...")
    # Include documents if you want them in hover data
    include_fields = ["embeddings", "metadatas", "documents"]
    data_df = vs_ops.get_all_data(collection, include=include_fields)

    if data_df is None or data_df.empty:
        logger.error("Failed to load data or no data found in the vector store.")
        sys.exit(1)

    logger.info(f"Loaded data shape: {data_df.shape}")
    logger.debug(f"Loaded columns: {data_df.columns.tolist()}")

    # Extract embeddings (ensure they are numpy arrays)
    if 'embedding' not in data_df.columns:
         logger.error("'embedding' column not found in loaded data.")
         sys.exit(1)
    embeddings = np.array(data_df['embedding'].tolist())
    logger.info(f"Extracted embeddings shape: {embeddings.shape}")

    # Prepare metadata DataFrame (data_df already contains metadata columns)
    metadata_df = data_df.drop(columns=['embedding']) # Keep ID as index


    # 4. Dimensionality Reduction (UMAP)
    # Get UMAP params from config, allowing overrides from CLI args
    umap_params = active_config.get('umap_params', config.DEFAULT_UMAP_PARAMS).copy()
    if args.umap_neighbors is not None:
        umap_params['n_neighbors'] = args.umap_neighbors
        logger.info(f"Overriding UMAP n_neighbors with CLI argument: {args.umap_neighbors}")
    # Set n_components specifically for visualization (e.g., 2 or 3)
    umap_params['n_components'] = args.dimensions
    logger.info(f"Using UMAP for {args.dimensions}D visualization.")

    if not dr.UMAP_AVAILABLE:
        logger.error("UMAP is required for visualization but is not installed.")
        sys.exit(1)

    embeddings_reduced = dr.reduce_dimensions_umap(
        embeddings,
        umap_params,
        random_state=active_config.get('optimization_params', {}).get('random_state', 42)
    )

    if embeddings_reduced is None:
        logger.error("UMAP dimensionality reduction failed.")
        sys.exit(1)

    # Add reduced dimensions to the metadata DataFrame for plotting
    for i in range(args.dimensions):
        metadata_df[f'UMAP_{i+1}'] = embeddings_reduced[:, i]

    # 5. Clustering (HDBSCAN) - Optional, mainly for coloring
    # Get HDBSCAN params from config, allowing overrides from CLI args
    hdbscan_params = active_config.get('hdbscan_params', config.DEFAULT_HDBSCAN_PARAMS).copy()
    if args.hdbscan_min_cluster is not None:
        hdbscan_params['min_cluster_size'] = args.hdbscan_min_cluster
        logger.info(f"Overriding HDBSCAN min_cluster_size with CLI argument: {args.hdbscan_min_cluster}")

    if not clustering.HDBSCAN_AVAILABLE:
        logger.warning("HDBSCAN not installed. Skipping clustering step.")
        metadata_df['cluster_label'] = "N/A"
        cluster_label_col = 'cluster_label' # Use a placeholder column
    else:
        logger.info("Running HDBSCAN clustering on reduced embeddings...")
        # Use the *reduced* embeddings for clustering in visualization script
        cluster_labels = clustering.cluster_hdbscan(embeddings_reduced, hdbscan_params)

        if cluster_labels is not None:
            metadata_df['cluster_label'] = cluster_labels
            metadata_df['cluster_label'] = metadata_df['cluster_label'].astype(str) # For discrete colors
            cluster_label_col = 'cluster_label'
        else:
            logger.warning("HDBSCAN clustering failed. Visualization will not be colored by cluster.")
            metadata_df['cluster_label'] = "Error"
            cluster_label_col = 'cluster_label' # Use placeholder column

    # 6. Generate Visualization
    # Determine color key: use CLI arg, fallback to cluster_label if available, else None
    color_key = args.color_by
    if color_key is None:
         color_key = cluster_label_col # Default to cluster label if available
    elif color_key not in metadata_df.columns:
         logger.warning(f"Requested color key '{color_key}' not found in metadata. Plot will not be colored.")
         color_key = None

    # Determine hover data keys (use a subset of available metadata)
    # Can be customized further via config or args if needed
    hover_keys = [col for col in active_config.get('metadata_columns_to_store', []) if col in metadata_df.columns]
    # Add document text and cluster label if available and not already included
    if 'document' in metadata_df.columns and 'document' not in hover_keys:
         hover_keys.append('document')
    if cluster_label_col in metadata_df.columns and cluster_label_col not in hover_keys:
        hover_keys.append(cluster_label_col)
    # Limit hover keys to maybe first 10 or a configured list?
    hover_keys = hover_keys[:15] # Limit hover data for clarity

    plot_title = f"{active_config['name']} Embeddings ({args.dimensions}D UMAP) - Colored by: {color_key if color_key else 'Default'}"

    # Prepare save path
    plot_filename = f"{active_config['name']}_umap_{args.dimensions}d_clusters.html"
    # Save plots to a dedicated 'plots' directory at the project root or data dir?
    save_dir = project_root / "plots"
    save_dir.mkdir(exist_ok=True)
    save_filepath = save_dir / plot_filename

    if not plots.PLOTLY_AVAILABLE:
        logger.error("Plotly is required for visualization but is not installed.")
        sys.exit(1)

    plots.plot_embeddings_interactive(
        embeddings_reduced=embeddings_reduced,
        metadata_df=metadata_df,
        plot_title=plot_title,
        color_by_key=color_key,
        hover_data_keys=hover_keys,
        dimensions=args.dimensions,
        show_figure=True, # Show plot interactively
        save_path=save_filepath
    )

    # Optionally save the dataframe with UMAP coords and labels
    # output_df_path = config.PROCESSED_DATA_DIR / f"{active_config['name']}_clustered_umap.csv"
    # helpers.save_dataframe(metadata_df, output_df_path, index=True)

    logger.info(f"--- Cluster Visualization Workflow for '{dataset_name}' Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize clustered embeddings using UMAP and HDBSCAN.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset configuration to use."
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=2,
        choices=[2, 3],
        help="Number of dimensions for UMAP reduction and plotting (2 or 3). Default: 2"
    )
    parser.add_argument(
        "--color_by",
        type=str,
        default=None,
        help="Metadata column name to color points by (e.g., cluster_label, Category). Default: cluster_label"
    )
    parser.add_argument(
        "--umap_neighbors",
        type=int,
        default=None,
        help="Override UMAP n_neighbors parameter from config."
    )
    parser.add_argument(
        "--hdbscan_min_cluster",
        type=int,
        default=None,
        help="Override HDBSCAN min_cluster_size parameter from config."
    )

    args = parser.parse_args()
    main(args) 