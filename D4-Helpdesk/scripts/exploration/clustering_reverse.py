import os
import chromadb
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from dotenv import load_dotenv
import traceback

# --- Clustering ---
try:
    import hdbscan

    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("Warning: hdbscan library not found. Clustering functionality disabled.")
    print("Install it using: pip install hdbscan")

# --- Dimensionality Reduction ---
try:
    import umap  # Requires umap-learn

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Error: umap-learn library not found. Visualization requires UMAP.")
    print("Install it using: pip install umap-learn")
    exit()

# Load environment variables (if needed, e.g., for ChromaDB config)
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

# --- Configuration ---
HELPDESK_DIR = Path(__file__).resolve().parents[2]
EMBEDDINGS_DIR = HELPDESK_DIR / "helpdesk-data" / "helpdesk-embeddings"
DB_PATH = EMBEDDINGS_DIR / "chroma_summaries"
#COLLECTION_NAME = "helpdesk_complete_summaries_embeddings"
COLLECTION_NAME = "helpdesk_summaries_embeddings"

# --- UMAP Configuration ---
N_COMPONENTS = 3  # 3 for 3D, 2 for 2D
UMAP_N_NEIGHBORS = 50  # For example, try 5, 15, 30, 50
UMAP_MIN_DIST = 0.5  # Higher values produce more spread-out clusters
UMAP_METRIC = "cosine"  # Using cosine for text embeddings

REDUCER_PARAMS = {
    "n_neighbors": UMAP_N_NEIGHBORS,
    "min_dist": UMAP_MIN_DIST,
    "metric": UMAP_METRIC,
    "n_components": N_COMPONENTS,
    "random_state": 42,
}

# --- HDBSCAN Configuration ---
# (These values may require tuning for your reduced space)
HDBSCAN_MIN_CLUSTER_SIZE = 300
HDBSCAN_MIN_SAMPLES = 3
HDBSCAN_METRIC = "euclidean"

# --- Visualization Configuration ---
VISUALIZATION_FILTER_OUTLIERS = True
VISUALIZATION_OUTLIER_QUANTILE = 0.025

# --- Helper Functions ---


def load_data_from_chroma(db_path: Path, collection_name: str):
    if not db_path.exists():
        raise FileNotFoundError(f"ChromaDB path not found: {db_path}")

    print(f"Connecting to ChromaDB at: {db_path}")
    client = chromadb.PersistentClient(path=str(db_path))

    try:
        print(f"Getting collection: {collection_name}")
        collection = client.get_collection(name=collection_name)
        collection_count = collection.count()
        print(f"Collection '{collection_name}' found with {collection_count} items.")
    except Exception as e:
        print(f"Error getting collection '{collection_name}': {e}")
        try:
            print("Available collections:", client.list_collections())
        except Exception as list_e:
            print(f"Could not list collections: {list_e}")
        raise

    if collection_count == 0:
        print("Warning: Collection count is 0. Attempting to get data anyway...")

    print("Retrieving all embeddings and metadata (this might take a moment)...")
    try:
        results = collection.get(include=["embeddings", "metadatas", "documents"])
    except Exception as e:
        print("\n--- Error during collection.get() ---")
        print(f"Error: {e}")
        traceback.print_exc()
        print("---------------------------------------\n")
        raise

    print("--- Debugging post-collection.get() ---")
    print(f"Keys in results: {results.keys()}")

    ids = results.get("ids", [])
    embeddings_list = results.get("embeddings", [])
    metadatas_list = results.get("metadatas", [])
    documents_list = results.get("documents", [])

    print(f"Retrieved {len(ids)} IDs.")

    if not results or len(embeddings_list) == 0 or len(metadatas_list) == 0:
        raise ValueError("No data (embeddings or metadata) retrieved from ChromaDB.")

    if not (
        len(ids) == len(embeddings_list) == len(metadatas_list) == len(documents_list)
    ):
        min_len = min(
            len(ids), len(embeddings_list), len(metadatas_list), len(documents_list)
        )
        ids = ids[:min_len]
        embeddings_list = embeddings_list[:min_len]
        metadatas_list = metadatas_list[:min_len]
        documents_list = documents_list[:min_len]
        print(f"Truncated lists to minimum length: {min_len}")

    try:
        print("DEBUG: Converting embeddings list to NumPy array...")
        embeddings = np.array(embeddings_list)
        print(f"DEBUG: Embeddings array shape: {embeddings.shape}")
    except Exception as e:
        print("Error converting embeddings to NumPy array:")
        traceback.print_exc()
        raise

    try:
        print("DEBUG: Converting metadata list to DataFrame...")
        metadata_df = pd.DataFrame(metadatas_list)
        metadata_df["doc_id"] = ids
        metadata_df["document_text"] = documents_list
        print("DEBUG: Created metadata DataFrame with columns added.")
    except Exception as e:
        print("Error creating Pandas DataFrame from metadata:")
        traceback.print_exc()
        raise

    potential_color_keys = []
    print("\nIdentifying potential metadata keys for coloring...")
    for col in metadata_df.columns:
        if col in [
            "doc_id",
            "document_text",
            "Description",
            "Summary",
            "Summary Solution",
            "Summary in English",
        ]:
            continue
        try:
            col_dtype_kind = metadata_df[col].dtype.kind
            unique_count = metadata_df[col].nunique()
            if col_dtype_kind in ["O", "b"] or (
                col_dtype_kind in ["i", "f", "M"] and unique_count < 50
            ):
                if unique_count > 0:
                    print(f"  - Considering '{col}' (Unique: {unique_count})")
                    potential_color_keys.append(col)
        except Exception as e:
            print(f"  - Error processing column '{col}': {e}")

    print("\nPotential metadata keys for coloring:", potential_color_keys)
    return embeddings, metadata_df, potential_color_keys


def reduce_dimensionality(embeddings: np.ndarray, params=None):
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP is required for dimensionality reduction.")

    print(f"\nStarting UMAP dimensionality reduction...")
    print(f"Input embeddings shape: {embeddings.shape}")

    if params is None:
        params = REDUCER_PARAMS

    target_dims = params.get("n_components", N_COMPONENTS)
    print(f"Reducing to {target_dims} dimensions with params: {params}")

    if embeddings.shape[1] <= target_dims:
        print(
            "Warning: Number of features is less than or equal to target dimensions. Skipping reduction."
        )
        return (
            embeddings[:, :target_dims]
            if embeddings.shape[1] > target_dims
            else embeddings
        )

    reducer = umap.UMAP(**params)
    try:
        embeddings_float32 = embeddings.astype(np.float32)
        embeddings_reduced = reducer.fit_transform(embeddings_float32)
        print(f"UMAP reduction complete. Reduced shape: {embeddings_reduced.shape}")
    except Exception as e:
        print("Error during UMAP dimensionality reduction:")
        traceback.print_exc()
        raise

    return embeddings_reduced


def run_hdbscan_clustering(
    embeddings: np.ndarray,
    min_cluster_size: int,
    min_samples: int = None,
    metric: str = "euclidean",
) -> np.ndarray:
    if not HDBSCAN_AVAILABLE:
        print("HDBSCAN not available; skipping clustering.")
        return None

    print("\n--- Starting HDBSCAN clustering on UMAP-reduced data ---")
    print(
        f"Parameters: min_cluster_size={min_cluster_size}, min_samples={min_samples}, metric={metric}"
    )
    print(f"Reduced embeddings shape: {embeddings.shape}")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        core_dist_n_jobs=-1,
    )

    try:
        cluster_labels = clusterer.fit_predict(embeddings)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = np.sum(cluster_labels == -1)
        print(f"Clustering complete: {n_clusters} clusters, {n_noise} noise points.")
        print(
            f"Unique labels and counts: {np.unique(cluster_labels, return_counts=True)}"
        )
        return cluster_labels
    except Exception as e:
        print("Error during HDBSCAN clustering:")
        traceback.print_exc()
        return None


def visualize_embeddings_plotly(
    embeddings_reduced: np.ndarray,
    metadata_df: pd.DataFrame,
    color_by_key: str,
    filter_outliers: bool = False,
    outlier_quantile: float = 0.01,
):
    print("\n--- Preparing Plotly visualization ---")
    if color_by_key not in metadata_df.columns:
        if color_by_key == "cluster_label":
            print(f"Warning: 'cluster_label' not found. Falling back to another key.")
            available_keys = [
                k for k in metadata_df.columns if k not in ["doc_id", "document_text"]
            ]
            if available_keys:
                color_by_key = available_keys[0]
            else:
                raise ValueError("No fallback key available for coloring.")
        else:
            raise ValueError(f"Color key '{color_by_key}' not found in metadata.")

    n_components = embeddings_reduced.shape[1]
    print(f"Plotting {n_components}D data.")
    if n_components not in [2, 3]:
        raise ValueError("Reduced embeddings must be 2D or 3D.")

    if len(metadata_df) != len(embeddings_reduced):
        min_len = min(len(metadata_df), len(embeddings_reduced))
        metadata_df = metadata_df.iloc[:min_len].copy()
        embeddings_reduced = embeddings_reduced[:min_len]
        print(f"Data aligned to {min_len} points.")

    plot_df = pd.DataFrame()
    if n_components == 3:
        plot_df["x"], plot_df["y"], plot_df["z"] = (
            embeddings_reduced[:, 0],
            embeddings_reduced[:, 1],
            embeddings_reduced[:, 2],
        )
        plot_func = px.scatter_3d
        coord_args = dict(x="x", y="y", z="z")
        title = f"3D UMAP (n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST}) colored by '{color_by_key}'"
    else:
        plot_df["x"], plot_df["y"] = embeddings_reduced[:, 0], embeddings_reduced[:, 1]
        plot_func = px.scatter
        coord_args = dict(x="x", y="y")
        title = f"2D UMAP (n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST}) colored by '{color_by_key}'"

    metadata_to_add = metadata_df.reset_index(drop=True)
    plot_df = pd.concat([plot_df, metadata_to_add], axis=1)

    if filter_outliers and n_components >= 2:
        print(f"Filtering outliers based on quantile {outlier_quantile}...")
        filter_mask = pd.Series(True, index=plot_df.index)
        for col in ["x", "y"] + (["z"] if n_components == 3 else []):
            lower_bound = plot_df[col].quantile(outlier_quantile)
            upper_bound = plot_df[col].quantile(1 - outlier_quantile)
            filter_mask &= (plot_df[col] >= lower_bound) & (plot_df[col] <= upper_bound)
            print(f"  - '{col}' between {lower_bound:.2f} and {upper_bound:.2f}")
        plot_df = plot_df[filter_mask]
        print(f"Remaining points after filtering: {len(plot_df)}")
        if len(plot_df) == 0:
            print("Warning: All points filtered out; reverting to unfiltered data.")
            plot_df = pd.concat(
                [
                    pd.DataFrame(
                        embeddings_reduced,
                        columns=("x", "y", "z") if n_components == 3 else ("x", "y"),
                    ),
                    metadata_to_add,
                ],
                axis=1,
            )

    if color_by_key == "cluster_label":
        if plot_df[color_by_key].isnull().any():
            plot_df[color_by_key] = plot_df[color_by_key].fillna(-1)
        plot_df[color_by_key] = plot_df[color_by_key].astype(int).astype(str)
        plot_df[color_by_key] = plot_df[color_by_key].replace("-1", "Noise")
    elif plot_df[color_by_key].isnull().any():
        plot_df[color_by_key] = plot_df[color_by_key].astype(str).fillna("Unknown")

    hover_cols = [
        col for col in metadata_df.columns if col not in ["doc_id", "document_text"]
    ]
    if "document_text" in metadata_df.columns:
        plot_df["hover_text"] = plot_df["document_text"].str[:100] + "..."
        hover_cols.insert(0, "hover_text")
    if color_by_key not in hover_cols and color_by_key in plot_df.columns:
        hover_cols.append(color_by_key)

    color_map = {"Noise": "grey"} if "Noise" in plot_df[color_by_key].unique() else None

    fig = plot_func(
        plot_df,
        **coord_args,
        color=color_by_key,
        title=title,
        hover_data=hover_cols,
        color_discrete_map=color_map,
        color_discrete_sequence=px.colors.qualitative.Plotly if not color_map else None,
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=50), legend_title_text=color_by_key)
    marker_size = 3 if n_components == 3 else 5
    fig.update_traces(marker=dict(size=marker_size))
    print("Displaying plot...")
    fig.show()


# --- Main Execution ---
if __name__ == "__main__":
    try:
        # 1. Load Data
        embeddings, metadata_df, potential_keys = load_data_from_chroma(
            DB_PATH, COLLECTION_NAME
        )

        # 2. Dimensionality Reduction via UMAP
        embeddings_reduced = reduce_dimensionality(embeddings, params=REDUCER_PARAMS)

        # 3. Run HDBSCAN clustering on the UMAP-reduced embeddings
        cluster_labels = None
        if HDBSCAN_AVAILABLE:
            cluster_labels = run_hdbscan_clustering(
                embeddings_reduced,
                min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                min_samples=HDBSCAN_MIN_SAMPLES,
                metric=HDBSCAN_METRIC,
            )
            if cluster_labels is not None and len(cluster_labels) == len(metadata_df):
                metadata_df["cluster_label"] = cluster_labels
                print("Added 'cluster_label' to metadata.")
                if "cluster_label" in potential_keys:
                    potential_keys.remove("cluster_label")
                potential_keys.insert(0, "cluster_label")
            else:
                print("Warning: Clustering failed or data lengths mismatch.")

        # 4. Choose a metadata key for coloring
        print("\nAvailable metadata keys for coloring:")
        key_map = {str(i + 1): key for i, key in enumerate(potential_keys)}
        for i, key in enumerate(potential_keys):
            if key in metadata_df.columns:
                unique_count = metadata_df[key].nunique()
                print(f"  {i+1}. {key} (Unique: {unique_count})")
            elif key == "cluster_label":
                print(f"  {i+1}. {key} (Clustering data not available)")
            else:
                print(f"  {i+1}. {key} (Not found)")
        selected_key = None
        default_key = (
            "cluster_label"
            if "cluster_label" in metadata_df.columns
            else (potential_keys[0] if potential_keys else None)
        )
        while selected_key is None:
            try:
                prompt = f"Enter the number or name of the metadata key to color by (default='{default_key}'): "
                choice = input(prompt).strip()
                if not choice and default_key:
                    selected_key = default_key
                elif choice in key_map:
                    selected_key = key_map[choice]
                    if (
                        selected_key not in metadata_df.columns
                        and selected_key != "cluster_label"
                    ):
                        print(f"Error: Key '{selected_key}' not in DataFrame.")
                        selected_key = None
                elif choice in metadata_df.columns:
                    selected_key = choice
                else:
                    print(f"Invalid choice '{choice}'. Please try again.")
            except KeyboardInterrupt:
                print("\nOperation cancelled by user.")
                exit()
            except Exception as e:
                print(f"Error during input: {e}")
                exit()

        print(f"\nSelected coloring key: '{selected_key}'")
        if selected_key in metadata_df.columns:
            print(f"Unique values: {metadata_df[selected_key].nunique()}")

        # 5. Visualize embeddings with Plotly
        visualize_embeddings_plotly(
            embeddings_reduced,
            metadata_df,
            selected_key,
            filter_outliers=VISUALIZATION_FILTER_OUTLIERS,
            outlier_quantile=VISUALIZATION_OUTLIER_QUANTILE,
        )

        print("\nScript finished successfully.")
    except Exception as e:
        print("\n--- An unexpected error occurred ---")
        print(f"Error: {e}")
        traceback.print_exc()
