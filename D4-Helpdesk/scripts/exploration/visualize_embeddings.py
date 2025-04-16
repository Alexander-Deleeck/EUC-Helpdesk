# D4-Helpdesk\scripts\visualization\visualize_embeddings.py

import os
import chromadb
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from dotenv import load_dotenv
import traceback  # Import traceback for detailed error printing

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
    import umap  # Make sure umap-learn is installed
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Error: umap-learn library not found. Visualization requires UMAP.")
    print("Install it using: pip install umap-learn")
    exit()  # UMAP is essential for this script

# Load environment variables (optional, but good practice if Chroma needs config)
load_dotenv(
    dotenv_path=Path(__file__).resolve().parents[2] / ".env"
)  # Adjust path to your .env if needed

# --- Configuration ---
HELPDESK_DIR = Path(__file__).resolve().parents[2]
EMBEDDINGS_DIR = HELPDESK_DIR / "helpdesk-data" / "helpdesk-embeddings"
DB_PATH = EMBEDDINGS_DIR / "chroma_summaries"
COLLECTION_NAME = "helpdesk_complete_summaries_embeddings"

# --- UMAP Configuration ---
N_COMPONENTS = 3  # 3 for 3D, 2 for 2D
UMAP_N_NEIGHBORS = 50  # Try varying (e.g., 5, 15, 30, 50)
UMAP_MIN_DIST = (
    0.5  # Try varying (e.g., 0.0, 0.1, 0.25, 0.5) - HIGHER = MORE SPREAD OUT
)
UMAP_METRIC = "cosine"  # Use 'cosine' for text embeddings!

REDUCER_PARAMS = {
    "n_neighbors": UMAP_N_NEIGHBORS,
    "min_dist": UMAP_MIN_DIST,
    "metric": UMAP_METRIC,
    "n_components": N_COMPONENTS,
    "random_state": 42,
}

# --- HDBSCAN Configuration ---
HDBSCAN_MIN_CLUSTER_SIZE = 300  # Minimum size for a group to be considered a cluster
HDBSCAN_MIN_SAMPLES = 3 #None     # How conservative to be (None lets HDBSCAN choose, or try e.g., 5, 10)
HDBSCAN_METRIC = 'euclidean'   # HDBSCAN often works well with Euclidean distance on the *original* embeddings

# --- Visualization Configuration ---
VISUALIZATION_FILTER_OUTLIERS = True  # Set to True to filter plot points far from center
VISUALIZATION_OUTLIER_QUANTILE = 0.025  # Filter points outside this quantile (e.g., 0.01 = keep central 98%)

# --- Helper Functions ---


def load_data_from_chroma(db_path: Path, collection_name: str):
    """Loads embeddings and metadata from the ChromaDB collection."""
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
        # Allow proceeding but expect potential errors or empty results

    print("Retrieving all embeddings and metadata (this might take a moment)...")
    try:
        # Retrieve all data points. include embeddings and metadatas
        # As per your comment, Chroma implicitly returns 'ids'.
        # We explicitly ask for embeddings and metadatas.
        results = collection.get(
            include=[
                "embeddings",
                "metadatas",
                "documents",
            ]  # Also get documents for hover info potentially
        )
    except Exception as e:
        print(f"\n--- Error during collection.get() ---")
        print(f"Error: {e}")
        traceback.print_exc()
        print("---------------------------------------\n")
        raise

    print("--- Debugging post-collection.get() ---")
    print(f"Type of results: {type(results)}")
    print(f"Keys in results: {results.keys()}")

    # Check existence and type of expected keys
    for key in ["ids", "embeddings", "metadatas", "documents"]:
        if key in results:
            print(f"Key '{key}': Found, Type: {type(results[key])}")
            if isinstance(results[key], list):
                print(f"  Length of results['{key}']: {len(results[key])}")
                if len(results[key]) > 0:
                    print(f"  Type of first element: {type(results[key][0])}")
                    if key == "embeddings" and isinstance(
                        results[key][0], (list, np.ndarray)
                    ):
                        try:
                            print(
                                f"    Length/Shape of first embedding: {len(results[key][0])}"
                            )
                        except TypeError:
                            print(
                                f"    First embedding is not list/array like: {results[key][0]}"
                            )

            elif isinstance(results[key], np.ndarray):
                print(f"  Shape of results['{key}']: {results[key].shape}")
            else:
                print(f"  Value: {results[key]}")  # Print if not list/array
        else:
            print(f"Key '{key}': Not found in results!")
    print("--- End Debugging post-collection.get() ---")

    # **Revised Check:** Use len() for checking if lists are empty
    # Use .get() with default empty list [] to avoid KeyError if a key is missing
    ids = results.get("ids", [])
    embeddings_list = results.get("embeddings", [])
    metadatas_list = results.get("metadatas", [])
    documents_list = results.get("documents", [])  # Added documents retrieval

    print(f"Retrieved {len(ids)} IDs.")  # Use the length of the retrieved ids list

    if not results or len(embeddings_list) == 0 or len(metadatas_list) == 0:
        print("\n--- ERROR DETECTED: Empty Results ---")
        print(f"Results dict empty: {not results}")
        print(f"Embeddings list empty: {len(embeddings_list) == 0}")
        print(f"Metadatas list empty: {len(metadatas_list) == 0}")
        print("--------------------------------------\n")
        raise ValueError(
            "No data (embeddings or metadata) retrieved from ChromaDB. Ensure the collection is populated correctly."
        )
    print("DEBUG: Passed check for empty results.")

    # Check for length mismatch (should not happen with collection.get())
    if not (
        len(ids) == len(embeddings_list) == len(metadatas_list) == len(documents_list)
    ):
        print("\n--- WARNING: Length Mismatch ---")
        print(
            f"Length mismatch between retrieved components: "
            f"IDs={len(ids)}, Embeddings={len(embeddings_list)}, "
            f"Metadata={len(metadatas_list)}, Documents={len(documents_list)}"
        )
        print("Attempting to proceed, but this might indicate an issue.")
        # Optional: Add logic here to truncate to the minimum length or raise error
        min_len = min(
            len(ids), len(embeddings_list), len(metadatas_list), len(documents_list)
        )
        ids = ids[:min_len]
        embeddings_list = embeddings_list[:min_len]
        metadatas_list = metadatas_list[:min_len]
        documents_list = documents_list[:min_len]
        print(f"Truncated lists to minimum length: {min_len}")
        print("----------------------------------\n")

    # Convert embeddings to a NumPy array
    try:
        print("DEBUG: Attempting np.array(embeddings_list)...")
        embeddings = np.array(embeddings_list)
        print(
            f"DEBUG: Successfully created embeddings array with shape: {embeddings.shape}"
        )
    except Exception as e:
        print("\n--- Error converting embeddings to NumPy array ---")
        print(f"Error: {e}")
        print(f"Type of embeddings_list: {type(embeddings_list)}")
        if isinstance(embeddings_list, list) and len(embeddings_list) > 0:
            print(f"Type of first element: {type(embeddings_list[0])}")
            # Check if embeddings have consistent dimensions
            try:
                first_len = len(embeddings_list[0])
                is_consistent = all(len(emb) == first_len for emb in embeddings_list)
                print(f"Embeddings lengths consistent: {is_consistent}")
                if not is_consistent:
                    inconsistent_indices = [
                        i
                        for i, emb in enumerate(embeddings_list)
                        if len(emb) != first_len
                    ]
                    print(
                        f"Indices with inconsistent lengths: {inconsistent_indices[:10]}..."
                    )  # Show first few
            except Exception as len_e:
                print(f"Could not check embedding lengths: {len_e}")
        traceback.print_exc()
        print("----------------------------------------------------\n")
        raise

    # Convert metadata list of dicts to a Pandas DataFrame
    try:
        print("DEBUG: Attempting pd.DataFrame(metadatas_list)...")
        metadata_df = pd.DataFrame(metadatas_list)
        print(
            f"DEBUG: Successfully created metadata DataFrame with shape: {metadata_df.shape}"
        )
        # Add IDs and potentially documents to the DataFrame
        metadata_df["doc_id"] = ids
        metadata_df["document_text"] = (
            documents_list  # Add original text if needed for hover
        )
        print("DEBUG: Added 'doc_id' and 'document_text' columns to DataFrame.")
    except Exception as e:
        print("\n--- Error creating Pandas DataFrame from metadata ---")
        print(f"Error: {e}")
        print(f"Type of metadatas_list: {type(metadatas_list)}")
        if isinstance(metadatas_list, list) and len(metadatas_list) > 0:
            print(f"Type of first element: {type(metadatas_list[0])}")
            if isinstance(metadatas_list[0], dict):
                print(f"Keys in first metadata dict: {list(metadatas_list[0].keys())}")
            else:
                print(f"First metadata element is not a dict: {metadatas_list[0]}")
        traceback.print_exc()
        print("-------------------------------------------------------\n")
        raise

    print("Data loaded successfully.")
    print("Metadata DataFrame info:")
    metadata_df.info()

    potential_color_keys = []
    print("\nIdentifying potential metadata keys for coloring...")
    for col in metadata_df.columns:
        try:
            # Exclude obviously unsuitable columns first
            if col in [
                "doc_id",
                "document_text",
                "Description",
                "Summary",
                "Summary Solution",
                "Summary in English",
            ]:
                continue  # Skip text fields and ID

            col_dtype_kind = metadata_df[
                col
            ].dtype.kind  #'O' (object), 'b' (bool), 'i' (int), 'f' (float), 'M' (datetime) etc.
            unique_count = metadata_df[col].nunique()

            # Heuristic: Categorical (object/string, boolean) or low-cardinality numerical/datetime
            if col_dtype_kind in ["O", "b"] or (
                col_dtype_kind in ["i", "f", "M"] and unique_count < 50
            ):
                if unique_count > 0:  # Ensure there's at least one value
                    print(
                        f"  - Considering '{col}' (Type: {metadata_df[col].dtype}, Unique: {unique_count})"
                    )
                    potential_color_keys.append(col)
                else:
                    print(
                        f"  - Skipping '{col}' (Type: {metadata_df[col].dtype}, Unique: 0 - No values)"
                    )
            # else: # Optional: Print why a column was skipped
            #    print(f"  - Skipping '{col}' (Type: {metadata_df[col].dtype}, Unique: {unique_count} - High cardinality or unsuitable type)")

        except Exception as e:
            print(
                f"  - Error processing column '{col}': {e}"
            )  # Catch errors during nunique() etc.

    print("\nPotential metadata keys for coloring:", potential_color_keys)

    return embeddings, metadata_df, potential_color_keys


def reduce_dimensionality(embeddings: np.ndarray, params=None):
    """Reduces dimensionality of embeddings using UMAP."""
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP is required for dimensionality reduction.")

    print(f"\nStarting dimensionality reduction using UMAP...")
    print(f"Input shape: {embeddings.shape}")
    if embeddings.shape[0] == 0:
        raise ValueError("Cannot reduce dimensionality of empty embeddings array.")

    # Use global REDUCER_PARAMS if params not provided
    if params is None:
        params = REDUCER_PARAMS

    target_dims = params.get('n_components', N_COMPONENTS)
    print(f"Target dimensions: {target_dims}")
    print(f"UMAP Parameters: {params}")

    if embeddings.shape[1] <= target_dims:
        print(f"Warning: Number of features ({embeddings.shape[1]}) is <= target dimensions ({target_dims}). Skipping reduction.")
        return embeddings[:, :target_dims] if embeddings.shape[1] > target_dims else embeddings

    reducer = umap.UMAP(**params)

    try:
        print("DEBUG: Fitting UMAP reducer...")
        # Ensure input is float32, UMAP sometimes prefers it
        embeddings_float32 = embeddings.astype(np.float32)
        embeddings_reduced = reducer.fit_transform(embeddings_float32)
        print(f"DEBUG: Fit complete. Reduced shape: {embeddings_reduced.shape}")
    except Exception as e:
        print("\n--- Error during dimensionality reduction ---")
        print(f"Reducer params: {params}")
        print(f"Input data type: {embeddings.dtype} -> {embeddings_float32.dtype}")
        if np.isnan(embeddings_float32).any():
            print("WARNING: Input embeddings contain NaNs!")
        if np.isinf(embeddings_float32).any():
            print("WARNING: Input embeddings contain Infs!")
        print(f"Error: {e}")
        traceback.print_exc()
        print("--------------------------------------------\n")
        raise

    print("Dimensionality reduction complete.")
    return embeddings_reduced


def visualize_embeddings_plotly(
    embeddings_reduced: np.ndarray,
    metadata_df: pd.DataFrame,
    color_by_key: str,
    filter_outliers: bool = False,
    outlier_quantile: float = 0.01
):
    """Creates an interactive scatter plot using Plotly, with optional outlier filtering."""
    print("\n--- Preparing visualization ---")
    if color_by_key not in metadata_df.columns:
        # Handle case where clustering failed and 'cluster_label' isn't present
        if color_by_key == 'cluster_label':
            print(f"Warning: Requested coloring by 'cluster_label', but it's not in the DataFrame.")
            available_keys = [k for k in metadata_df.columns if k not in ['doc_id', 'document_text', 'embeddings']]
            if available_keys:
                fallback_key = available_keys[0]
                print(f"Falling back to coloring by '{fallback_key}'.")
                color_by_key = fallback_key
            else:
                raise ValueError("Color key 'cluster_label' not found and no fallback keys available.")
        else:
            raise ValueError(f"Color key '{color_by_key}' not found in metadata columns: {metadata_df.columns.tolist()}")

    n_components = embeddings_reduced.shape[1]
    print(f"Plotting {n_components}D data.")
    if n_components not in [2, 3]:
        raise ValueError(f"Reduced embeddings must have 2 or 3 dimensions for plotting, found {n_components}")

    if len(metadata_df) != len(embeddings_reduced):
        print(f"Warning: Length mismatch! Metadata rows ({len(metadata_df)}) != Reduced embedding points ({len(embeddings_reduced)}).")
        min_len = min(len(metadata_df), len(embeddings_reduced))
        metadata_df = metadata_df.iloc[:min_len].copy()
        embeddings_reduced = embeddings_reduced[:min_len]
        print(f"Aligned data to length {min_len}.")

    # Create a plotting DataFrame
    plot_df = pd.DataFrame()
    coord_cols = []
    if n_components == 3:
        coord_cols = ['x', 'y', 'z']
        plot_df['x'] = embeddings_reduced[:, 0]
        plot_df['y'] = embeddings_reduced[:, 1]
        plot_df['z'] = embeddings_reduced[:, 2]
        plot_func = px.scatter_3d
        coord_args = dict(x='x', y='y', z='z')
        title = f"3D UMAP ({UMAP_N_NEIGHBORS} neighbors, {UMAP_MIN_DIST} min_dist, {UMAP_METRIC}) colored by '{color_by_key}'"
    else:
        coord_cols = ['x', 'y']
        plot_df['x'] = embeddings_reduced[:, 0]
        plot_df['y'] = embeddings_reduced[:, 1]
        plot_func = px.scatter
        coord_args = dict(x='x', y='y')
        title = f"2D UMAP ({UMAP_N_NEIGHBORS} neighbors, {UMAP_MIN_DIST} min_dist, {UMAP_METRIC}) colored by '{color_by_key}'"

    # Add metadata
    metadata_to_add = metadata_df.reset_index(drop=True)
    plot_df = pd.concat([plot_df, metadata_to_add], axis=1)

    # Filter outliers if requested
    if filter_outliers and n_components >= 2:
        print(f"Filtering visual outliers based on {outlier_quantile} quantile...")
        original_count = len(plot_df)
        filter_mask = pd.Series(True, index=plot_df.index)
        for col in coord_cols:
            lower_bound = plot_df[col].quantile(outlier_quantile)
            upper_bound = plot_df[col].quantile(1 - outlier_quantile)
            filter_mask &= (plot_df[col] >= lower_bound) & (plot_df[col] <= upper_bound)
            print(f"  - Filtering '{col}' between {lower_bound:.2f} and {upper_bound:.2f}")

        plot_df = plot_df[filter_mask]
        filtered_count = len(plot_df)
        print(f"Filtered {original_count - filtered_count} points ({100*(original_count - filtered_count)/original_count:.1f}%).")
        if filtered_count == 0:
            print("Warning: Filtering removed all points! Reverting...")
            plot_df = pd.concat([pd.DataFrame(embeddings_reduced[:, :n_components], columns=coord_cols), metadata_to_add], axis=1)

    # Handle color column
    if color_by_key == 'cluster_label':
        if plot_df[color_by_key].isnull().any():
            plot_df[color_by_key] = plot_df[color_by_key].fillna(-1)
        plot_df[color_by_key] = plot_df[color_by_key].astype(int).astype(str)
        plot_df[color_by_key] = plot_df[color_by_key].replace('-1', 'Noise')
    elif plot_df[color_by_key].isnull().any():
        if not pd.api.types.is_string_dtype(plot_df[color_by_key]):
            plot_df[color_by_key] = plot_df[color_by_key].astype(str)
        plot_df[color_by_key] = plot_df[color_by_key].fillna('Unknown')

    # Determine hover data
    hover_cols = [col for col in metadata_df.columns if col not in ['Description','Solution', 'Summary Solution', 'Summary in English','document_text']]
    hover_cols = [col for col in hover_cols if col not in ['x', 'y', 'z']]
    if 'document_text' in metadata_df.columns:
        plot_df['hover_text'] = plot_df['document_text'].str[:100] + '...'
        hover_cols.insert(0, 'hover_text')
    if color_by_key not in hover_cols and color_by_key in plot_df.columns:
        hover_cols.append(color_by_key)

    # Define color map and opacity settings
    color_discrete_map = {'Noise': 'grey'} if 'Noise' in plot_df[color_by_key].unique() else None
    
    # Create base figure without opacity setting
    fig = plot_func(
        plot_df,
        **coord_args,
        color=color_by_key,
        title=title,
        hover_data=hover_cols,
        color_discrete_map=color_discrete_map,
        color_discrete_sequence=px.colors.qualitative.Plotly if not color_discrete_map else None
    )

    # Update traces with different opacity values
    for trace in fig.data:
        if trace.name == 'Noise':
            trace.marker.opacity = 0.3  # Lower opacity for noise points
        else:
            trace.marker.opacity = 0.7  # Higher opacity for regular points

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=50),
        legend_title_text=color_by_key
    )
    marker_size = 3 if n_components == 3 else 5
    fig.update_traces(marker=dict(size=marker_size))

    print("Plot generated. Displaying in browser...")
    fig.show()


def run_hdbscan_clustering(embeddings: np.ndarray, min_cluster_size: int, min_samples: int = None, metric: str = 'euclidean') -> np.ndarray:
    """Performs HDBSCAN clustering on the embeddings."""
    if not HDBSCAN_AVAILABLE:
        print("HDBSCAN not available, skipping clustering.")
        return None

    print("\n--- Starting HDBSCAN Clustering ---")
    print(f"Parameters: min_cluster_size={min_cluster_size}, min_samples={min_samples}, metric='{metric}'")
    print(f"Input embedding shape: {embeddings.shape}")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        core_dist_n_jobs=-1  # Use all available CPU cores
    )

    try:
        cluster_labels = clusterer.fit_predict(embeddings)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = np.sum(cluster_labels == -1)
        print(f"Clustering complete.")
        print(f"  - Found {n_clusters} clusters.")
        print(f"  - Found {n_noise} noise points (label -1).")
        print(f"  - Cluster label counts: {np.unique(cluster_labels, return_counts=True)}")
        print("--- HDBSCAN Clustering Finished ---")
        return cluster_labels
    except Exception as e:
        print("\n--- Error during HDBSCAN clustering ---")
        print(f"Error: {e}")
        traceback.print_exc()
        print("-------------------------------------\n")
        return None


# --- Main Execution ---
if __name__ == "__main__":
    if not UMAP_AVAILABLE:
        print("UMAP is required to run this script. Please install umap-learn.")
        exit()

    try:
        # 1. Load Data
        embeddings, metadata_df, potential_keys = load_data_from_chroma(
            DB_PATH, COLLECTION_NAME
        )

        # 2. Perform Clustering (Optional but recommended)
        cluster_labels = None
        if HDBSCAN_AVAILABLE:
            cluster_labels = run_hdbscan_clustering(
                embeddings,
                min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                min_samples=HDBSCAN_MIN_SAMPLES,
                metric=HDBSCAN_METRIC
            )
            if cluster_labels is not None:
                if len(cluster_labels) == len(metadata_df):
                    metadata_df['cluster_label'] = cluster_labels
                    print("Added 'cluster_label' column to metadata.")
                    if 'cluster_label' in potential_keys:
                        potential_keys.remove('cluster_label')
                    potential_keys.insert(0, 'cluster_label')
                else:
                    print(f"Warning: Length mismatch between embeddings and metadata after clustering.")

        # 3. Select Metadata Key for Coloring
        print("\nAvailable metadata keys for coloring:")
        key_map = {str(i+1): key for i, key in enumerate(potential_keys)}
        for i, key in enumerate(potential_keys):
            if key in metadata_df.columns:
                unique_count = metadata_df[key].nunique()
                print(f"  {i+1}. {key} (Unique: {unique_count}, Type: {metadata_df[key].dtype})")
            elif key == 'cluster_label':
                print(f"  {i+1}. {key} (Clustering data not available)")
            else:
                print(f"  {i+1}. {key} (Key not found in DataFrame)")

        # Select coloring key with default option
        selected_key = None
        default_key = 'cluster_label' if 'cluster_label' in metadata_df.columns else (potential_keys[0] if potential_keys else None)

        while selected_key is None:
            try:
                prompt = f"Enter the number or name of the metadata key to color by (default='{default_key}'): "
                choice = input(prompt).strip()
                if not choice and default_key:
                    if default_key in metadata_df.columns:
                        selected_key = default_key
                    else:
                        print("Default key unavailable. Please choose from the list.")
                        continue
                elif choice in key_map:
                    selected_key = key_map[choice]
                    if selected_key not in metadata_df.columns and selected_key != 'cluster_label':
                        print(f"Error: Key '{selected_key}' not found in DataFrame.")
                        selected_key = None
                elif choice in metadata_df.columns:
                    selected_key = choice
                else:
                    print(f"Invalid choice '{choice}'. Please try again.")

            except KeyboardInterrupt:
                print("\nOperation cancelled by user.")
                exit()
            except Exception as e:
                print(f"An error occurred during input: {e}")
                exit()

        print(f"\nSelected coloring key: '{selected_key}'")
        if selected_key in metadata_df.columns:
            print(f"Number of unique values: {metadata_df[selected_key].nunique()}")
            print(f"Data type: {metadata_df[selected_key].dtype}")

        # 4. Reduce Dimensionality
        print("\nRunning UMAP with configured parameters...")
        embeddings_reduced = reduce_dimensionality(
            embeddings, params=REDUCER_PARAMS
        )

        # 5. Visualize
        visualize_embeddings_plotly(
            embeddings_reduced,
            metadata_df,
            selected_key,
            filter_outliers=VISUALIZATION_FILTER_OUTLIERS,
            outlier_quantile=VISUALIZATION_OUTLIER_QUANTILE
        )

        print("\nScript finished successfully.")

    except FileNotFoundError as e:
        print(f"\n--- Error: File Not Found ---")
        print(f"{e}")
        print("Please ensure the ChromaDB path is correct and the database exists.")
    except ValueError as e:
        print(f"\n--- Error: Value Error ---")
        print(f"{e}")
        traceback.print_exc()
    except ImportError as e:
        print(f"\n--- Error: Import Error ---")
        print(f"{e}")
        print("Check installations: pip install chromadb pandas numpy plotly umap-learn hdbscan python-dotenv")
    except Exception as e:
        print(f"\n--- An unexpected error occurred ---")
        print(f"Error: {e}")
        traceback.print_exc()
