# D4-Helpdesk\scripts\exploration\visualize_embeddings.py

import os
import chromadb
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from dotenv import load_dotenv
import traceback
import random
from functools import partial
import time  # For timing searches

# --- Local Utility Import ---
# Ensure utils.py is accessible (e.g., in the same directory or added to PYTHONPATH)
try:
    from utils_clustering import save_hyperopt_results 
except ImportError:
    print("Error: Could not import 'save_hyperopt_results' from utils.py.")
    print("Ensure utils.py is in the correct path.")

    # Define a dummy function to avoid NameError later if import fails
    def save_hyperopt_results(*args, **kwargs):
        print("Warning: save_hyperopt_results function is unavailable.")


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
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Error: umap-learn library not found. Visualization requires UMAP.")
    print("Install it using: pip install umap-learn")
    # We might still proceed if only doing tuning without final vis, but let's exit for now
    # if not needed for other parts. Let's assume it's needed.
    exit()

# --- Hyperparameter Optimization ---
try:
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False
    print("Warning: hyperopt library not found. Bayesian optimization disabled.")
    print("Install it using: pip install hyperopt")

# --- Progress Bar ---
try:
    from tqdm import trange, tqdm  # For progress bars in searches

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

    # Define dummy functions if tqdm is not available
    def trange(x, *args, **kwargs):
        return range(x)

    def tqdm(x, *args, **kwargs):
        return x

    print("Warning: tqdm library not found. Progress bars will be disabled.")
    print("Install it using: pip install tqdm")


# Load environment variables
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

# --- Configuration ---
HELPDESK_DIR = Path(__file__).resolve().parents[2]
EMBEDDINGS_DIR = HELPDESK_DIR / "helpdesk-data" / "helpdesk-embeddings"
DB_PATH = EMBEDDINGS_DIR / "chroma_summaries"
HYPEROPT_DIR = EMBEDDINGS_DIR / "hyperopt"

COLLECTION_NAME = "helpdesk_summaries_embeddings"
RANDOM_STATE = None #42  # Consistent random state for UMAP/searches

# --- Default UMAP/HDBSCAN Parameters (if tuning is skipped) ---
DEFAULT_N_COMPONENTS = 3  # 3 for 3D, 2 for 2D visualization
DEFAULT_UMAP_N_NEIGHBORS = 15  # Default from article/common practice
DEFAULT_UMAP_MIN_DIST = 0.1  # Default from umap-learn
DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE = 10  # Default if skipping tuning

# --- Hyperparameter Tuning Configuration ---
# Choose 'bayesian', 'random', or 'none'
TUNING_METHOD = "bayesian"  # Or 'random' or 'none'
NUM_EVALS = 30  # Number of iterations for random/bayesian search (adjust based on time)
PROB_THRESHOLD = (
    0.05  # Minimum probability for a point to be considered 'well clustered'
)

# --- Domain Knowledge Constraints for Tuning ---
# Adjust these based on your expectation for helpdesk ticket categories
# How many distinct types of emails do you expect?
LABEL_LOWER_BOUND = 10  # Minimum expected number of meaningful clusters
LABEL_UPPER_BOUND = 75  # Maximum expected number of meaningful clusters

# --- Search Space Definition ---
# Define ranges for hyperparameters to search over
# Using hyperopt's hp for Bayesian search compatibility
# For random search, we'll sample from these using random.choice or similar
SEARCH_SPACE = {
    "n_neighbors": hp.choice("n_neighbors", range(5, 51, 5)),  # e.g., 5, 10, 15... 50
    "n_components": hp.choice(
        "n_components", range(3, 16, 2)
    ),  # e.g., 3, 5, 7... 15 (Note: higher dims can be better for clustering)
    "min_cluster_size": hp.choice(
        "min_cluster_size", range(5, 31, 5)
    ),  # e.g., 5, 10, 15... 30
    # We will use a fixed random_state for UMAP within generate_clusters
}

# --- Visualization Configuration ---
VISUALIZATION_N_COMPONENTS = (
    3  # Force 2D or 3D for the *final* plot, separate from tuning n_components
)
VISUALIZATION_FILTER_OUTLIERS = True
VISUALIZATION_OUTLIER_QUANTILE = 0.025

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


# --- Functions from the Article (Adapted) ---


def generate_clusters(
    message_embeddings,
    n_neighbors,
    n_components,
    min_cluster_size,
    random_state=RANDOM_STATE,
):  # Use global random state
    """
    Generate HDBSCAN cluster object after reducing embedding dimensionality with UMAP.
    Uses cosine metric for UMAP and Euclidean for HDBSCAN on reduced embeddings.
    Uses 'eom' cluster selection method as recommended.
    """
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP not found.")
    if not HDBSCAN_AVAILABLE:
        raise ImportError("HDBSCAN not found.")

    print(f"  Running UMAP (k={n_neighbors}, dim={n_components})...", end="")
    umap_embeddings = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric="cosine",  # Good for text embeddings
        random_state=random_state,
        low_memory=True,
    ).fit_transform(message_embeddings)
    print(" Done. Running HDBSCAN...", end="")
    clusters = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",  # Recommended after UMAP
        cluster_selection_method="eom",  # Excess of Mass
        # min_samples = None, # Let HDBSCAN default (usually = min_cluster_size)
    ).fit(umap_embeddings)
    print(" Done.")
    return clusters, umap_embeddings  # Return reduced embeddings too


def score_clusters(clusters, prob_threshold=PROB_THRESHOLD):
    """
    Returns the number of clusters and cost (percent of data points low confidence)
    for a given HDBSCAN cluster object.
    """
    cluster_labels = clusters.labels_
    label_count = len(np.unique(cluster_labels)) - (
        1 if -1 in cluster_labels else 0
    )  # Exclude noise label (-1)
    total_num = len(clusters.labels_)
    if total_num == 0:
        return 0, 1.0  # Avoid division by zero if no points

    cost = np.count_nonzero(clusters.probabilities_ < prob_threshold) / total_num
    return label_count, cost


# --- Random Search Function ---
def random_search(embeddings, space_dict, num_evals, label_lower, label_upper):
    """
    Randomly search hyperparameter space limited number of times
    and return a summary of the results, filtered by label count constraints.
    """
    results = []
    space_options = {
        "n_neighbors": list(
            space_dict["n_neighbors"].pos_args[1]._obj
        ),  # Extract range/list
        "n_components": list(space_dict["n_components"].pos_args[1]._obj),
        "min_cluster_size": list(space_dict["min_cluster_size"].pos_args[1]._obj),
    }

    print(f"\n--- Starting Random Search ({num_evals} evaluations) ---")
    start_time = time.time()

    for i in trange(num_evals, desc="Random Search Progress"):
        params = {
            "n_neighbors": random.choice(space_options["n_neighbors"]),
            "n_components": random.choice(space_options["n_components"]),
            "min_cluster_size": random.choice(space_options["min_cluster_size"]),
        }
        print(f"\nTrial {i+1}/{num_evals}: PARAMS = {params}")

        try:
            clusters, _ = generate_clusters(
                embeddings, **params, random_state=RANDOM_STATE
            )
            label_count, cost = score_clusters(clusters, prob_threshold=PROB_THRESHOLD)
            print(f"  RESULT: Clusters={label_count}, Cost={cost:.4f}")

            # Store results regardless of constraints for now
            results.append(
                [
                    i,
                    params["n_neighbors"],
                    params["n_components"],
                    params["min_cluster_size"],
                    label_count,
                    cost,
                ]
            )

        except Exception as e:
            print(f"  ERROR during trial {i+1} with params {params}: {e}")
            # Optionally store error marker: results.append([i, params['n_neighbors'], ... , -1, 999.0])

    end_time = time.time()
    print(f"--- Random Search Finished ({end_time - start_time:.2f} seconds) ---")

    if not results:
        print("No results generated from random search.")
        return None, None

    result_df = pd.DataFrame(
        results,
        columns=[
            "run_id",
            "n_neighbors",
            "n_components",
            "min_cluster_size",
            "label_count",
            "cost",
        ],
    )

    # Filter based on constraints
    filtered_df = result_df[
        (result_df["label_count"] >= label_lower)
        & (result_df["label_count"] <= label_upper)
    ].copy()

    if filtered_df.empty:
        print(
            f"Warning: No runs met the cluster count constraints ({label_lower}-{label_upper})."
        )
        print("Returning the overall best run (lowest cost) regardless of constraints.")
        best_run = result_df.loc[result_df["cost"].idxmin()]
    else:
        print(f"Found {len(filtered_df)} runs meeting cluster constraints.")
        # Sort the *filtered* results by cost and get the best
        filtered_df = filtered_df.sort_values(by="cost")
        best_run = filtered_df.iloc[0]

    best_params_dict = {
        "n_neighbors": int(best_run["n_neighbors"]),
        "n_components": int(best_run["n_components"]),
        "min_cluster_size": int(best_run["min_cluster_size"]),
    }

    print("\nBest parameters found (meeting constraints if possible):")
    print(best_params_dict)
    print(f"Label Count: {int(best_run['label_count'])}")
    print(f"Cost: {best_run['cost']:.4f}")

    return best_params_dict, result_df  # Return best params and all results


# --- Bayesian Optimization Functions (using Hyperopt) ---
def objective(
    params, embeddings, label_lower, label_upper, prob_threshold=PROB_THRESHOLD
):
    """
    Objective function for hyperopt to minimize.
    Includes penalty for cluster counts outside the desired range.
    """
    start_time = time.time()
    print(
        f"  Evaluating: n_neighbors={params['n_neighbors']}, "
        f"n_components={params['n_components']}, "
        f"min_cluster_size={params['min_cluster_size']} ... ",
        end="",
    )

    try:
        clusters, _ = generate_clusters(embeddings, **params, random_state=RANDOM_STATE)
        label_count, cost = score_clusters(clusters, prob_threshold=prob_threshold)

        # Penalty for being outside the desired cluster range
        if not (label_lower <= label_count <= label_upper):
            penalty = 0.15  # As suggested in the article
            print(
                f" OUTSIDE CONSTRAINTS ({label_count} clusters). Cost={cost:.4f}, Penalty={penalty:.2f}",
                end="",
            )
        else:
            penalty = 0
            print(f" OK ({label_count} clusters). Cost={cost:.4f}", end="")

        loss = cost + penalty
        eval_time = time.time() - start_time
        print(f" -> Loss={loss:.4f} (Time: {eval_time:.2f}s)")

        return {
            "loss": loss,
            "status": STATUS_OK,
            "label_count": label_count,
            "params": params,
        }

    except Exception as e:
        print(f" FAILED. Error: {e}")
        # traceback.print_exc() # Optionally print full traceback for debugging
        # Return high loss for failed trials
        return {"loss": 999.0, "status": hyperopt.STATUS_FAIL, "error": str(e)}


def bayesian_search(embeddings, space, label_lower, label_upper, max_evals=NUM_EVALS):
    """
    Perform Bayesian search using Hyperopt to minimize the objective function.
    """
    if not HYPEROPT_AVAILABLE:
        print("Error: Hyperopt is not available. Cannot perform Bayesian search.")
        return None, None, None

    trials = Trials()
    # Use partial to fix the embeddings, bounds arguments for the objective function
    fmin_objective = partial(
        objective,
        embeddings=embeddings,
        label_lower=label_lower,
        label_upper=label_upper,
    )

    print(f"\n--- Starting Bayesian Optimization ({max_evals} evaluations) ---")
    start_time = time.time()

    # Add random_state to the space if not implicitly handled
    space_with_rs = space.copy()
    # space_with_rs['random_state'] = RANDOM_STATE # Not needed as generate_clusters uses it

    best = fmin(
        fn=fmin_objective,
        space=space,
        algo=tpe.suggest,  # Tree-structured Parzen Estimator
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(RANDOM_STATE),
    )  # Seed the search process

    end_time = time.time()
    print(
        f"--- Bayesian Optimization Finished ({end_time - start_time:.2f} seconds) ---"
    )

    if not trials.best_trial or trials.best_trial["result"]["status"] != STATUS_OK:
        print("Warning: Bayesian optimization did not find a successful best trial.")
        # Optionally, you could analyze trials.results to find the best *successful* one
        return None, None, trials

    # Retrieve the best parameters using space_eval
    best_params_evaluated = space_eval(space, best)

    print("\nBest parameters found by Bayesian search:")
    print(best_params_evaluated)
    best_result = trials.best_trial["result"]
    print(f"Label Count: {best_result['label_count']}")
    print(f"Loss (Cost + Penalty): {best_result['loss']:.4f}")

    # Note: generate_clusters needs to be called again with best params
    # outside this function to get the final cluster object if needed immediately.

    return best_params_evaluated, trials  # Return best params and trials object


# --- Visualization Function (Modified slightly) ---
def visualize_embeddings_plotly(
    embeddings_reduced: np.ndarray,
    metadata_df: pd.DataFrame,
    color_by_key: str,
    title: str,  # Add title parameter
    filter_outliers: bool = False,
    outlier_quantile: float = 0.01,
):
    """Creates an interactive scatter plot using Plotly, with optional outlier filtering."""
    print("\n--- Preparing visualization ---")
    if color_by_key not in metadata_df.columns:
        raise ValueError(
            f"Color key '{color_by_key}' not found in metadata columns: {metadata_df.columns.tolist()}"
        )

    n_components = embeddings_reduced.shape[1]
    print(f"Plotting {n_components}D data.")
    if n_components not in [2, 3]:
        raise ValueError(
            f"Reduced embeddings must have 2 or 3 dimensions for plotting, found {n_components}"
        )

    if len(metadata_df) != len(embeddings_reduced):
        print(
            f"Warning: Length mismatch! Metadata rows ({len(metadata_df)}) != Reduced embedding points ({len(embeddings_reduced)})."
        )
        min_len = min(len(metadata_df), len(embeddings_reduced))
        metadata_df = metadata_df.iloc[:min_len].copy()
        embeddings_reduced = embeddings_reduced[:min_len]
        print(f"Aligned data to length {min_len}.")

    # Create a plotting DataFrame
    plot_df = pd.DataFrame()
    coord_cols = []
    if n_components == 3:
        coord_cols = ["x", "y", "z"]
        plot_df["x"] = embeddings_reduced[:, 0]
        plot_df["y"] = embeddings_reduced[:, 1]
        plot_df["z"] = embeddings_reduced[:, 2]
        plot_func = px.scatter_3d
        coord_args = dict(x="x", y="y", z="z")
    else:  # n_components == 2
        coord_cols = ["x", "y"]
        plot_df["x"] = embeddings_reduced[:, 0]
        plot_df["y"] = embeddings_reduced[:, 1]
        plot_func = px.scatter
        coord_args = dict(x="x", y="y")

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
            print(
                f"  - Filtering '{col}' between {lower_bound:.2f} and {upper_bound:.2f}"
            )

        plot_df = plot_df[filter_mask]
        filtered_count = len(plot_df)
        print(
            f"Filtered {original_count - filtered_count} points ({100*(original_count - filtered_count)/original_count:.1f}%)."
        )
        if filtered_count == 0:
            print("Warning: Filtering removed all points! Reverting...")
            plot_df = pd.concat(
                [
                    pd.DataFrame(
                        embeddings_reduced[:, :n_components], columns=coord_cols
                    ),
                    metadata_to_add,
                ],
                axis=1,
            )

    # Handle color column (especially cluster labels)
    plot_df[color_by_key] = plot_df[color_by_key].fillna(
        "Unknown"
    )  # Fill NaNs before converting
    if color_by_key == "cluster_label":
        # Ensure conversion to string for categorical coloring, map -1 to 'Noise'
        plot_df[color_by_key] = (
            plot_df[color_by_key].astype(int).astype(str).replace("-1", "Noise")
        )
    else:
        # Ensure other types are strings for consistent coloring if needed
        if not pd.api.types.is_numeric_dtype(plot_df[color_by_key]):
            plot_df[color_by_key] = plot_df[color_by_key].astype(str)

    # Determine hover data
    hover_cols = [
        col
        for col in metadata_df.columns
        if col
        not in [
            "Description",
            "Solution",
            "Summary Solution",
            "Summary in English",
            "document_text",
        ]
    ]
    hover_cols = [col for col in hover_cols if col not in ["x", "y", "z"]]
    if "document_text" in metadata_df.columns:
        plot_df["hover_text"] = plot_df["document_text"].str[:100] + "..."
        # Ensure hover_text is added to hover_data list if the column exists
        if "hover_text" not in hover_cols:
            hover_cols.insert(0, "hover_text")  # Add truncated text at the beginning
    if color_by_key not in hover_cols and color_by_key in plot_df.columns:
        hover_cols.append(color_by_key)

    # Define color map for Noise points
    color_discrete_map = (
        {"Noise": "lightgrey"} if "Noise" in plot_df[color_by_key].unique() else None
    )

    fig = plot_func(
        plot_df,
        **coord_args,
        color=color_by_key,
        title=title,  # Use passed title
        hover_data=hover_cols,
        opacity=0.7,
        color_discrete_map=color_discrete_map,  # Map Noise to grey
        # Use a qualitative color sequence if not using discrete map or if Noise is not the only category
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=50),  # Add more top margin for title
        legend_title_text=color_by_key,
    )
    marker_size = 3 if n_components == 3 else 5
    fig.update_traces(marker=dict(size=marker_size))

    print("Plot generated. Displaying in browser...")
    fig.show()


# --- Main Execution ---
if __name__ == "__main__":
    # --- Setup ---
    if not UMAP_AVAILABLE:
        exit("UMAP is required.")
    # Allow running without HDBSCAN, but tuning/clustering won't work fully
    if not HDBSCAN_AVAILABLE:
        print(
            "Warning: HDBSCAN not found. Tuning and clustering steps will be skipped."
        )

    try:
        # 1. Load Data
        embeddings, metadata_df, potential_keys = load_data_from_chroma(
            DB_PATH, COLLECTION_NAME
        )

        # --- 2. Hyperparameter Tuning ---
        best_params = None
        tuning_results = None  # Store results (DataFrame or Trials)

        # Determine if tuning can/should run
        can_tune = HDBSCAN_AVAILABLE and (TUNING_METHOD in ["random", "bayesian"])
        if TUNING_METHOD == "bayesian" and not HYPEROPT_AVAILABLE:
            print(
                "Hyperopt not found, cannot perform Bayesian optimization. Skipping tuning."
            )
            can_tune = False
        if not can_tune and TUNING_METHOD != "none":
            print(
                f"Tuning method '{TUNING_METHOD}' requires libraries that are not available. Skipping tuning."
            )
            TUNING_METHOD = "none"  # Force skip if requirements not met

        if TUNING_METHOD == "none":
            print("\n--- Skipping Hyperparameter Tuning ---")
            best_params = {
                "n_neighbors": DEFAULT_UMAP_N_NEIGHBORS,
                "n_components": DEFAULT_N_COMPONENTS,  # Use vis components if not tuning
                "min_cluster_size": DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE,
            }
        elif TUNING_METHOD == "random":
            best_params, tuning_results = random_search(
                embeddings,
                SEARCH_SPACE,
                NUM_EVALS,
                LABEL_LOWER_BOUND,
                LABEL_UPPER_BOUND,
            )
            if best_params is None:
                exit("Random search failed.")
        elif TUNING_METHOD == "bayesian":
            best_params, tuning_results = bayesian_search(
                embeddings,
                SEARCH_SPACE,
                LABEL_LOWER_BOUND,
                LABEL_UPPER_BOUND,
                NUM_EVALS,
            )
            if best_params is None:
                exit("Bayesian search failed.")

        # --- 3. Final Run with Best Parameters ---
        print(f"\n--- Running Final Clustering with Best Parameters ---")
        print(f"Best Parameters: {best_params}")

        final_clusters = None
        final_umap_embeddings = None  # Embeddings reduced with best UMAP params
        final_cluster_labels = np.full(len(metadata_df), -1)  # Default to noise

        if HDBSCAN_AVAILABLE:  # Only run if HDBSCAN is present
            try:
                final_clusters, final_umap_embeddings = generate_clusters(
                    embeddings,
                    n_neighbors=best_params["n_neighbors"],
                    n_components=best_params[
                        "n_components"
                    ],  # Use tuned component count
                    min_cluster_size=best_params["min_cluster_size"],
                    random_state=RANDOM_STATE,
                )
                final_cluster_labels = final_clusters.labels_
                label_count, cost = score_clusters(
                    final_clusters, prob_threshold=PROB_THRESHOLD
                )
                print(
                    f"Final Run Results: Found {label_count} clusters, Cost={cost:.4f}"
                )

                if len(final_cluster_labels) == len(metadata_df):
                    metadata_df["cluster_label"] = final_cluster_labels
                    print("Added final 'cluster_label' column to metadata.")
                    if "cluster_label" not in potential_keys:
                        potential_keys.insert(0, "cluster_label")
                else:
                    print(
                        f"Warning: Mismatch between final cluster labels ({len(final_cluster_labels)}) and metadata ({len(metadata_df)}). Skipping label assignment."
                    )

            except Exception as e:
                print(f"\n--- Error during final clustering run --- Error: {e}")
                traceback.print_exc()
                print("Proceeding without final cluster labels.")
        else:
            print("HDBSCAN not available, skipping final clustering step.")

        # --- 4. Prepare Embeddings for Visualization ---
        print(
            f"\n--- Preparing Embeddings for {VISUALIZATION_N_COMPONENTS}D Visualization ---"
        )
        embeddings_reduced_vis = None
        # Check if final UMAP run already produced the desired dimensions
        if (
            final_umap_embeddings is not None
            and final_umap_embeddings.shape[1] == VISUALIZATION_N_COMPONENTS
        ):
            print("Using UMAP embeddings generated during the final clustering run.")
            embeddings_reduced_vis = final_umap_embeddings
        else:
            # Need to rerun UMAP for visualization dimensions
            current_ncomp = (
                final_umap_embeddings.shape[1]
                if final_umap_embeddings is not None
                else best_params["n_components"]
            )
            print(
                f"Best/current 'n_components' ({current_ncomp}) differs from vis target ({VISUALIZATION_N_COMPONENTS})."
            )
            print(
                f"Re-running UMAP with k={best_params['n_neighbors']} and target dim={VISUALIZATION_N_COMPONENTS}."
            )
            vis_reducer = umap.UMAP(
                n_neighbors=best_params["n_neighbors"],
                n_components=VISUALIZATION_N_COMPONENTS,
                min_dist=DEFAULT_UMAP_MIN_DIST,  # Use default or tune this too? For now, default.
                metric="cosine",
                random_state=RANDOM_STATE,
                low_memory=True,
            )
            try:
                # Ensure input is float32 for UMAP
                embeddings_float32 = embeddings.astype(np.float32)
                if np.isnan(embeddings_float32).any():
                    print("Warning: NaNs detected in embeddings before UMAP vis run.")
                embeddings_reduced_vis = vis_reducer.fit_transform(embeddings_float32)
            except Exception as e:
                print(f"Error running UMAP for visualization: {e}.")
                traceback.print_exc()
                print("Visualization might fail or be incorrect.")
                # As a last resort, create dummy data of the right shape? Or just let it fail later?
                # Let it fail later for now, providing None

        # --- 5. Visualize Results ---
        if embeddings_reduced_vis is not None:
            # Determine color key - prioritize cluster_label if available
            color_key = (
                "cluster_label"
                if "cluster_label" in metadata_df.columns
                else (potential_keys[0] if potential_keys else None)
            )

            if color_key is None:
                print(
                    "Error: Cannot determine a column to color the plot by. Visualization skipped."
                )
            else:
                print(
                    f"\nVisualizing results colored by '{color_key}' using best parameters..."
                )
                plot_title = (
                    f"{VISUALIZATION_N_COMPONENTS}D UMAP Viz "
                    f"(Params: k={best_params['n_neighbors']}, comp={best_params['n_components']}, mcs={best_params['min_cluster_size']}) | "
                    f"Color: '{color_key}'"
                )

                visualize_embeddings_plotly(
                    embeddings_reduced_vis,
                    metadata_df,
                    color_by_key=color_key,
                    title=plot_title,
                    filter_outliers=VISUALIZATION_FILTER_OUTLIERS,
                    outlier_quantile=VISUALIZATION_OUTLIER_QUANTILE,
                )
        else:
            print(
                "Skipping visualization because reduced embeddings for visualization could not be generated."
            )

        # --- 6. Save Tuning Results ---
        if TUNING_METHOD in ["random", "bayesian"] and tuning_results is not None:
            print("\n--- Saving Tuning Results ---")
            save_hyperopt_results(
                results=tuning_results,
                tuning_method=TUNING_METHOD,
                num_evals=NUM_EVALS,
                output_dir=HYPEROPT_DIR,  # Pass the defined Path object
            )
        else:
            print("\n--- No tuning results to save ---")

        print("\nScript finished.")

    # --- Error Handling ---
    except FileNotFoundError as e:
        print(f"\n--- Error: File Not Found --- \n{e}")
    except ValueError as e:
        print(f"\n--- Error: Value Error --- \n{e}")
        traceback.print_exc()
    except ImportError as e:
        print(f"\n--- Error: Import Error --- \n{e}")
    except Exception as e:
        print(f"\n--- An unexpected error occurred ---")
        print(f"Error: {e}")
        traceback.print_exc()
