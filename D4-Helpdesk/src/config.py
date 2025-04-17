"""
Centralized configuration management for the Helpdesk Clustering project.

Handles loading environment variables, defining file paths, and dataset-specific settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file in the project root (D4-Helpdesk)
# Assumes the .env file is located one level up from the 'src' directory
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / '.env')


# --- Core Path Definitions ---
BASE_DIR = Path(__file__).resolve().parents[1]  # Project root: D4-Helpdesk/
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
SCRIPTS_DIR = BASE_DIR / "scripts"
NOTEBOOKS_DIR = BASE_DIR / "notebooks" # Optional, but good practice

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
HYPEROPT_DIR = VECTOR_STORE_DIR / "hyperopt" # Store hyperopt results within vector_store


# --- Environment Variable Loading ---
# Azure OpenAI Credentials (Ensure these are set in your .env file)
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2023-05-15") # Default value if not set

# Embedding Model Details (Ensure these are set in your .env file)
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_OPENAI_EMBEDDING_MODEL_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_NAME")

# --- Default Analysis Parameters (Can be overridden in dataset config or scripts) ---
DEFAULT_UMAP_PARAMS = {
    "n_neighbors": 15,
    "n_components": 5,
    "min_dist": 0.0,
    "metric": "cosine",
    # Add other relevant UMAP defaults
}

DEFAULT_HDBSCAN_PARAMS = {
    "min_cluster_size": 15,
    "metric": "euclidean",
    "cluster_selection_method": "eom",
    # Add other relevant HDBSCAN defaults
}

DEFAULT_OPTIMIZATION_PARAMS = {
    "random_state": 42,
    "max_evals_random": 100,
    "max_evals_bayesian": 100,
    # Add other optimization defaults
}

# --- Dataset Specific Configurations ---
# This dictionary holds the specific settings for each dataset we want to process.
# Add new datasets here by creating a new key-value pair.
DATASET_CONFIGS = {
    "helpdesk": {
        # --- Paths (Derived from base paths and dataset name) ---
        "name": "helpdesk", # Added for clarity/potential use in naming
        "raw_path_pattern": RAW_DATA_DIR / "helpdesk_dataset1" / "Export *.csv", # Example subfolder
        "processed_path_template": PROCESSED_DATA_DIR / "helpdesk_dataset1" / "{}_cleaned.csv", # Example subfolder
        "vector_store_path": VECTOR_STORE_DIR / "helpdesk_dataset1_chroma",
        "hyperopt_path": HYPEROPT_DIR / "helpdesk_dataset1",
        "vector_store_collection_name": "helpdesk_summaries_embeddings",

        # --- Data Loading & Schema ---
        "columns_to_load": None, # Load all initially, or specify: ["Summary", "Issue key", ...]
        "id_column": "Issue id",
        "text_column_for_embedding": "cleaned_summary",
        "metadata_columns_to_store": ["Summary", "Issue key", "Assignee", "Status", "Created", "Summary in English", "Description", "Language", "Inward issue link (Relates)"], # Example

        # --- Cleaning Specific Config ---
        "cleaning_steps": ["clean_custom_fields", "filter_language", "filter_related", "clean_description", "clean_summary"], # Example pipeline
        "language_column": "Language",
        "language_filter_value": "EN - English",
        "related_issue_column": "Inward issue link (Relates)",
        "description_column": "Description",
        "summary_column_original": "Summary in English", # Assuming this is the one to clean
        "summary_column_cleaned": "cleaned_summary", # Name of the output column after cleaning
        # Add any regex patterns or specific values needed for cleaning this dataset
        "custom_fields_to_clean": ["customfield_10050", "customfield_10051"], # Example

        # --- Analysis Parameters (Optional Overrides) ---
        "umap_params": DEFAULT_UMAP_PARAMS, # Use defaults or override: {"n_neighbors": 10, ...}
        "hdbscan_params": DEFAULT_HDBSCAN_PARAMS,
        "optimization_params": DEFAULT_OPTIMIZATION_PARAMS,
    },
    "other_dataset": {
        # --- Paths ---
        "name": "other_dataset",
        "raw_path_pattern": RAW_DATA_DIR / "other_dataset2" / "other_data_*.csv", # Example pattern
        "processed_path_template": PROCESSED_DATA_DIR / "other_dataset2" / "{}_cleaned.csv",
        "vector_store_path": VECTOR_STORE_DIR / "other_dataset2_chroma",
        "hyperopt_path": HYPEROPT_DIR / "other_dataset2",
        "vector_store_collection_name": "other_dataset_embeddings",

        # --- Data Loading & Schema ---
        "columns_to_load": ["TicketID", "SubjectLine", "BodyText", "Category", "SubmitDate"], # Example
        "id_column": "TicketID",
        "text_column_for_embedding": "ProcessedBody", # Example cleaned column name
        "metadata_columns_to_store": ["TicketID", "SubjectLine", "Category", "SubmitDate"], # Example

        # --- Cleaning Specific Config ---
        "cleaning_steps": ["clean_body", "clean_subject"], # Different steps
        "language_column": None, # No language filtering
        "language_filter_value": None,
        "related_issue_column": None, # No related issue filtering
        "description_column": "BodyText", # Column to apply generic text cleaning
        "summary_column_original": "SubjectLine", # Column to apply specific subject cleaning
        "summary_column_cleaned": "cleaned_subject", # Output name
        # Add any specific cleaning params for this dataset
        "custom_fields_to_clean": [], # None for this dataset

        # --- Analysis Parameters (Optional Overrides) ---
        "umap_params": DEFAULT_UMAP_PARAMS,
        "hdbscan_params": DEFAULT_HDBSCAN_PARAMS,
        "optimization_params": DEFAULT_OPTIMIZATION_PARAMS,
    }
    # Add configurations for more datasets here...
}


# --- Active Configuration Selection ---

def get_active_config(dataset_name: str = None) -> dict:
    """
    Selects and returns the configuration dictionary for the specified dataset.

    Args:
        dataset_name: The name of the dataset config to retrieve (e.g., "helpdesk").
                      If None, attempts to get from ACTIVE_DATASET env variable,
                      defaulting to the first key in DATASET_CONFIGS.

    Returns:
        The configuration dictionary for the selected dataset.

    Raises:
        ValueError: If the specified or defaulted dataset_name is not found
                    in DATASET_CONFIGS.
    """
    if dataset_name is None:
        # Default to environment variable, then to the first dataset config key
        default_dataset = list(DATASET_CONFIGS.keys())[0] if DATASET_CONFIGS else None
        dataset_name = os.getenv("ACTIVE_DATASET", default_dataset)

    if dataset_name is None:
        raise ValueError("No dataset name provided and no default dataset configured.")

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset name: '{dataset_name}'. "
                         f"Available datasets: {list(DATASET_CONFIGS.keys())}")

    print(f"----- Using configuration for dataset: '{dataset_name}' -----")
    # Potentially perform deep copies or validation here if needed
    return DATASET_CONFIGS[dataset_name]


# --- Helper to create directories if they don't exist ---
def ensure_dir(path: Path):
    """Creates a directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


# --- Example Usage (can be removed or kept for testing) ---
if __name__ == "__main__":
    print(f"Project Base Directory: {BASE_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Raw Data Directory: {RAW_DATA_DIR}")
    print(f"Processed Data Directory: {PROCESSED_DATA_DIR}")
    print(f"Vector Store Directory: {VECTOR_STORE_DIR}")
    print(f"Hyperopt Results Directory: {HYPEROPT_DIR}")

    print("\n--- Loaded Environment Variables ---")
    print(f"Azure OpenAI Key Loaded: {'Yes' if AZURE_OPENAI_API_KEY else 'No'}")
    print(f"Azure OpenAI Endpoint: {AZURE_OPENAI_ENDPOINT}")
    print(f"Azure OpenAI Emb Deployment: {AZURE_OPENAI_EMBEDDING_DEPLOYMENT}")

    print("\n--- Default Parameters ---")
    print(f"Default UMAP: {DEFAULT_UMAP_PARAMS}")
    print(f"Default HDBSCAN: {DEFAULT_HDBSCAN_PARAMS}")

    print("\n--- Dataset Configurations ---")
    print(f"Available Datasets: {list(DATASET_CONFIGS.keys())}")

    try:
        print("\n--- Testing Active Config Selection ---")
        # Test default selection (first key or ACTIVE_DATASET env var)
        active_config_default = get_active_config()
        print(f"Default Active Dataset ('{active_config_default['name']}'): {active_config_default}")

        # Test specific selection
        if "helpdesk" in DATASET_CONFIGS:
            active_config_helpdesk = get_active_config("helpdesk")
            print(f"\nHelpdesk Config: {active_config_helpdesk}")
            # Example accessing a config value
            print(f"Helpdesk ID column: {active_config_helpdesk['id_column']}")
            print(f"Helpdesk Vector Store Path: {active_config_helpdesk['vector_store_path']}")
            # Ensure the path exists (optional - scripts might handle this)
            # ensure_dir(active_config_helpdesk['vector_store_path'])
            # ensure_dir(active_config_helpdesk['hyperopt_path'])

        # Test invalid selection
        try:
            get_active_config("non_existent_dataset")
        except ValueError as e:
            print(f"\nCaught expected error: {e}")

    except ValueError as e:
        print(f"\nError getting active configuration: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}") 