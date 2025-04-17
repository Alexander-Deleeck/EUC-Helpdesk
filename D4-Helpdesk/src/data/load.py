"""
Functions for loading raw datasets based on configuration.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional


def load_raw_data(file_paths: List[Path], columns_to_load: Optional[List[str]] = None, sep: str = ';') -> pd.DataFrame:
    """
    Load and concatenate data from multiple CSV files.

    Args:
        file_paths: A list of Path objects pointing to the CSV files to load.
        columns_to_load: Optional list of column names to load. If None, loads all columns.
                         Uses the configuration from the calling script.
        sep: The separator used in the CSV files.

    Returns:
        A pandas DataFrame containing the concatenated data from all files.

    Raises:
        FileNotFoundError: If any file in file_paths does not exist.
        ValueError: If file_paths is empty.
        Exception: For issues during CSV parsing.
    """
    if not file_paths:
        raise ValueError("No file paths provided for loading data.")

    all_dfs = []
    print(f"Loading data from {len(file_paths)} file(s)...")
    for file_path in file_paths:
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        try:
            print(f" -- Loading: {file_path.name}")
            # Use 'low_memory=False' to potentially avoid dtype warnings with mixed types
            df = pd.read_csv(
                file_path,
                sep=sep,
                header=0,
                usecols=columns_to_load,
                encoding='utf-8', # Common encoding, adjust if needed
                low_memory=False
            )
            all_dfs.append(df)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            raise # Re-raise the exception to halt execution if a file fails

    if not all_dfs:
        print("Warning: No data loaded. Returning empty DataFrame.")
        return pd.DataFrame(columns=columns_to_load if columns_to_load else None)

    # Concatenate all dataframes
    concatenated_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Successfully loaded and concatenated data. Total rows: {len(concatenated_df)}")

    # Optional: Log columns loaded if specific columns were requested
    if columns_to_load:
        print(f"Loaded columns: {concatenated_df.columns.tolist()}")
    else:
        print(f"Loaded all {len(concatenated_df.columns)} columns found in files.")

    return concatenated_df

# Example Usage (within src/data/load.py, for testing)
# You would typically call this from a script in the scripts/ directory

# if __name__ == "__main__":
#     # This is an example, paths need to be adjusted or passed correctly
#     from src.config import DATA_DIR # Assuming config.py is set up
#     example_dir = DATA_DIR / "raw" / "helpdesk_dataset1" # Example path
#     example_files = list(example_dir.glob("Export *.csv"))
#     example_cols = ["Summary", "Issue key", "Issue id", "Description", "Language", "Inward issue link (Relates)"] # Example columns

#     if not example_files:
#         print(f"No example files found in {example_dir}")
#     else:
#         try:
#             print("\n--- Loading specific columns ---")
#             df_loaded = load_raw_data(example_files, columns_to_load=example_cols)
#             print("Loaded DataFrame head:")
#             print(df_loaded.head())
#             print(f"\nShape: {df_loaded.shape}")

#             print("\n--- Loading all columns ---")
#             df_all_cols = load_raw_data(example_files)
#             print("Loaded DataFrame head (all columns):")
#             print(df_all_cols.head())
#             print(f"\nShape: {df_all_cols.shape}")

#         except FileNotFoundError as e:
#             print(f"Error: {e}")
#         except ValueError as e:
#             print(f"Error: {e}")
#         except Exception as e:
#             print(f"An unexpected error occurred: {e}") 