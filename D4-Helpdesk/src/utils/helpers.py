"""
General utility functions used across different modules.
"""

import time
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Hyperopt Availability Check (needed for type hinting if used) ---
try:
    from hyperopt import Trials
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False
    # Define Trials as object to avoid NameError if hyperopt is not installed
    # This allows type hinting without crashing if the library isn't present.
    # Code using Trials will still fail if HYPEROPT_AVAILABLE is False.
    class Trials:
        pass


def save_hyperopt_results(
    results: Union[pd.DataFrame, Trials, None],
    tuning_method: str,
    num_evals: int,
    output_dir: Path, # Directory path comes from config (e.g., config["hyperopt_path"])
    filename_prefix: str = "tuning_results"
) -> Optional[Path]:
    """
    Saves hyperparameter tuning results (from random search or hyperopt) to a CSV file
    in the specified output directory.

    Args:
        results: The results object (Pandas DataFrame for random search,
                 hyperopt.Trials object for Bayesian search, or None).
        tuning_method: String indicating the method used ('random' or 'bayesian').
        num_evals: The number of evaluations performed.
        output_dir: The Path object representing the directory to save the results
                    (e.g., config["hyperopt_path"]).
        filename_prefix: Optional prefix for the output CSV file.

    Returns:
        The Path object of the saved file, or None if saving failed or no results.
    """
    if results is None:
        logger.warning("No tuning results provided to save.")
        return None

    try:
        # Ensure output directory exists
        logger.info(f"Ensuring hyperopt results directory exists: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{tuning_method}_{num_evals}evals_{timestamp}.csv"
        filepath = output_dir / filename

        results_df = None

        if tuning_method == "random":
            if isinstance(results, pd.DataFrame):
                results_df = results.copy()
                # Basic validation/standardization can happen here if needed
            else:
                logger.error(f"Expected a Pandas DataFrame for random search results, but got {type(results)}")
                return None

        elif tuning_method == "bayesian":
            if HYPEROPT_AVAILABLE and isinstance(results, Trials):
                trial_data = []
                for i, trial in enumerate(results.trials):
                    # Extract parameters safely using .get()
                    params = trial["result"].get("params", {})
                    trial_info = {
                        "trial_id": trial.get("tid", i),
                        "status": trial["result"].get("status", "N/A"),
                        "loss": trial["result"].get("loss", None),
                        "label_count": trial["result"].get("label_count", None),
                        "n_neighbors": params.get("n_neighbors", None),
                        "n_components": params.get("n_components", None),
                        "min_cluster_size": params.get("min_cluster_size", None),
                        # Add other params if they are in the search space
                        "eval_time": (
                            (trial["refresh_time"] - trial["book_time"]).total_seconds()
                            if trial.get("refresh_time") and trial.get("book_time")
                            else None
                        ),
                    }
                    trial_data.append(trial_info)

                if not trial_data:
                    logger.warning("No trial data extracted from Hyperopt Trials object.")
                    return None

                results_df = pd.DataFrame(trial_data)
                # Define preferred column order
                cols_order = [
                    "trial_id", "status", "loss", "label_count",
                    "n_neighbors", "n_components", "min_cluster_size", # Match common params
                    "eval_time",
                ]
                # Add any missing columns from the order and reorder
                present_cols = [col for col in cols_order if col in results_df.columns]
                missing_ordered_cols = [col for col in cols_order if col not in results_df.columns]
                # Add any other columns present in df but not in the preferred order
                other_cols = [col for col in results_df.columns if col not in cols_order]
                final_col_order = present_cols + missing_ordered_cols + other_cols

                for col in missing_ordered_cols:
                     results_df[col] = None # Add missing as None
                results_df = results_df[final_col_order]
                results_df = results_df.sort_values(by="loss", ascending=True) # Sort by best loss

            elif not HYPEROPT_AVAILABLE:
                 logger.error("Cannot process Bayesian results: hyperopt library is not available.")
                 return None
            else:
                logger.error(f"Expected a hyperopt Trials object for bayesian search results, but got {type(results)}")
                return None
        else:
            logger.warning(f"Unknown tuning_method '{tuning_method}'. Cannot save results.")
            return None

        # Save the DataFrame
        if results_df is not None and not results_df.empty:
            results_df.to_csv(filepath, index=False, encoding='utf-8-sig')
            logger.info(f"Successfully saved tuning results to: {filepath}")
            return filepath
        elif results_df is not None:
            logger.warning("Tuning results DataFrame is empty. Nothing saved.")
            return None
        else:
            # Should have been caught earlier, but as a safeguard
            logger.error("Results DataFrame could not be created. Nothing saved.")
            return None

    except OSError as e:
         logger.error(f"Failed to create directory or save file at {filepath}: {e}", exc_info=True)
         return None
    except Exception as e:
        logger.error(f"Error saving hyperopt results: {e}", exc_info=True)
        return None


def format_metadata_for_display(
    metadata_dict: Dict[str, Any],
    relevant_fields: List[str]
) -> str:
    """
    Formats a metadata dictionary into a readable string for display.

    Only includes fields present in the relevant_fields list and the dictionary,
    and whose values are not None or NaN.

    Args:
        metadata_dict: The dictionary containing metadata key-value pairs.
        relevant_fields: A list of keys (column names) considered relevant for display.

    Returns:
        A formatted string (newline-separated key: value pairs).
    """
    if not metadata_dict:
        return "(No metadata available)"

    formatted_lines = []
    for field in relevant_fields:
        if field in metadata_dict:
            value = metadata_dict[field]
            # Check for None and NaN (common in pandas data)
            if value is not None and not (isinstance(value, float) and pd.isna(value)):
                formatted_lines.append(f"- {field}: {value}")

    if not formatted_lines:
        return "(No relevant metadata found)"

    return "\n".join(formatted_lines)


# --- Add other potential helpers below ---

# Example: Function to save DataFrames safely
def save_dataframe(
    df: pd.DataFrame,
    output_path: Path,
    index: bool = False,
    encoding: str = 'utf-8-sig'
) -> bool:
    """
    Safely saves a pandas DataFrame to a CSV file, creating parent directories.

    Args:
        df: The DataFrame to save.
        output_path: The full Path object for the output file.
        index: Whether to write the DataFrame index as a column.
        encoding: The file encoding to use.

    Returns:
        True if successful, False otherwise.
    """
    if df is None:
        logger.warning(f"DataFrame is None. Cannot save to {output_path}.")
        return False
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=index, encoding=encoding)
        logger.info(f"DataFrame saved successfully to: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save DataFrame to {output_path}: {e}", exc_info=True)
        return False

# Example Usage (for testing)
# if __name__ == "__main__":
#     print("--- Testing Helpers --- ")

#     # Test format_metadata
#     print("\nTesting format_metadata...")
#     meta1 = {'Issue key': 'ABC-123', 'Status': 'Open', 'Assignee': None, 'Priority': 'High', 'Extra': 'Field'}
#     relevant = ['Issue key', 'Status', 'Priority', 'Assignee', 'MissingField']
#     formatted1 = format_metadata_for_display(meta1, relevant)
#     print(f"Metadata 1:\n{formatted1}")

#     meta2 = {'TicketID': 456, 'Category': 'Hardware', 'Submitter': 'UserX'}
#     relevant2 = ['TicketID', 'Category', 'Submitter']
#     formatted2 = format_metadata_for_display(meta2, relevant2)
#     print(f"\nMetadata 2:\n{formatted2}")

#     meta3 = {'A': 1, 'B': None}
#     formatted3 = format_metadata_for_display(meta3, ['A', 'B', 'C'])
#     print(f"\nMetadata 3:\n{formatted3}")

#     # Test save_dataframe (creates a dummy file)
#     print("\nTesting save_dataframe...")
#     dummy_df_path = Path("./temp_test_output/dummy_df.csv")
#     dummy_df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
#     success = save_dataframe(dummy_df, dummy_df_path, index=True)
#     if success and dummy_df_path.is_file():
#         print(f"Dummy DataFrame saved successfully to {dummy_df_path}")
#         # Clean up the dummy file/dir
#         try:
#             dummy_df_path.unlink()
#             dummy_df_path.parent.rmdir()
#         except OSError:
#             pass # Ignore cleanup errors
#     else:
#         print("Failed to save dummy DataFrame.")

#     # Test save_hyperopt_results (creates dummy file)
#     print("\nTesting save_hyperopt_results...")
#     dummy_results_dir = Path("./temp_test_output/hyperopt")
#     dummy_random_results = pd.DataFrame({
#         'run_id': [1, 2],
#         'n_neighbors': [10, 20],
#         'n_components': [5, 5],
#         'min_cluster_size': [15, 20],
#         'label_count': [8, 12],
#         'cost': [0.45, 0.32]
#     })
#     saved_path_random = save_hyperopt_results(dummy_random_results, "random", 2, dummy_results_dir, "test_prefix")
#     if saved_path_random and saved_path_random.is_file():
#         print(f"Random results saved to: {saved_path_random}")
#         try: saved_path_random.unlink() # cleanup
#         except OSError: pass
#     else:
#         print("Failed to save random results.")

#     # Simulate Bayesian results (if hyperopt installed)
#     if HYPEROPT_AVAILABLE:
#         # Need to construct a dummy Trials object or use a simplified dict/list for testing
#         # This is complex; skipping full Trials simulation for basic helper test
#         print("Skipping Bayesian save test (requires complex Trials object mock).")
#     else:
#         print("Skipping Bayesian save test (hyperopt not installed).")

#     # Clean up dummy dir
#     try:
#         dummy_results_dir.rmdir()
#         dummy_results_dir.parent.rmdir()
#     except OSError:
#         pass

#     print("\n--- Helpers tests finished ---") 