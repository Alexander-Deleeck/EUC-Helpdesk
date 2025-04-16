import pandas as pd
import re
import os
import time  # Import time for timestamping
from pathlib import Path
from openai import AzureOpenAI
from dotenv import load_dotenv

# --- Add imports for type hinting if using hyperopt ---
try:
    from hyperopt import Trials

    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False
# --- End of added imports ---


def save_hyperopt_results(
    results,
    tuning_method: str,
    num_evals: int,
    output_dir: Path,
    filename_prefix: str = "tuning_results",
):
    """
    Saves hyperparameter tuning results (from random search or hyperopt) to a CSV file.

    Args:
        results: The results object (Pandas DataFrame for random search,
                 hyperopt.Trials object for Bayesian search).
        tuning_method: String indicating the method used ('random' or 'bayesian').
        num_evals: The number of evaluations performed.
        output_dir: The Path object representing the directory to save the results.
        filename_prefix: Optional prefix for the output CSV file.
    """
    if results is None:
        print("No tuning results provided to save.")
        return

    try:
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{tuning_method}_{num_evals}evals_{timestamp}.csv"
        filepath = output_dir / filename

        results_df = None

        if tuning_method == "random":
            if isinstance(results, pd.DataFrame):
                results_df = results.copy()
                # Ensure standard columns exist if possible (may vary based on creation)
                expected_cols = [
                    "run_id",
                    "n_neighbors",
                    "n_components",
                    "min_cluster_size",
                    "label_count",
                    "cost",
                ]
                for col in expected_cols:
                    if col not in results_df.columns:
                        results_df[col] = None  # Add missing expected columns as None
            else:
                print(
                    f"Error: Expected a Pandas DataFrame for random search results, but got {type(results)}"
                )
                return

        elif tuning_method == "bayesian":
            if HYPEROPT_AVAILABLE and isinstance(results, Trials):
                # Extract data from hyperopt Trials object into a list of dictionaries
                trial_data = []
                # Use trials.trials for more direct access if trials.results is complex
                for i, trial in enumerate(results.trials):
                    trial_info = {
                        "trial_id": trial.get("tid", i),  # Get trial id
                        "status": trial["result"].get("status", "N/A"),
                        "loss": trial["result"].get("loss", None),
                        "label_count": trial["result"].get("label_count", None),
                        # Parameters need careful extraction based on space definition
                        # Using trial['misc']['vals'] which stores chosen *indices* or values
                        # Converting indices back to values requires space_eval or similar logic,
                        # which is complex here. Easier to get from trial['result']['params'] if stored there
                        "n_neighbors": trial["result"]
                        .get("params", {})
                        .get("n_neighbors", None),
                        "n_components": trial["result"]
                        .get("params", {})
                        .get("n_components", None),
                        "min_cluster_size": trial["result"]
                        .get("params", {})
                        .get("min_cluster_size", None),
                        "eval_time": (
                            (trial["refresh_time"] - trial["book_time"]).total_seconds()
                            if trial.get("refresh_time") and trial.get("book_time")
                            else None
                        ),
                    }
                    trial_data.append(trial_info)

                if not trial_data:
                    print(
                        "Warning: No trial data extracted from Hyperopt Trials object."
                    )
                    return

                results_df = pd.DataFrame(trial_data)
                # Define standard column order
                cols_order = [
                    "trial_id",
                    "n_neighbors",
                    "n_components",
                    "min_cluster_size",
                    "label_count",
                    "loss",
                    "status",
                    "eval_time",
                ]
                # Add any missing columns and reorder
                for col in cols_order:
                    if col not in results_df.columns:
                        results_df[col] = None
                results_df = results_df[cols_order]

            else:
                print(
                    f"Error: Expected a hyperopt Trials object for bayesian search results, but got {type(results)}"
                )
                if not HYPEROPT_AVAILABLE:
                    print("Hyperopt library is not available.")
                return
        else:
            print(
                f"Warning: Unknown tuning_method '{tuning_method}'. Cannot save results."
            )
            return

        # Save the DataFrame
        if results_df is not None and not results_df.empty:
            results_df.to_csv(filepath, index=False)
            print(f"Successfully saved tuning results to: {filepath}")
        elif results_df is not None and results_df.empty:
            print("Warning: Tuning results DataFrame is empty. Nothing saved.")
        # else: case already handled by prior checks

    except Exception as e:
        print(f"Error saving hyperopt results: {e}")
        traceback.print_exc()
