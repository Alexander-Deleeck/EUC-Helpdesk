"""
Workflow Script 1: Data Cleaning

Reads raw data files for a specified dataset, applies cleaning steps based on
the dataset's configuration, and saves the cleaned data to the processed directory.

Usage:
  python scripts/1_clean_data.py --dataset <dataset_name>

Example:
  python scripts/1_clean_data.py --dataset helpdesk
"""

import argparse
import logging
from pathlib import Path
import sys
import glob
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure the src directory is in the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Import necessary functions from the src library
try:
    from src import config
    from src.data import load, clean
    from src.utils import helpers
except ImportError as e:
    logger.error(f"Error importing src modules: {e}. Make sure the script is run from the project root or src is in PYTHONPATH.")
    sys.exit(1)

def main(dataset_name: str):
    """Main function to execute the data cleaning workflow."""
    logger.info(f"--- Starting Data Cleaning Workflow for dataset: '{dataset_name}' ---")

    # 1. Load Active Configuration
    try:
        active_config = config.get_active_config(dataset_name)
        logger.info(f"Loaded configuration: {active_config['name']}")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # 2. Find Raw Data Files
    raw_path_pattern = str(active_config["raw_path_pattern"])
    raw_file_paths = [Path(p) for p in glob.glob(raw_path_pattern)]

    if not raw_file_paths:
        logger.error(f"No raw data files found matching pattern: {raw_path_pattern}")
        sys.exit(1)
    logger.info(f"Found {len(raw_file_paths)} raw data file(s): {[p.name for p in raw_file_paths]}")

    # 3. Load Raw Data
    try:
        df_raw = load.load_raw_data(
            file_paths=raw_file_paths,
            columns_to_load=active_config.get("columns_to_load")
        )
        if df_raw.empty:
             logger.warning("Loaded DataFrame is empty. Exiting.")
             sys.exit(0)
    except (FileNotFoundError, ValueError, Exception) as e:
        logger.error(f"Failed to load raw data: {e}", exc_info=True)
        sys.exit(1)

    # 4. Apply Cleaning Steps (Dynamically based on config)
    df_cleaned = df_raw.copy()
    cleaning_steps = active_config.get("cleaning_steps", [])
    logger.info(f"Applying cleaning steps: {cleaning_steps}")

    for step in cleaning_steps:
        logger.info(f"--> Running cleaning step: {step}")
        try:
            if step == "clean_custom_fields":
                df_cleaned = clean.clean_custom_fields(
                    df_cleaned,
                    fields_to_rename=active_config.get("custom_fields_rename_map"),
                    fields_to_remove=active_config.get("custom_fields_to_remove")
                )
            elif step == "filter_language":
                df_cleaned = clean.filter_by_column_value(
                    df_cleaned,
                    column_name=active_config.get("language_column"),
                    filter_value=active_config.get("language_filter_value")
                )
            elif step == "filter_related":
                df_cleaned = clean.filter_related_issues(
                    df_cleaned,
                    related_issue_column=active_config.get("related_issue_column")
                )
            elif step == "clean_description":
                output_col_name = active_config.get("cleaned_description_column", "cleaned_description")
                df_cleaned = clean.clean_text_column(
                    df_cleaned,
                    input_col=active_config.get("description_column"),
                    output_col=output_col_name
                )
            elif step == "clean_summary":
                df_cleaned = clean.clean_and_filter_summary(
                    df_cleaned,
                    input_col=active_config.get("summary_column_original"),
                    output_col=active_config.get("summary_column_cleaned", "cleaned_summary")
                )
            else:
                logger.warning(f"Unknown cleaning step defined in config: '{step}'. Skipping.")

            logger.info(f"Shape after step '{step}': {df_cleaned.shape}")
            if df_cleaned.empty:
                 logger.warning(f"DataFrame became empty after step '{step}'. Stopping cleaning.")
                 break

        except Exception as e:
             logger.error(f"Error during cleaning step '{step}': {e}", exc_info=True)
             sys.exit(1)

    # 5. Save Cleaned Data
    if df_cleaned.empty:
        logger.warning("Final cleaned DataFrame is empty. No file will be saved.")
    else:
        # Construct output path - Use the *template* from config and format it
        processed_template = active_config.get("processed_path_template")
        if not processed_template:
            logger.error("Config key 'processed_path_template' is missing or empty. Cannot determine output path.")
            sys.exit(1)

        try:
            # Example formatting - adjust if template needs more variables
            output_path = Path(str(processed_template).format(active_config['name']))
        except KeyError as e:
             logger.error(f"Error formatting 'processed_path_template': Missing key {e} in config or template string.")
             logger.error(f"Template string: {processed_template}")
             sys.exit(1)
        except Exception as e:
             logger.error(f"Error constructing output path from template '{processed_template}': {e}")
             sys.exit(1)

        logger.info(f"Attempting to save cleaned data to: {output_path}")
        save_success = helpers.save_dataframe(df_cleaned, output_path)

        if save_success:
            logger.info(f"Cleaned data saved successfully (Shape: {df_cleaned.shape}).")
        else:
            logger.error("Failed to save cleaned data.")

    logger.info(f"--- Data Cleaning Workflow for '{dataset_name}' Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the data cleaning workflow for a specific dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=f"Name of the dataset configuration to use (e.g., {list(config.DATASET_CONFIGS.keys())})."
    )

    args = parser.parse_args()

    # Dataset name validation happens within get_active_config
    main(args.dataset)
