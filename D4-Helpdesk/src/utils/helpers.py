"""
General utility functions used across different modules.
"""

from bs4 import BeautifulSoup
import glob
import os
import re
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import traceback
from datetime import datetime

import spacy
import tiktoken
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

def ensure_dir(path: Path) -> bool:
    """
    Creates a directory and any necessary parent directories if they don't exist.

    Args:
        path: A Path object representing the directory to create.

    Returns:
        True if the directory exists or was created successfully, False otherwise.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ensured: {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}", exc_info=True)
        return False


def save_cluster_assignments(
    assignments_df: pd.DataFrame,
    output_dir: str,
    filename_prefix: str = "cluster_assignments"
) -> str:
    """
    Saves the DataFrame containing data IDs and cluster labels to a CSV file.

    Args:
        assignments_df: DataFrame with columns like 'data_id' and 'cluster_label'.
        output_dir: The directory to save the file in.
        filename_prefix: Prefix for the output filename.

    Returns:
        The full path to the saved file, or None if saving failed.
    """
    if not isinstance(assignments_df, pd.DataFrame):
        logger.error("Invalid input: assignments_df must be a pandas DataFrame.")
        return None
    if 'data_id' not in assignments_df.columns or 'cluster_label' not in assignments_df.columns:
        logger.error("DataFrame must contain 'data_id' and 'cluster_label' columns.")
        return None
    if not output_dir:
        logger.error("Output directory not specified.")
        return None

    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.csv"
        full_path = output_path / filename

        assignments_df.to_csv(full_path, index=False)
        logger.info(f"Cluster assignments saved successfully to: {full_path}")
        return str(full_path)

    except Exception as e:
        logger.error(f"Error saving cluster assignments: {e}", exc_info=True)
        return None

# --- Helper to Extract Best Params ---
# This can live here or within the optimization script if preferred

def get_best_params_from_results(
    results: Union[pd.DataFrame, 'Trials'], # Use quotes for Trials if hyperopt is optional
    tuning_method: str,
    search_space: Dict[str, Any] # Needed for Bayesian space_eval
) -> Optional[Dict[str, Any]]:
    """
    Extracts the best hyperparameter set from optimization results.

    Args:
        results: The results object (DataFrame for random, Trials for Bayesian).
        tuning_method: 'random' or 'bayesian'.
        search_space: The search space definition (required for Bayesian).

    Returns:
        A dictionary containing the best parameters found, or None on error.
    """
    best_params = None
    try:
        if tuning_method == 'random' and isinstance(results, pd.DataFrame) and not results.empty:
            # Assuming 'loss' column exists and lower is better
            best_run = results.loc[results['loss'].idxmin()]
            # Extract parameters based on column names (needs to match objective fn)
            # This is slightly fragile, depends on how results_df is structured
            # Let's assume columns exist like 'n_neighbors', 'n_components', 'min_cluster_size' etc.
            param_keys = [k for k in results.columns if k in search_space] # Find keys matching search space
            best_params = best_run[param_keys].to_dict()
            # Convert numpy types to native Python types if necessary
            best_params = {k: v.item() if isinstance(v, np.generic) else v for k, v in best_params.items()}


        elif tuning_method == 'bayesian':
            # Check if hyperopt is available and results is a Trials object
            from hyperopt import space_eval, Trials # Import locally if needed
            if isinstance(results, Trials) and results.best_trial:
                 # Use space_eval to convert best trial parameters (indices) back to values
                 best_params = space_eval(search_space, results.argmin)

        if best_params:
            logger.info(f"Extracted best parameters: {best_params}")
            return best_params
        else:
            logger.warning("Could not extract best parameters from results.")
            return None

    except Exception as e:
        logger.error(f"Error extracting best parameters: {e}", exc_info=True)
        return None



def find_latest_file(directory: str, pattern: str) -> Optional[str]:
    """Finds the most recently modified file in a directory matching a pattern."""
    try:
        dir_path = Path(directory)
        if not dir_path.is_dir():
            logger.error(f"Directory not found: {directory}")
            return None

        list_of_files = glob.glob(str(dir_path / pattern)) # Use glob for pattern matching
        if not list_of_files:
            logger.warning(f"No files found matching pattern '{pattern}' in '{directory}'.")
            return None

        latest_file = max(list_of_files, key=os.path.getmtime)
        logger.info(f"Found latest file: {latest_file}")
        print(f"\n\n\nLatest file:\n\n{latest_file}\n\n\n\n")
        return latest_file
    except Exception as e:
        logger.error(f"Error finding latest file: {e}", exc_info=True)
        return None

def truncate_text(text: str, max_length: int) -> str:
    """Truncates text to a maximum character length, adding ellipsis."""
    if len(text) > max_length:
        return text[:max_length-3] + "..."
    return text

# --- Optional: Token Counting Helper ---
_tokenizer = None
def get_tokenizer(encoding_name="cl100k_base"): # Encoding used by gpt-4, gpt-3.5-turbo, text-embedding-ada-002
    """Initializes and returns a tiktoken tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        try:
            _tokenizer = tiktoken.get_encoding(encoding_name)
            logger.info(f"Initialized tokenizer with encoding: {encoding_name}")
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer '{encoding_name}': {e}")
            _tokenizer = None # Ensure it stays None if init fails
    return _tokenizer

def count_tokens(text: str) -> int:
    """Counts tokens in a string using the initialized tokenizer."""
    tokenizer = get_tokenizer()
    if tokenizer and text:
        return len(tokenizer.encode(text))
    return 0


def replace_email_addresses(text: str) -> str:
    """Replaces email addresses with generic 'email@domain.com'."""
    email_pattern = re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        re.IGNORECASE
    )
    return email_pattern.sub('email@domain.com', text)


def replace_phone_numbers(text: str) -> str:
    """Replaces phone numbers with generic 'PHONE_NUMBER'."""
    phone_pattern = re.compile(
        r'''
        (?:(?:Tel(?:ephone)?|Phone|Mobile|T|Fax|GSM)\.?\s*[:\-]?\s*)?  # optional prefix
        (?:\+?\(?\d{1,4}\)?\s*(?:\(0\))?\s*)?                 # country code and optional (0)
        (?:\d{1,4}[\s\-]?){2,5}                               # main number body
        ''',
        re.VERBOSE | re.IGNORECASE
    )
    return phone_pattern.sub('PHONE_NUMBER', text)


def replace_person_names(ner_model: spacy.Language, text: str, placeholder: str = "PERSON_NAME") -> str:
    """
    Replaces person names in `text` with `placeholder`.  
    First uses spaCy NER to find PERSON spans, then a regex fallback
    to catch any leftover Title‑Case multi‑word sequences.
    
    Args:
        ner_model: spaCy NER model (e.g. "xx_ent_wiki_sm")
        text: text to replace person names in
        placeholder: placeholder to replace person names with

    Returns:
        text with person names replaced with placeholder
    """
    # regex fallback: sequences of 2+ Title‑Case words
    TITLE_CASE_NAMES = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,})\b')
    
    doc = ner_model(text)
    spans = []
    # collect all PERSON entities
    for ent in doc.ents:
        if ent.label_.upper() in ("PER", "PERSON"):
            spans.append((ent.start_char, ent.end_char))
    # replace in reverse order so indices stay valid
    redacted = text
    for start, end in sorted(spans, reverse=True):
        redacted = redacted[:start] + placeholder + redacted[end:]
    # fallback regex pass
    redacted = TITLE_CASE_NAMES.sub(placeholder, redacted)
    return redacted



def remove_email_formatting(text: str) -> str:
    """
    Removes common markup/formatting from text, including:
      - HTML tags
      - horizontal rules (---, ***, ___)
      - Markdown headings, bold/italic, code blocks and inline code
      - Markdown links & images
      - blockquotes (>), list bullets (-, *, +)
      - leftover empty lines
    """
    # 1) Strip HTML tags
    text = BeautifulSoup(text, "html.parser").get_text(separator="\n")

    # 2) Remove Markdown/Fenced code blocks
    text = re.sub(r"```.+?```", "", text, flags=re.DOTALL)

    # 3) Remove horizontal rules (---, ***, ___) on their own line
    text = re.sub(r"(?m)^[ \t]*([-*_]){3,}[ \t]*$", "", text)

    # 4) Strip Markdown headings
    text = re.sub(r"(?m)^[ \t]{0,3}#{1,6}[ \t]*", "", text)

    # 5) Unwrap inline code and bold/italic, but keep the text
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*",   r"\1", text)
    text = re.sub(r"__([^_]+)__",   r"\1", text)
    text = re.sub(r"_([^_]+)_",     r"\1", text)

    # 6) Remove images and links (leave link text)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)               # images
    text = re.sub(r"\[([^\]]+)\]\((?:.*?)\)", r"\1", text)     # links

    # 7) Strip blockquotes and list bullets at start of lines
    text = re.sub(r"(?m)^[ \t]{0,3}>\s?", "", text)            # blockquotes
    text = re.sub(r"(?m)^[ \t]*[-+*]\s+", "", text)            # bullets

    # 8) Collapse multiple blank lines to one
    text = re.sub(r"\n{2,}", "\n\n", text)

    return text.strip()

def clean_email(email_text: str, ner_model: spacy.Language) -> str:
    """
    Cleans email text by applying:
        1. removal of common formatting and markup.
        2. anonymization of p.i.i via:
            a) replacement of email addresses with generic 'email@domain.com'
            b) replacement of phone numbers with generic 'PHONE_NUMBER'
            c) replacement of person names with generic 'PERSON_NAME'

    Args:
        email_text: The email text to clean.
        ner_model: spaCy NER model (e.g. "xx_ent_wiki_sm")

    Returns:
        cleaned email text
    """
    # 1. remove common formatting and markup
    cleaned_text = remove_email_formatting(email_text)

    # 2. anonymize PII
    cleaned_text = replace_email_addresses(cleaned_text)
    cleaned_text = replace_phone_numbers(cleaned_text)
    cleaned_text = replace_person_names(ner_model, cleaned_text)

    return cleaned_text

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