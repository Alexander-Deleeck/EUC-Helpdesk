"""
Functions for cleaning and filtering datasets based on configuration.
"""

import pandas as pd
import re
from typing import List, Dict, Any, Optional, Tuple

# --- Regex Patterns (Compiled for efficiency) ---
HARDCODED_FIELDS_TO_REMOVE = ["Custom field (Classification (Calculated))"] # Example

HEADER_PATTERN = re.compile(r"^\s*\*?\s*(From|Sent|To|Subject|Importance|CC)\s*:", flags=re.IGNORECASE)
TABLE_PATTERN = re.compile(r"^\s*\|")
# Improved sentence split pattern to better handle abbreviations, URLs etc.
SENTENCE_SPLIT_PATTERN = re.compile(
    r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|!)\s'
)
UNWANTED_SUMMARY_PATTERNS = re.compile(
    r"""(?ix)                # Case-insensitive and verbose mode
    (?:
        ^hd\s*cannot\s*answer$ |              # Exact match for "HD cannot answer"
        hd[0-9]*\s*cannot\s*(?:help|answer) | # Variations with HD1, HD2, etc.
        not\s+related\s+to\s+op |             # Not related to OP
        more\s+info(?:rmation)?\s+(?:needed|required) |  # More info variations
        not\s+enough\s+info(?:rmation)?\s+for\s+hd[0-9]* | # Not enough info for HD1, etc.
        insufficient\s+info(?:rmation)?\s+(?:provided|given|available) | # Additional variations
        cannot\s+(?:be\s+)?(?:resolved|answered)\s+(?:by|with)\s+(?:the\s+)?hd[0-9]* # Cannot be resolved by HD
    )""",
    re.IGNORECASE | re.VERBOSE
)
COLOR_TAG_PATTERN = re.compile(r"\{color(?::[^\}]+)?\}", flags=re.IGNORECASE)
ORIGINAL_MESSAGE_PATTERN = re.compile(r"-----Original Message-----")
EMAIL_PREFIX_PATTERN = re.compile(r"(?i)(?:subject|fwd|fw|re):\s*")
WHITESPACE_PATTERN = re.compile(r"\s+")
X000D_PATTERN = re.compile(r"_x000D_")


# === Specific Cleaning Functions (Helpdesk Dataset Oriented) ===

def clean_custom_fields(df: pd.DataFrame, fields_to_rename: Optional[Dict[str, str]] = None, fields_to_remove: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Rename specific custom fields based on provided mapping and remove others.
    This function is somewhat specific to the initial dataset's naming conventions.

    Args:
        df: Input DataFrame.
        fields_to_rename: Optional dictionary mapping old field names to new names.
                          If None, uses a default mapping.
        fields_to_remove: Optional list of field names to remove.
                          If None, uses a default list.

    Returns:
        DataFrame with cleaned custom field names.
    """
    if fields_to_remove is None:
        fields_to_remove = HARDCODED_FIELDS_TO_REMOVE

    # Dynamic identification of "Custom field" columns might be less robust
    # than specifying them in the config. This approach tries to find them.
    potential_custom_fields = {
        old_name: old_name.replace("Custom field (", "").replace(")", "")
        for old_name in df.columns if isinstance(old_name, str) and "Custom field (" in old_name
    }

    # Remove specified fields before renaming
    for field in fields_to_remove:
        potential_custom_fields.pop(field, None) # Remove safely
        if field in df.columns:
            df = df.drop(columns=[field])
            print(f" -- Dropped column: {field}")

    # Apply specific renames from config or defaults
    if fields_to_rename:
        # Ensure only existing columns are renamed
        actual_renames = {k: v for k, v in fields_to_rename.items() if k in df.columns}
        df = df.rename(columns=actual_renames)
        print(f" -- Applied specific renames: {actual_renames}")
    else:
        # Apply default derived renames if no specific map provided
        default_renames = {k: v for k, v in potential_custom_fields.items() if k in df.columns}
        df = df.rename(columns=default_renames)
        print(f" -- Applied default custom field renames: {default_renames}")

    return df


# === Filtering Functions (Configurable) ===

def filter_by_column_value(df: pd.DataFrame, column_name: Optional[str], filter_value: Any) -> pd.DataFrame:
    """
    Filter a DataFrame based on a specific value in a given column.
    Skips filtering if column_name is None or not in the DataFrame.

    Args:
        df: Input DataFrame.
        column_name: Name of the column to filter on.
        filter_value: The value to keep (rows where df[column_name] == filter_value).

    Returns:
        Filtered DataFrame.
    """
    if column_name and column_name in df.columns:
        original_rows = len(df)
        df_filtered = df[df[column_name] == filter_value].copy()
        print(f" -- Filtered by '{column_name}' == '{filter_value}'. Rows remaining: {len(df_filtered)} ({len(df_filtered)/original_rows:.1%} of original)")
        return df_filtered
    elif column_name:
        print(f" -- Warning: Column '{column_name}' not found for filtering. Skipping.")
    else:
        print(" -- Skipping filtering by column value (column name not provided).")
    return df


def filter_related_issues(df: pd.DataFrame, related_issue_column: Optional[str]) -> pd.DataFrame:
    """
    Filter out rows where the related issue column is not NA (i.e., keep only originals).
    Skips filtering if related_issue_column is None or not in the DataFrame.

    Args:
        df: Input DataFrame.
        related_issue_column: Name of the column indicating related issues.

    Returns:
        Filtered DataFrame.
    """
    if related_issue_column and related_issue_column in df.columns:
        original_rows = len(df)
        df_filtered = df[df[related_issue_column].isna()].copy()
        print(f" -- Filtered out related issues (column '{related_issue_column}'). Rows remaining: {len(df_filtered)} ({len(df_filtered)/original_rows:.1%} of original)")
        return df_filtered
    elif related_issue_column:
        print(f" -- Warning: Related issue column '{related_issue_column}' not found. Skipping filter.")
    else:
        print(" -- Skipping related issue filtering (column name not provided).")
    return df


# === Generic Text Cleaning Utilities ===

def is_header_or_table(paragraph: str) -> bool:
    """Check if a paragraph is a header or table line."""
    return bool(TABLE_PATTERN.match(paragraph) or HEADER_PATTERN.match(paragraph))


def check_signoff(text_lower: str) -> bool:
    """Check if a lowercased text contains a signoff pattern."""
    # Keep common signoffs
    signoff_patterns = {
        "kind regards", "best regards", "best wishes", "sincerely",
        "thank you in advance", "regards", "yours sincerely", "thanks and regards"
    }
    return any(pattern in text_lower for pattern in signoff_patterns)


def check_privacy_notice(text_lower: str) -> bool:
    """Check if a lowercased text contains a privacy notice pattern."""
    # Keep common privacy/confidentiality phrases
    privacy_patterns = {
        "this e-mail and any attachments", "this email and any attachments",
        "this message and any attachments", "this communication is confidential",
        "this email is confidential", "confidentiality notice", "privacy notice",
        "privileged/confidential information", "if you are not the intended recipient",
        "if you have received this email in error", "this email transmission is intended only",
        "disclaimer:", "legal disclaimer:", "confidentiality disclaimer"
    }
    return any(pattern in text_lower for pattern in privacy_patterns)


def process_paragraph(paragraph: str, found_signoff: bool) -> Tuple[str, bool, bool]:
    """
    Process a single paragraph, checking for unwanted content.

    Returns: (processed_text, found_privacy, found_signoff)
    """
    paragraph_stripped = paragraph.strip()
    if not paragraph_stripped:
        return "", False, found_signoff

    if is_header_or_table(paragraph_stripped):
        return "", False, found_signoff

    paragraph_lower = paragraph_stripped.lower()

    # Check for privacy notices first (often at the end)
    if check_privacy_notice(paragraph_lower):
        return "", True, found_signoff

    # If we already found a signoff in previous paragraphs, ignore subsequent ones
    if found_signoff:
        return "", False, True

    # Check for signoffs in the current paragraph
    if check_signoff(paragraph_lower):
        # Check if the whole paragraph is just a signoff, or if it contains other text
        # This simple check assumes signoffs are usually short. More complex logic could be added.
        if len(paragraph_stripped) < 50: # Heuristic threshold
             return "", False, True
        # If paragraph is long, maybe the signoff is embedded? Process sentences.
        pass # Fall through to sentence processing

    # Process sentences if no paragraph-level match triggered exit
    sentences = SENTENCE_SPLIT_PATTERN.split(paragraph_stripped)
    filtered_sentences = []
    sentence_found_signoff = False

    for sentence in sentences:
        sentence_stripped = sentence.strip()
        if not sentence_stripped:
            continue

        sentence_lower = sentence_stripped.lower()

        if check_privacy_notice(sentence_lower):
            return "", True, found_signoff # Privacy notice stops everything

        if sentence_found_signoff:
            continue # Skip sentences after a signoff within the paragraph

        if check_signoff(sentence_lower):
            sentence_found_signoff = True
            continue # Don't include the signoff sentence itself

        filtered_sentences.append(sentence_stripped)

    if filtered_sentences:
        # Rejoin sentences, ensuring correct spacing and final punctuation
        processed_text = '. '.join(filtered_sentences)
        # Add trailing period if missing (handle ?, ! as well if needed)
        if processed_text and processed_text[-1] not in ('.', '!', '?'):
             processed_text += '.'
        return processed_text, False, sentence_found_signoff
    else:
        # Return paragraph signoff status if no sentences kept but signoff detected
        return "", False, sentence_found_signoff


def filter_email_body(text: Optional[str]) -> str:
    """
    Clean email body content by removing headers, signoffs, and privacy notices.
    Iterates through paragraphs/lines.
    """
    if not isinstance(text, str):
        return "" # Return empty string for non-string input

    paragraphs = text.splitlines()
    filtered_content = []
    found_signoff = False
    found_privacy = False

    for paragraph in paragraphs:
        processed_text, paragraph_found_privacy, paragraph_found_signoff = process_paragraph(
            paragraph, found_signoff
        )

        if paragraph_found_privacy:
            found_privacy = True
            break # Stop processing once a privacy notice is found

        found_signoff = found_signoff or paragraph_found_signoff

        if processed_text:
            filtered_content.append(processed_text)

    # If privacy notice was found mid-way, the content before it is returned
    return ' '.join(filtered_content)


# === Main Cleaning Orchestration Functions (Configurable) ===

def clean_text_column(df: pd.DataFrame, input_col: str, output_col: str) -> pd.DataFrame:
    """
    Applies generic email body cleaning to a specified column.

    Args:
        df: Input DataFrame.
        input_col: Name of the column containing the text to clean.
        output_col: Name of the new column to store the cleaned text.

    Returns:
        DataFrame with the new cleaned text column.
    """
    if input_col not in df.columns:
        print(f" -- Warning: Input column '{input_col}' not found for cleaning. Skipping.")
        return df

    print(f" -- Cleaning text in column '{input_col}' -> '{output_col}'")
    # Apply the comprehensive filter_email_body function
    df[output_col] = df[input_col].apply(filter_email_body)

    # Additional steps from original clean_description_content
    if output_col in df.columns:
         df[output_col] = df[output_col].str.replace(COLOR_TAG_PATTERN, "", regex=True)
         df[output_col] = df[output_col].str.replace(X000D_PATTERN, "", regex=True)
         df[output_col] = df[output_col].str.replace(ORIGINAL_MESSAGE_PATTERN, "", regex=True)
         df[output_col] = df[output_col].str.replace(WHITESPACE_PATTERN, " ", regex=True).str.strip()
         print(f" -- Applied additional pattern removal and whitespace cleanup to '{output_col}'")

    return df


def clean_and_filter_summary(df: pd.DataFrame, input_col: str, output_col: str) -> pd.DataFrame:
    """
    Cleans a summary column (removing prefixes, lowercasing) and filters out rows
    based on unwanted patterns (e.g., "HD cannot answer").

    Args:
        df: Input DataFrame.
        input_col: Name of the column containing the original summary.
        output_col: Name of the new column to store the cleaned summary.

    Returns:
        Filtered DataFrame with the new cleaned summary column.
    """
    if input_col not in df.columns:
        print(f" -- Warning: Summary input column '{input_col}' not found. Skipping summary cleaning/filtering.")
        return df

    print(f" -- Cleaning summary column '{input_col}' -> '{output_col}'")
    # 1. Clean the summary content
    def _clean_summary(text):
        if not isinstance(text, str):
            return ""
        text = EMAIL_PREFIX_PATTERN.sub("", text)
        text = WHITESPACE_PATTERN.sub(" ", text).strip().lower()
        return text

    df[output_col] = df[input_col].apply(_clean_summary)

    # 2. Filter based on unwanted patterns
    original_rows = len(df)
    mask = df[output_col].str.contains(UNWANTED_SUMMARY_PATTERNS, regex=True, na=False)
    df_filtered = df[~mask].copy() # Keep rows that DO NOT match the unwanted patterns
    print(f" -- Filtered summaries by unwanted patterns. Rows remaining: {len(df_filtered)} ({len(df_filtered)/original_rows:.1%} of original)")

    return df_filtered


# --- Example Pipeline Function (Illustrative) ---
# The actual pipeline will be run from scripts/1_clean_data.py

# def run_cleaning_pipeline(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
#     """
#     Runs a series of cleaning and filtering steps based on the provided config.
#     This is a placeholder; the real logic lives in the script.
#     """
#     print("\nStarting cleaning pipeline...")
#     df_cleaned = df.copy()

#     # Example sequence based on config keys:
#     if "clean_custom_fields" in config.get("cleaning_steps", []):
#         df_cleaned = clean_custom_fields(df_cleaned, config.get("custom_fields_rename_map"), config.get("custom_fields_to_remove"))

#     if "filter_language" in config.get("cleaning_steps", []):
#         df_cleaned = filter_by_column_value(df_cleaned, config.get("language_column"), config.get("language_filter_value"))

#     if "filter_related" in config.get("cleaning_steps", []):
#         df_cleaned = filter_related_issues(df_cleaned, config.get("related_issue_column"))

#     if "clean_description" in config.get("cleaning_steps", []):
#         df_cleaned = clean_text_column(df_cleaned, config.get("description_column"), "cleaned_description") # Example output col name

#     if "clean_summary" in config.get("cleaning_steps", []):
#         # Note: This function both cleans AND filters
#         df_cleaned = clean_and_filter_summary(df_cleaned, config.get("summary_column_original"), config.get("summary_column_cleaned"))

#     print("Cleaning pipeline finished.")
#     return df_cleaned 