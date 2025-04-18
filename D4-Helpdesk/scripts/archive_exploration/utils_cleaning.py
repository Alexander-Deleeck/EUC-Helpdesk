import pandas as pd
import re
import os
from pathlib import Path
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()


def clean_custom_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the custom fields in the dataframe.
    """
    # Get the custom fields
    custom_fields = {
        old_name: new_name
        for old_name, new_name in zip(
            [col for col in df.columns if "Custom field" in str(col)],
            [
                col.replace("Custom field (", "").replace(")", "")
                for col in df.columns
                if "Custom field" in str(col)
            ],
        )
    }

    custom_fields.pop("Custom field (Classification (Calculated))")

    # Rename the custom fields
    custom_fields["Component/s"] = "Components"    
    custom_fields["Custom field (Classification (Calculated))"] = "Classification (Calculated)"
    return df.rename(columns=custom_fields)


# Compile patterns once at module level for efficiency
HEADER_PATTERN = re.compile(r"^\s*\*?\s*(From|Sent|To|Subject|Subject|Importance|CC)\s*:", flags=re.IGNORECASE)
TABLE_PATTERN = re.compile(r"^\s*\|")
SENTENCE_SPLIT_PATTERN = re.compile(
    r'(?<!Mr)(?<!Ms)(?<!Mrs)(?<!Dr)(?<!etc)(?<!i\.e)(?<!e\.g)(?<!com)(?<!eu)(?<!org)(?<!gov)(?<!edu)\. +'
)

def is_header_or_table(paragraph):
    """Check if a paragraph is a header or table line."""
    return bool(TABLE_PATTERN.match(paragraph) or HEADER_PATTERN.match(paragraph))


def check_signoff(paragraph):
    """Check if a paragraph or sentence contains a signoff pattern."""
    signoff_patterns = {
        "kind regards",
        "best regards",
        "best wishes",
        "sincerely",
        "thank you in advance",
        "regards",
        "yours sincerely",
        "thanks and regards"
    }
    return any(pattern in paragraph.lower() for pattern in signoff_patterns)


def check_privacy_notice(paragraph):
    """Check if a paragraph or sentence contains a privacy notice pattern."""
    privacy_patterns = {
        "this e-mail and any attachments",
        "this email and any attachments",
        "this message and any attachments",
        "this communication is confidential",
        "this email is confidential",
        "confidentiality notice",
        "privacy notice",
        "privileged/confidential information",
        "if you are not the intended recipient",
        "if you have received this email in error",
        "this email transmission is intended only",
        "disclaimer:",
        "legal disclaimer:",
        "confidentiality disclaimer"
    }
    return any(pattern in paragraph.lower() for pattern in privacy_patterns)


def process_paragraph(paragraph, found_signoff=False):
    """
    Process a single paragraph, splitting into sentences and checking each sentence.
    Returns (processed_text, found_privacy, found_signoff)
    """
    if not paragraph.strip():
        return "", False, found_signoff

    if is_header_or_table(paragraph):
        return "", False, found_signoff

    # First check at paragraph level
    if check_privacy_notice(paragraph):
        return "", True, found_signoff

    if check_signoff(paragraph):
        return "", False, True

    if found_signoff:
        return "", False, True

    # Process sentences within the paragraph
    sentences = SENTENCE_SPLIT_PATTERN.split(paragraph)
    filtered_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if check_privacy_notice(sentence):
            return "", True, found_signoff

        if check_signoff(sentence):
            return "", False, True

        filtered_sentences.append(sentence)

    if filtered_sentences:
        # Rejoin sentences with proper punctuation
        paragraph_text = '. '.join(filtered_sentences)
        if not paragraph_text.endswith('.'):
            paragraph_text += '.'
        return paragraph_text, False, found_signoff

    return "", False, found_signoff


def filter_email_content(text):
    """
    Clean email content by removing headers, signoffs, and privacy notices in a single pass.
    Uses modular operations while maintaining efficient single-pass processing.
    """
    if not isinstance(text, str):
        return text

    paragraphs = text.splitlines()
    filtered_content = []
    found_signoff = False
    
    for paragraph in paragraphs:
        processed_text, found_privacy, found_signoff = process_paragraph(
            paragraph, found_signoff
        )
        
        if found_privacy:
            break
            
        if processed_text:
            filtered_content.append(processed_text)
    
    return ' '.join(filtered_content)


def create_embedding_client():
    client = AzureOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )
    print("loaded client")
    return client


def generate_embeddings(to_embed, client, embedding_model_name):
    list_text = list(to_embed.values())
    list_text_response = client.embeddings.create(
        input=list_text, model=embedding_model_name
    )

    for idx, key in enumerate(to_embed.keys()):
        to_embed[key] = list_text_response.data[idx].embedding
    return to_embed


def clean_description_content(text):
    """
    Cleans email content by removing various unwanted elements.
    """
    if not isinstance(text, str):
        return text

    # 1. Remove color tags
    text = re.sub(r"\{color(?::[^\}]+)?\}", "", text, flags=re.IGNORECASE)

    # 2. Apply content filtering
    text = filter_email_content(text)

    # 3. Remove unwanted artifacts
    text = re.sub(r"_x000D_", "", text)
    text = re.sub(r"-----Original Message-----", "", text)

    # 4. Clean up whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Add these patterns at module level with other patterns
UNWANTED_SUMMARY_PATTERNS = re.compile(
    r"""(?ix)                # Case-insensitive and verbose mode
    (?:
        ^hd\s*(?:cannot|can't|can not)\s*(?:help|answer) |              # Exact match for "HD cannot answer"
        hd[0-9]*\s*(?:cannot|can't|can not)\s*(?:help|answer) | # Variations with HD1, HD2, etc.
        not\s+related\s+to\s+op |             # Not related to OP
        more\s+info(?:rmation)?\s+(?:needed|required) |  # More info variations
        not\s+enough\s+info(?:rmation)?\s+for\s+hd[0-9]* | # Not enough info for HD1, etc.
        insufficient\s+info(?:rmation)?\s+(?:provided|given|available) | # Additional variations
        cannot\s+(?:be\s+)?(?:resolved|answered)\s+(?:by|with)\s+(?:the\s+)?hd[0-9]* # Cannot be resolved by HD
    )""",
    re.IGNORECASE | re.VERBOSE
)


def clean_summary_content(text):
    """
    Remove "Subject: " and other prefixes from the text and perform basic normalization.
    """
    if not isinstance(text, str):
        return ""
        
    # Remove common email prefixes
    text = re.sub(r"(?i)(?:subject|fwd|fw|re):\s*", "", text, flags=re.IGNORECASE)
    # Normalize whitespace
    return text.strip().lower()


def filter_summary_content(df):
    """
    Clean the "Summary in English" column and filter out rows with unwanted content.
    Uses regex pattern matching for efficient filtering of various HD-related patterns.
    
    Parameters:
        df (pandas.DataFrame): Input dataframe with "Summary in English" column
        
    Returns:
        pandas.DataFrame: Filtered dataframe with cleaned summaries
    """
    # Create cleaned summary column
    df["cleaned_summary"] = df["Summary in English"].apply(clean_summary_content)
    
    # Filter out rows where cleaned_summary matches any unwanted pattern
    mask = df["cleaned_summary"].str.contains(
        UNWANTED_SUMMARY_PATTERNS, 
        regex=True, 
        na=False
    )
    
    filtered_df = df[~mask].copy()
    return filtered_df
