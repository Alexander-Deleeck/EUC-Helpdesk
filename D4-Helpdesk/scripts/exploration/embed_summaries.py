# process_embeddings_chroma.py

from typing import List
import os
import re
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import tiktoken
# from utils_cleaning import clean_custom_fields, clean_email_content
# Load environment variables from .env file
load_dotenv()


def count_tokens(text: str) -> int:
    """
    Count tokens in a text string using the cl100k_base tokenizer (used by text-embedding-ada-002).
    
    Args:
        text (str): Text to tokenize
        
    Returns:
        int: Number of tokens in the text
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def calculate_embedding_cost(total_tokens: int, price_per_1k_tokens: float = 0.000121) -> float:
    """
    Calculate the estimated cost for embedding tokens.
    
    Args:
        total_tokens (int): Total number of tokens to embed
        price_per_1k_tokens (float): Price in euros per 1000 tokens
        
    Returns:
        float: Estimated cost in euros
    """
    return (total_tokens / 1000) * price_per_1k_tokens


# --- Cleaning and Filtering Functions ---
# def clean_summary_content(text):
#     """
#     Remove "Subject: " from the text and perform basic normalization.
#     """
#     text = re.sub("Subject: ", "", text, flags=re.IGNORECASE)
#     text = re.sub("FWD: ", "", text, flags=re.IGNORECASE)
#     return text.strip().lower()


# def filter_summary_content(df):
#     """
#     Clean the "Summary in English" column and filter out rows where the cleaned content equals
#     'hd cannot answer' or contains undesired phrases.
#     """
#     # Create a cleaned summary column
#     df["cleaned_summary"] = df["Summary in English"].apply(clean_summary_content)
#     # Build filter mask: remove rows with unwanted content
#     mask = (
#         (df["cleaned_summary"] == "hd cannot answer")
#         | (
#             df["cleaned_summary"].str.contains(
#                 "not related to op", case=False, na=False
#             )
#         )
#         | (
#             df["cleaned_summary"].str.contains(
#                 "more information needed", case=False, na=False
#             )
#         )
#         | (
#             df["cleaned_summary"].str.contains(
#                 "more information required", case=False, na=False
#             )
#         )
#     )
#     filtered_df = df[~mask].copy()
#     return filtered_df


# --- Batch Storage Function for Chroma ---
def add_documents_in_batches(
    collection, documents, docs_metadata, docs_ids: List[str], batch_size: int=500, sleep_time: float=1.0
):
    """
    Add documents to the Chroma collection in batches with optional rate limiting.
    """
    for i in range(0, len(documents), batch_size):
        batch_end = min(i + batch_size, len(documents))
        print(
            f"Processing batch {i//batch_size + 1} ({i} to {batch_end} of {len(documents)})"
        )

        collection.add(
            documents=documents[i:batch_end],
            metadatas=docs_metadata[i:batch_end],
            ids=docs_ids[i:batch_end],
        )

        if batch_end < len(documents):
            time.sleep(sleep_time)
    return


# --- Main Processing Function ---
def main():
    # Load your dataframe from a CSV file (adjust the path as needed)

    scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    helpdesk_dir = os.path.dirname(scripts_dir)
    helpdesk_data_dir = os.path.join(helpdesk_dir, "helpdesk-data")

    csv_dir = os.path.join(helpdesk_data_dir, "JIRA FULL EXTRACT", "CSV")
    embeddings_dir = os.path.join(helpdesk_data_dir, "helpdesk-embeddings")

    # Initialize the Chroma embedding function with Azure OpenAI settings from your environment
    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_type="azure",
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        model_name=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
    )

    # Set up the Chroma persistent client and collection
    db_path = os.path.join(embeddings_dir, "chroma_summaries")
    Path(db_path).mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=db_path)

    collection = chroma_client.get_or_create_collection(
        # name=f"helpdesk_summaries_embeddings",
        name=f"helpdesk_complete_summaries_embeddings",
        embedding_function=embedding_function,
        metadata={
            "description": f"Embeddings for email summaries from all the EC helpdesk datasets",
            "created": str(datetime.now()),
            "author": "Alexander Deleeck",
        },
    )
    print(f"Collection '{collection.name}' successfully created")

    # Define the path to the file
    for filename in [file for file in os.listdir(csv_dir) if file.endswith("_cleaned.csv")]:
        print(f"\n\n\t\tProcessing file: {filename}\n{'-'*50}\n")
        filepath = Path(f"{csv_dir}/{filename}")

        df = pd.read_csv(filepath, header=0, encoding='utf-8-sig')

        # --- Prepare Documents, Metadata, and IDs ---
        # Define the metadata fields from your dataframe
        metadata_fields = [
            "Summary",
            "Issue key",
            "Issue id",
            "Parent id",
            "Issue Type",
            "Status",
            "Priority",
            "Resolution",
            "Creator",
            "Components",
            "Description",
            "Category",
            "Classification",
            "Classification (Calculated)",
            "Country",
            "Solution",
            "Summary Solution",
            "Summary in English",
            "User category",
        ]

        documents = []  # List of document texts (cleaned summary)
        docs_metadata = []  # List of metadata dictionaries (each linked to a document)
        docs_ids = []  # List of document IDs (using "Issue id")
        total_tokens = 0  # Track total tokens

        # Iterate over each row in the filtered dataframe
        for index, row in df.iterrows():
            # Use the cleaned summary as the document text. Fallback to original if needed.
            doc_text = row.get("cleaned_summary", row["Summary in English"])
            documents.append(doc_text)

            # Count tokens for this document
            doc_tokens = count_tokens(doc_text)
            total_tokens += doc_tokens

            # Build metadata dictionary using the specified metadata fields
            meta = {field: row[field] for field in metadata_fields if field in row}
            meta['token_count'] = doc_tokens  # Add token count to metadata
            docs_metadata.append(meta)

            # Use 'Issue id' as the document's unique identifier (converted to string)
            doc_id = str(row["Issue id"])
            docs_ids.append(doc_id)

        # Calculate estimated cost
        estimated_cost = calculate_embedding_cost(total_tokens)

        print(f"\n{'='*50}\nToken Statistics:")
        print(f"Total documents: {len(documents)}")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Average tokens per document: {total_tokens/len(documents):.1f}")
        print(f"Estimated embedding cost: €{estimated_cost:.4f}")
        print(f"\n{'='*50}\n")

        print(f"\nAdding {len(documents)} documents to the Chroma collection...")
        # Process and add the documents into the Chroma collection by batches
        add_documents_in_batches(
            collection, documents, docs_metadata, docs_ids, batch_size=500, sleep_time=18.0
        )
        
    print(f"\n\nALL Documents added successfully to the Chroma database.")

    print(f"\n{'#'*50}\nCompleted. Total documents in chromadb: {collection.count()}\n")

if __name__ == "__main__":
    main()
