"""
Defines common operations for interacting with a ChromaDB collection,
including adding data, retrieving data, and querying.
"""

import time
import logging
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from tqdm import tqdm # Optional progress bar

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_chroma_data(
    df: pd.DataFrame,
    id_column: str,
    text_column: str,
    metadata_columns: List[str]
) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """
    Prepares data from a DataFrame for insertion into ChromaDB.

    Extracts documents (text), metadata, and IDs based on specified column names.
    Handles potential missing columns gracefully.

    Args:
        df: The pandas DataFrame containing the data.
        id_column: The name of the column to use for unique document IDs.
        text_column: The name of the column containing the text to be embedded.
        metadata_columns: A list of column names to include in the metadata.

    Returns:
        A tuple containing three lists:
        - documents (List[str]): The text content for each document.
        - metadatas (List[Dict[str, Any]]): Metadata dictionaries for each document.
        - ids (List[str]): Unique string IDs for each document.

    Raises:
        ValueError: If essential columns (id_column, text_column) are missing.
    """
    logger.info(f"Preparing data for ChromaDB from DataFrame (Shape: {df.shape})...")
    logger.info(f"ID Column: '{id_column}', Text Column: '{text_column}'")
    logger.info(f"Metadata Columns: {metadata_columns}")

    # Validate essential columns exist
    if id_column not in df.columns:
        raise ValueError(f"ID column '{id_column}' not found in DataFrame.")
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in DataFrame.")

    documents = []
    metadatas = []
    ids = []

    # Check which requested metadata columns actually exist in the DataFrame
    available_metadata_cols = [col for col in metadata_columns if col in df.columns]
    missing_metadata_cols = set(metadata_columns) - set(available_metadata_cols)
    if missing_metadata_cols:
        logger.warning(f"Requested metadata columns not found in DataFrame: {missing_metadata_cols}")

    # Iterate and prepare data
    for index, row in df.iterrows():
        # Get ID, ensure it's a string
        doc_id = str(row[id_column])

        # Get text content, handle potential NaN/None
        doc_text = row[text_column]
        if pd.isna(doc_text):
            logger.warning(f"Row {index} (ID: {doc_id}) has null text in column '{text_column}'. Skipping row.")
            continue # Skip rows with no text
        doc_text = str(doc_text)

        # Build metadata dictionary only from available columns, handle NaN
        meta = {}
        for col in available_metadata_cols:
            value = row[col]
            if pd.isna(value):
                meta[col] = None # Or use an empty string: ""
            else:
                # Attempt to convert to standard Python types if needed
                # ChromaDB might handle numpy types, but explicit conversion is safer
                if isinstance(value, (pd.Timestamp, pd.Period)):
                    meta[col] = str(value) # Convert timestamps/periods to string
                elif hasattr(value, 'item'): # Handle numpy types like int64, float64
                     meta[col] = value.item()
                else:
                     meta[col] = value

        documents.append(doc_text)
        metadatas.append(meta)
        ids.append(doc_id)

    logger.info(f"Prepared {len(ids)} documents for ChromaDB.")
    if len(ids) != len(df):
         logger.warning(f"Number of prepared documents ({len(ids)}) differs from input DataFrame rows ({len(df)}) due to filtering (e.g., null text).")

    return documents, metadatas, ids


def add_data_to_collection(
    collection: chromadb.Collection,
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    ids: List[str],
    batch_size: int = 100, # Chroma default is high, smaller might be safer for some APIs
    sleep_time: float = 0.1 # Small delay between batches can help avoid rate limits
):
    """
    Adds documents, metadata, and IDs to the specified Chroma collection in batches.

    Uses `collection.add` which handles embedding generation internally if embeddings
    are not provided explicitly.

    Args:
        collection: The initialized ChromaDB collection object.
        documents: List of document texts.
        metadatas: List of metadata dictionaries.
        ids: List of unique document IDs.
        batch_size: Number of documents to add in each batch.
        sleep_time: Time in seconds to sleep between batches (0 to disable).
    """
    if not (len(documents) == len(metadatas) == len(ids)):
        raise ValueError("Lengths of documents, metadatas, and ids must match.")

    total_docs = len(documents)
    logger.info(f"Adding {total_docs} documents to collection '{collection.name}'...")

    # Use tqdm for progress bar if available
    for i in tqdm(range(0, total_docs, batch_size), desc="Adding batches to ChromaDB"):
        batch_end = min(i + batch_size, total_docs)
        batch_docs = documents[i:batch_end]
        batch_meta = metadatas[i:batch_end]
        batch_ids = ids[i:batch_end]

        try:
            collection.add(
                documents=batch_docs,
                metadatas=batch_meta,
                ids=batch_ids
            )
            # logger.debug(f"Added batch {i//batch_size + 1} ({i}-{batch_end}) successfully.")
        except Exception as e:
            logger.error(f"Error adding batch {i//batch_size + 1} ({i}-{batch_end}) to ChromaDB: {e}", exc_info=True)
            # Decide whether to raise, continue, or implement retries here
            raise # Re-raise by default to indicate failure

        if sleep_time > 0 and batch_end < total_docs:
            time.sleep(sleep_time)

    logger.info(f"Finished adding {total_docs} documents to collection '{collection.name}'. New count: {collection.count()}")


def get_all_data(collection: chromadb.Collection, include: List[str] = ["metadatas", "documents", "embeddings"]) -> Optional[pd.DataFrame]:
    """
    Retrieves all data (IDs, embeddings, metadata, documents) from a ChromaDB collection.

    Args:
        collection: The initialized ChromaDB collection object.
        include: List of fields to include in the results (default: all).

    Returns:
        A pandas DataFrame containing the retrieved data, with IDs as the index.
        Returns None if the collection is empty or an error occurs.
    """
    collection_count = collection.count()
    logger.info(f"Retrieving all {collection_count} items from collection '{collection.name}'...")
    if collection_count == 0:
        logger.warning(f"Collection '{collection.name}' is empty. Returning None.")
        return None

    try:
        # ChromaDB's get() retrieves all items if no IDs/where clause is specified
        # Using a large limit ensures all items are fetched (adjust if needed for huge collections)
        # Note: Large fetches can consume significant memory.
        results = collection.get(include=include, limit=collection_count) # Fetch all

        if not results or not results.get('ids'):
             logger.warning(f"Received empty or invalid results from collection.get() for '{collection.name}'")
             return None

        # Construct DataFrame
        data_dict = {'id': results['ids']}
        logger.debug(f"Results: {results['ids'][:5]}")
        # Handle embeddings separately due to their 2D nature
        embeddings = results.get('embeddings')
        if 'embeddings' in include and embeddings is not None and len(embeddings) > 0:
            # Convert each embedding vector to a list to ensure proper DataFrame handling
            data_dict['embedding'] = [list(emb) for emb in embeddings]
            logger.info(f"Embeddings shape: {len(embeddings)}x{len(embeddings[0])}")
        
        if 'documents' in include and results.get('documents'):
            data_dict['document'] = results['documents']
            
        # Create initial DataFrame
        temp_df = pd.DataFrame(data_dict)
        
        # Handle metadata separately
        if 'metadatas' in include and results.get('metadatas'):
            # Convert list of metadata dicts into DataFrame columns
            metadata_df = pd.DataFrame(results['metadatas'])
            # Reset index if metadata_df aligns, otherwise need careful merge
            if len(metadata_df) == len(temp_df):
                df_all = pd.concat([temp_df.reset_index(drop=True), metadata_df.reset_index(drop=True)], axis=1)
            else:
                logger.warning("Metadata length mismatch, returning only IDs, embeddings, documents.")
                df_all = temp_df
        else:
            df_all = temp_df

        # Set ID as index
        # if 'id' in df_all.columns:
        #     df_all = df_all.set_index('id')

        logger.info(f"Successfully retrieved {len(df_all)} items into DataFrame (Shape: {df_all.shape}).")
        return df_all

    except Exception as e:
        logger.error(f"Error retrieving data from collection '{collection.name}': {e}", exc_info=True)
        return None


def query_collection(
    collection: chromadb.Collection,
    query_texts: List[str],
    n_results: int = 5,
    include: List[str] = ["documents", "metadatas", "distances"]
) -> Optional[Dict[str, Any]]:
    """
    Performs a query against the ChromaDB collection.

    Args:
        collection: The initialized ChromaDB collection object.
        query_texts: A list of query texts.
        n_results: The number of results to return for each query.
        include: List of fields to include in the results.

    Returns:
        The query results dictionary from ChromaDB, or None if an error occurs.
    """
    if not query_texts:
        logger.warning("No query texts provided.")
        return None
    logger.info(f"Querying collection '{collection.name}' with {len(query_texts)} query text(s) for {n_results} results each.")

    try:
        results = collection.query(
            query_texts=query_texts,
            n_results=min(n_results, collection.count()), # Don't ask for more results than exist
            include=include
        )
        logger.info(f"Query successful. Returning results.")
        return results
    except Exception as e:
        logger.error(f"Error querying collection '{collection.name}': {e}", exc_info=True)
        return None


# Example Usage (for testing)
# if __name__ == "__main__":
#     # Assume client.py functions work and config is set up
#     from .client import get_chroma_client, get_openai_embedding_function, get_or_create_collection
#     from .. import config # Go up two levels to reach src, then config

#     try:
#         print("--- Testing Vector Store Ops --- ")
#         # 1. Setup
#         active_config = config.get_active_config("helpdesk") # Or another dataset
#         chroma_client = get_chroma_client(active_config["vector_store_path"])
#         emb_func = get_openai_embedding_function()
#         collection = get_or_create_collection(
#             client=chroma_client,
#             collection_name=active_config["vector_store_collection_name"],
#             embedding_function=emb_func
#         )
#         print(f"Using collection '{collection.name}' with {collection.count()} items.")

#         # 2. Prepare Dummy Data
#         dummy_data = {
#             'ticket_id': ['TKT-001', 'TKT-002', 'TKT-003'],
#             'summary_clean': ['Cannot login to system', 'Printer not working', 'Need software update'],
#             'category': ['Login Issue', 'Hardware', 'Software Request'],
#             'status': ['Open', 'Open', 'Closed']
#         }
#         dummy_df = pd.DataFrame(dummy_data)

#         # Assume config for this dummy scenario
#         dummy_id_col = 'ticket_id'
#         dummy_text_col = 'summary_clean'
#         dummy_meta_cols = ['category', 'status']

#         docs, metas, ids = prepare_chroma_data(dummy_df, dummy_id_col, dummy_text_col, dummy_meta_cols)
#         print(f"\nPrepared data: {len(ids)} items")
#         print(f"Sample Doc: {docs[0]}")
#         print(f"Sample Meta: {metas[0]}")
#         print(f"Sample ID: {ids[0]}")

#         # 3. Add Data (use upsert=True if running multiple times)
#         print("\nAdding/Upserting data...")
#         # For testing, maybe upsert is safer if running repeatedly
#         # add_data_to_collection(collection, docs, metas, ids, batch_size=2)
#         collection.upsert(ids=ids, documents=docs, metadatas=metas)
#         print(f"Collection count after add/upsert: {collection.count()}")

#         # 4. Get All Data
#         print("\nGetting all data...")
#         all_data_df = get_all_data(collection)
#         if all_data_df is not None:
#             print("Retrieved DataFrame sample:")
#             print(all_data_df.head())
#             # Check if dummy metadata columns are present
#             print(f"Columns: {all_data_df.columns.tolist()}")
#         else:
#             print("Failed to retrieve data.")

#         # 5. Query Collection
#         print("\nQuerying collection...")
#         query_results = query_collection(collection, query_texts=["login problem", "update software"], n_results=2)
#         if query_results:
#             print("Query results obtained:")
#             # print(query_results)
#             # Basic print of first result's documents
#             if query_results.get('documents') and query_results['documents'][0]:
#                  print(f"  Docs for 'login problem': {query_results['documents'][0]}")
#             if query_results.get('documents') and len(query_results['documents']) > 1 and query_results['documents'][1]:
#                  print(f"  Docs for 'update software': {query_results['documents'][1]}")
#         else:
#             print("Query failed.")

#         print("\n--- Vector Store Ops tests passed --- ")

#     except Exception as e:
#          print(f"\nAn unexpected error occurred during testing: {e}", exc_info=True) 