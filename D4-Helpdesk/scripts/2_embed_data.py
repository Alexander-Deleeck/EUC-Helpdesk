"""
Workflow Script 2: Data Embedding and Vector Store Population

Loads cleaned data for a specified dataset, generates embeddings for the designated
text column, calculates token counts and costs, and populates a ChromaDB vector store.

Usage:
  python scripts/2_embed_data.py --dataset <dataset_name>

Example:
  python scripts/2_embed_data.py --dataset helpdesk
"""

import argparse
import logging
from pathlib import Path
import sys
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
    from src.embeddings import client as embed_client, core as embed_core, tokens as embed_tokens
    from src.vector_store import client as vs_client, ops as vs_ops
    from src.utils import helpers
except ImportError as e:
    logger.error(f"Error importing src modules: {e}. Make sure the script is run from the project root or src is in PYTHONPATH.")
    sys.exit(1)

def main(dataset_name: str):
    """Main function to execute the data embedding and vector store population workflow."""
    logger.info(f"--- Starting Embedding Workflow for dataset: '{dataset_name}' ---")

    # 1. Load Active Configuration
    try:
        active_config = config.get_active_config(dataset_name)
        logger.info(f"Loaded configuration: {active_config['name']}")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # 2. Load Cleaned Data
    processed_template = active_config.get("processed_path_template")
    if not processed_template:
        logger.error("Config key 'processed_path_template' is missing. Cannot load cleaned data.")
        sys.exit(1)

    try:
        cleaned_data_path = Path(str(processed_template).format(active_config['name']))
        if not cleaned_data_path.is_file():
            logger.error(f"Cleaned data file not found at expected path: {cleaned_data_path}")
            logger.error("Please run the cleaning script (1_clean_data.py) first.")
            sys.exit(1)

        logger.info(f"Loading cleaned data from: {cleaned_data_path}")
        df_cleaned = pd.read_csv(cleaned_data_path, encoding='utf-8-sig')
        if df_cleaned.empty:
             logger.warning("Cleaned DataFrame is empty. Nothing to embed. Exiting.")
             sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to load cleaned data from {cleaned_data_path}: {e}", exc_info=True)
        sys.exit(1)

    # 3. Prepare Data for ChromaDB
    id_col = active_config.get('id_column')
    text_col = active_config.get('text_column_for_embedding')
    meta_cols = active_config.get('metadata_columns_to_store')

    if not id_col or not text_col or not meta_cols:
        logger.error("Configuration missing 'id_column', 'text_column_for_embedding', or 'metadata_columns_to_store'.")
        sys.exit(1)

    try:
        documents, metadatas, ids = vs_ops.prepare_chroma_data(
            df=df_cleaned,
            id_column=id_col,
            text_column=text_col,
            metadata_columns=meta_cols
        )
        if not documents:
             logger.warning("No documents were prepared for embedding (possibly due to empty text column). Exiting.")
             sys.exit(0)
    except ValueError as e:
        logger.error(f"Error preparing data for ChromaDB: {e}")
        sys.exit(1)

    # 4. Calculate Token Counts and Estimated Cost
    try:
        token_stats = embed_tokens.get_token_stats(documents)
        logger.info("--- Token Statistics ---")
        for key, value in token_stats.items():
            if 'cost' in key:
                 logger.info(f"  {key}: €{value:.4f}")
            elif 'tokens' in key and isinstance(value, (int, float)):
                 logger.info(f"  {key}: {value:,.0f}")
            else:
                 logger.info(f"  {key}: {value}")
        logger.info("------------------------")
    except Exception as e:
        logger.warning(f"Could not calculate token stats: {e}")

    # 5. Initialize Embedding Client
    try:
        azure_client = embed_client.get_azure_openai_client()
    except ValueError as e:
        logger.error(f"Failed to initialize Azure OpenAI Client: {e}")
        sys.exit(1)
    except Exception as e:
         logger.error(f"Unexpected error initializing Azure OpenAI Client: {e}", exc_info=True)
         sys.exit(1)

    # 6. Generate Embeddings (using core embedding function)
    # Note: generate_embeddings_batch expects only the list of texts
    try:
        logger.info(f"Generating embeddings for {len(documents)} documents...")
        embeddings = embed_core.generate_embeddings_batch(
            texts=documents,
            client=azure_client,
            # Optionally get batch size etc. from config if needed
            # embedding_model_name=active_config.get('embedding_model_override') # Example override
        )
        if embeddings is None:
            logger.error("Embedding generation failed. Check logs for details.")
            sys.exit(1)
        if len(embeddings) != len(documents):
             logger.error(f"Mismatch between number of documents ({len(documents)}) and generated embeddings ({len(embeddings)}). Aborting.")
             sys.exit(1)

    except Exception as e:
        logger.error(f"Error during embedding generation: {e}", exc_info=True)
        sys.exit(1)

    # 7. Setup Vector Store Client and Collection
    vs_path = active_config.get("vector_store_path")
    collection_name = active_config.get("vector_store_collection_name")
    if not vs_path or not collection_name:
        logger.error("Configuration missing 'vector_store_path' or 'vector_store_collection_name'.")
        sys.exit(1)

    try:
        chroma_client = vs_client.get_chroma_client(vs_path)
        # We don't need the embedding function here if using collection.add
        # Embedding function is associated with the collection itself
        collection = vs_client.get_or_create_collection(
            client=chroma_client,
            collection_name=collection_name,
            # Embedding function will be retrieved/set by get_or_create if needed
            collection_metadata={'dataset': active_config['name']} # Example metadata
        )
    except Exception as e:
        logger.error(f"Failed to setup ChromaDB client or collection: {e}", exc_info=True)
        sys.exit(1)

    # 8. Add Data to Collection (with Embeddings)
    # IMPORTANT: ChromaDB's collection.add can take embeddings directly.
    # If we provide embeddings, it skips calling its internal embedding function.
    try:
        logger.info(f"Adding {len(ids)} items with pre-generated embeddings to collection '{collection_name}'...")
        # Modify vs_ops.add_data_to_collection if needed, or use collection.add directly
        # Using collection.add/upsert directly might be simpler here
        collection.upsert(
             ids=ids,
             embeddings=embeddings, # Provide the generated embeddings
             metadatas=metadatas,
             documents=documents # Also store the source document text
         )
        logger.info(f"Upsert operation complete. Final collection count: {collection.count()}")

        # Old batching approach (if needed, adapt vs_ops.add_data_to_collection to accept embeddings)
        # vs_ops.add_data_to_collection(
        #     collection=collection,
        #     documents=documents, # Text is still useful to store
        #     metadatas=metadatas,
        #     ids=ids,
        #     embeddings=embeddings # Requires function modification
        # )

    except Exception as e:
        logger.error(f"Failed to add data to ChromaDB collection: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"--- Embedding Workflow for '{dataset_name}' Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the data embedding and vector store population workflow.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=f"Name of the dataset configuration to use (e.g., {list(config.DATASET_CONFIGS.keys())})."
    )

    args = parser.parse_args()
    main(args.dataset) 