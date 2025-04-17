"""
Workflow Script 5: Interactive Vector Store Querying

Sets up a connection to the vector store for a specified dataset and allows
interactive querying to find similar documents based on text input.

Usage:
  python scripts/5_validate_query.py --dataset <dataset_name> [--results <int>]

Example:
  python scripts/5_validate_query.py --dataset helpdesk --results 3
"""

import argparse
import logging
from pathlib import Path
import sys
import pandas as pd # Although not directly used, good practice

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure the src directory is in the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Import necessary functions from the src library
try:
    from src import config
    from src.vector_store import client as vs_client, ops as vs_ops
    from src.utils import helpers
except ImportError as e:
    logger.error(f"Error importing src modules: {e}. Make sure the script is run from the project root or src is in PYTHONPATH.")
    sys.exit(1)

def main(args):
    """Main function to run the interactive query validation loop."""
    dataset_name = args.dataset
    num_results = args.results
    logger.info(f"--- Starting Interactive Query Validation for dataset: '{dataset_name}' ---")

    # 1. Load Active Configuration
    try:
        active_config = config.get_active_config(dataset_name)
        logger.info(f"Loaded configuration: {active_config['name']}")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # 2. Setup Vector Store Client and Get Collection
    vs_path = active_config.get("vector_store_path")
    collection_name = active_config.get("vector_store_collection_name")
    if not vs_path or not collection_name:
        logger.error("Config missing 'vector_store_path' or 'vector_store_collection_name'.")
        sys.exit(1)

    try:
        chroma_client = vs_client.get_chroma_client(vs_path)
        # Get embedding function details from config for the query
        # Note: The collection itself has an associated embedding function,
        # but ChromaDB query likely uses the one passed during query or client init
        # Let's rely on the collection's internal function if possible,
        # or initialize one if needed for query (ChromaDB handles this mostly).
        collection = vs_client.get_or_create_collection(chroma_client, collection_name)

        if collection.count() == 0:
            logger.error(f"Vector store collection '{collection_name}' is empty. Cannot query.")
            sys.exit(1)
        logger.info(f"Connected to collection '{collection_name}' with {collection.count()} items.")

    except Exception as e:
        logger.error(f"Failed to setup ChromaDB client or collection: {e}", exc_info=True)
        sys.exit(1)

    # 3. Interactive Query Loop
    print("\nWelcome to the Vector Store Query Validator!")
    print("Enter your search query below. Type 'q' or 'quit' to exit.")

    # Determine relevant fields for metadata display from config
    relevant_meta_fields = active_config.get('metadata_columns_to_store', [])

    while True:
        try:
            query_text = input("\nEnter search query: ").strip()
            if query_text.lower() in ['q', 'quit']:
                break
            if not query_text:
                continue

            # Perform the query using the vector store ops function
            results = vs_ops.query_collection(
                collection=collection,
                query_texts=[query_text],
                n_results=num_results,
                include=["documents", "metadatas", "distances"] # Ensure these are included
            )

            if results is None:
                print("Query failed. Check logs for errors.")
                continue

            # Check if results are empty for this specific query
            if not results.get('ids') or not results['ids'][0]:
                 print("No relevant documents found for this query.")
                 continue

            print(f"\n--- Top {len(results['ids'][0])} Results for '{query_text}' ---")
            print("=" * 60)

            # Iterate through the results for the single query
            for i in range(len(results['ids'][0])):
                doc_id = results['ids'][0][i]
                distance = results['distances'][0][i]
                document = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                relevance_score = 1 - distance # Higher is better (assuming cosine distance)

                print(f"\nResult {i+1} (ID: {doc_id}, Score: {relevance_score:.4f})")
                print("-" * 40)
                print("Document Text:")
                print(document)
                print("\nMetadata:")
                print(helpers.format_metadata_for_display(metadata, relevant_meta_fields))
                print("=" * 60)

        except EOFError:
            # Handle Ctrl+D or unexpected end of input
            print("\nExiting due to end of input.")
            break
        except KeyboardInterrupt:
            # Handle Ctrl+C
            print("\nExiting due to user interrupt.")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred during query loop: {e}", exc_info=True)
            print("An error occurred. Please check logs. Exiting query loop.")
            break

    logger.info(f"--- Interactive Query Validation for '{dataset_name}' Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactively query the vector store.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset configuration to use."
    )
    parser.add_argument(
        "--results",
        type=int,
        default=5,
        help="Number of results to return per query. Default: 5"
    )

    args = parser.parse_args()
    main(args) 