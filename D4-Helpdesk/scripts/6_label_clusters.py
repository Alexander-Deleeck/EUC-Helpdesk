"""
Workflow Script 5: Label Clusters using LLM

Loads cluster assignments and original documents, samples documents per cluster,
instructs an LLM (Azure OpenAI) to generate a title and description for each cluster,
and saves the results.

Usage:
  python scripts/5_label_clusters.py --dataset <dataset_name> [--assignments_file <path_to_csv>] [--sample_size <int>]

Example:
  python scripts/5_label_clusters.py --dataset helpdesk --sample_size 15
  python scripts/5_label_clusters.py --dataset helpdesk --assignments_file ./data/cluster_assignments/helpdesk_assignments_20231027_103000.csv
"""

import argparse
import logging
from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np
import random
import time
from datetime import datetime
from typing import Optional, Tuple, List
from dotenv import load_dotenv
from openai import AzureOpenAI, APIError, RateLimitError
from tqdm import tqdm # For progress bar

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
    from src.utils import helpers # Import helpers, including the new ones
except ImportError as e:
    logger.error(f"Error importing src modules: {e}. Make sure the script is run from the project root or src is in PYTHONPATH.")
    sys.exit(1)

# --- Constants ---
# Default prompt template - consider making this configurable if needed
PROMPT_TEMPLATE = """You are an AI assistant specialized in analyzing and summarizing clusters of text documents with similar content, specifically helpdesk tickets or similar support interactions with the European Commission Helpdesk.

Based *only* on the following {num_samples} example documents from a single cluster, please perform the following tasks:
1.  **Generate a concise, descriptive Title** (maximum 10 words) that captures the core topic or issue of this cluster. The title should be specific and informative.
2.  **Generate a brief Description** (1-2 sentences) summarizing the main theme, common problem, or type of request represented by the documents in this cluster.

Cluster Context: The documents represent helpdesk tickets or support requests. Focus on the problem or request being made.

Example Documents:
{document_samples}

---
Output Instructions:
Provide your response *only* in the following format, with no additional explanation or commentary:

Title: <Your generated title here>
Description: <Your generated description here>
"""

# --- Helper Function for LLM Call ---
def get_llm_completion(
    client: AzureOpenAI,
    prompt: str,
    deployment_name: str,
    max_retries: int = 3,
    retry_delay: int = 5,
    temperature: float = 0.2,
    max_tokens: int = 500
) -> Optional[str]:
    """Sends a prompt to Azure OpenAI and handles retries."""
    messages = [{"role": "user", "content": prompt}]
    attempt = 0
    while attempt < max_retries:
        try:
            response = client.chat.completions.create(
                model=deployment_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                # top_p=0.95, # Optional: adjust other parameters if needed
                # frequency_penalty=0,
                # presence_penalty=0,
                # stop=None # Optional: specify stop sequences
            )
            # Check if response is valid and has content
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                 completion = response.choices[0].message.content.strip()
                 logger.debug(f"LLM Response received: {completion[:100]}...") # Log truncated response
                 return completion
            else:
                 logger.warning(f"LLM response structure invalid or empty content. Response: {response}")
                 return None # Treat as failure if structure is wrong

        except RateLimitError as e:
            logger.warning(f"Rate limit exceeded. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(retry_delay)
        except APIError as e:
            logger.error(f"Azure OpenAI API error: {e}. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})", exc_info=True)
            time.sleep(retry_delay)
        except Exception as e:
            logger.error(f"An unexpected error occurred during LLM call: {e}", exc_info=True)
            return None # Don't retry on unexpected errors
        attempt += 1

    logger.error(f"Failed to get LLM completion after {max_retries} retries.")
    return None

# --- Helper Function for Parsing LLM Response ---
def parse_llm_response(response_text: str) -> Tuple[str, str]:
    """Parses the Title and Description from the LLM response string."""
    title = "Error: Could not parse title"
    description = "Error: Could not parse description"
    if not response_text:
        return title, description

    lines = response_text.strip().split('\n')
    for line in lines:
        if line.lower().startswith("title:"):
            title = line[len("title:"):].strip()
        elif line.lower().startswith("description:"):
            description = line[len("description:"):].strip()

    # Basic cleanup if prefixes remain (sometimes happens)
    if title.lower().startswith("title:"): title = title[len("title:"):].strip()
    if description.lower().startswith("description:"): description = description[len("description:"):].strip()

    return title, description

def main(args):
    """Main function to execute the cluster labeling workflow."""
    load_dotenv() # Load environment variables from .env file

    dataset_name = args.dataset
    assignments_file_override = args.assignments_file
    sample_size_override = args.sample_size

    logger.info(f"--- Starting Cluster Labeling for dataset: '{dataset_name}' ---")

    # 1. Load Active Configuration & LLM Settings
    try:
        active_config = config.get_active_config(dataset_name)
        logger.info(f"Loaded configuration: {active_config['name']}")
        llm_defaults = active_config.get('llm_labeling_defaults', {})
        sample_size = sample_size_override if sample_size_override is not None else llm_defaults.get('sample_size', 5)
        max_chars_per_doc = llm_defaults.get('max_chars_per_doc', 2000)
        # max_prompt_tokens = llm_defaults.get('max_prompt_tokens', 3000) # Optional: for token-based truncation
        llm_temperature = llm_defaults.get('llm_temperature', 0.2)
        llm_max_tokens = llm_defaults.get('llm_max_tokens', 500) # For the response
        logger.info(f"Using sample size: {sample_size}")
        logger.info(f"Max chars per doc in prompt: {max_chars_per_doc}")

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # 2. Setup Azure OpenAI Client
    try:
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-07-01-preview") # Default if not set

        if not all([azure_api_key, azure_endpoint, azure_deployment]):
            raise ValueError("Missing Azure OpenAI environment variables (API_KEY, ENDPOINT, DEPLOYMENT_NAME)")

        llm_client = AzureOpenAI(
            api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        logger.info(f"Azure OpenAI client initialized for endpoint: {azure_endpoint} and deployment: {azure_deployment}")
    except Exception as e:
        logger.error(f"Failed to initialize Azure OpenAI client: {e}", exc_info=True)
        sys.exit(1)

    # 3. Load Cluster Assignments
    assignments_dir = active_config.get("cluster_results_path")
    if not assignments_dir:
        logger.error("Config key 'cluster_results_path' is missing.")
        sys.exit(1)

    if assignments_file_override:
        assignments_file_path = Path(assignments_file_override)
        print(f"\n\n\nOverridden Assignments file path:\n\n{assignments_file_path}\n\n\n\n")
        if not assignments_file_path.is_file():
            logger.error(f"Specified assignments file not found: {assignments_file_path}")
            sys.exit(1)
    else:
        # Find the latest assignments file
        assignments_pattern = f"{active_config['name']}_assignments_*.csv"
        assignments_file_path_str = helpers.find_latest_file(assignments_dir, assignments_pattern)
        print(f"\n\n\nlatest Assignments file path string:\n\n{assignments_file_path_str}\n\n\n\n")
        if not assignments_file_path_str:
            logger.error(f"Could not find any assignment files in '{assignments_dir}' matching '{assignments_pattern}'.")
            sys.exit(1)
        assignments_file_path = Path(assignments_file_path_str)

    logger.info(f"Loading cluster assignments from: {assignments_file_path}")
    try:
        assignments_df = pd.read_csv(assignments_file_path)
        print(f"\n\n\nAssignments DataFrame:\n\n{assignments_df.head(5)}\n\n\n\n")
        #time.sleep(60)
        if 'data_id' not in assignments_df.columns or 'cluster_label' not in assignments_df.columns:
            logger.error("Assignments file missing required columns: 'data_id', 'cluster_label'.")
            sys.exit(1)
        logger.info(f"Loaded {len(assignments_df)} assignments.")
    except Exception as e:
        logger.error(f"Error loading assignments file: {e}", exc_info=True)
        sys.exit(1)

    # 4. Load Original Documents from Vector Store
    vs_path = active_config.get("vector_store_path")
    collection_name = active_config.get("vector_store_collection_name")
    if not vs_path or not collection_name:
        logger.error("Config missing 'vector_store_path' or 'vector_store_collection_name'.")
        sys.exit(1)

    logger.info("Connecting to vector store to retrieve documents...")
    try:
        chroma_client = vs_client.get_chroma_client(vs_path)
        collection = vs_client.get_or_create_collection(chroma_client, collection_name)

        all_data_ids = [str(id) for id in assignments_df['data_id'].unique().tolist()] #### I think this is the issue, causes incompatible dtypes
        logger.info(f"Fetching documents for {len(all_data_ids)} unique IDs...")

        # Fetch documents and potentially relevant metadata in batches if needed
        # For simplicity, fetching all at once here. Handle potential memory issues for huge datasets.
        retrieved_data = collection.get(ids=all_data_ids, include=['documents', 'metadatas']) # Adjust include as needed

        # Store documents efficiently for lookup
        document_map = {}
        retrieved_ids = retrieved_data.get('ids', [])
        logger.info(f"Data type of ID from retrieved_ids:\t{retrieved_ids[0]} = {type(retrieved_ids[0])}\n\n")
        retrieved_docs = retrieved_data.get('documents', [])
        # retrieved_metadatas = retrieved_data.get('metadatas', [{} for _ in retrieved_ids]) # Handle missing metadata

        if len(retrieved_ids) != len(all_data_ids):
             logger.warning(f"Requested {len(all_data_ids)} docs, but retrieved {len(retrieved_ids)}. Some IDs might be missing in the vector store.")

        for idx, doc_id in enumerate(retrieved_ids):
            if idx < len(retrieved_docs):
                 document_map[doc_id] = retrieved_docs[idx]
            # Add metadata if needed: document_map[doc_id] = {'document': retrieved_docs[idx], 'metadata': retrieved_metadatas[idx]}

        logger.info(f"Successfully loaded {len(document_map)} documents from vector store.")

    except Exception as e:
        logger.error(f"Failed to retrieve documents from vector store: {e}", exc_info=True)
        sys.exit(1)

    # 5. Iterate, Sample, Prompt LLM, and Parse for each Cluster
    cluster_labels = sorted([l for l in assignments_df['cluster_label'].unique() if l != -1]) # Exclude noise (-1)
    logger.info(f"Found {len(cluster_labels)} non-noise clusters to label.")

    results_list = []
    for idx, cluster_label in enumerate(tqdm(cluster_labels, desc="Labeling Clusters")):
        if idx > 8:
            break
        logger.info(f"\n\n{'-'*100}\n\nProcessing Cluster {cluster_label}\n{'-'*100}\n\n")

        cluster_ids = assignments_df[assignments_df['cluster_label'] == int(cluster_label)]['data_id'].tolist()
        logger.info(f"Cluster {cluster_label} | Cluster IDs: {cluster_ids}")
        cluster_size = len(cluster_ids)
        logger.info(f"Cluster {cluster_label} | Cluster size: {cluster_size}")

        if cluster_size == 0:
            logger.warning(f"Cluster {cluster_label} has no documents. Skipping.")
            continue

        # Determine sample size for this cluster
        actual_sample_size = min(sample_size, cluster_size)
        logger.info(f"Sampling {actual_sample_size} documents...")

        # Sample IDs and retrieve documents
        sampled_ids = random.sample(cluster_ids, actual_sample_size)
        logger.info(f"Sampled IDs: {sampled_ids}")
        
        #logger.info(f"\n{'|'*100}\n\nDocument map key dtype:\n\n{type(list(document_map.keys())[0])}\n\n")
        #logger.info(f"\n\nSampled IDs dtype:\n\n{type(sampled_ids[0])}\n\n{'|'*100}\n\n\n")
        
        #logger.info(f"\n\n\nDocument map:\n\n{document_map}\n\n\n\n")
        
        sampled_docs_raw = [document_map.get(str(doc_id), "") for doc_id in sampled_ids if str(doc_id) in document_map]
        logger.info(f"\n\nCluster {cluster_label} | Sampled docs first entry raw:\n\n{sampled_docs_raw[0]}\n\n")
        #time.sleep(60)
        # Truncate documents and format for prompt
        # Optional: Implement token-based truncation here if needed
        # current_token_count = 0
        # tokenizer = helpers.get_tokenizer() # Ensure tokenizer is available
        sampled_docs_formatted = []
        for i, doc in enumerate(sampled_docs_raw):
            truncated_doc = helpers.truncate_text(doc, max_chars_per_doc)
            # Optional Token Check:
            # doc_tokens = helpers.count_tokens(truncated_doc)
            # if tokenizer and (current_token_count + doc_tokens) > max_prompt_tokens:
            #    logger.warning(f"Token limit ({max_prompt_tokens}) reached for cluster {cluster_label}. Stopping sample inclusion early.")
            #    break
            # current_token_count += doc_tokens
            sampled_docs_formatted.append(f"--- Document {i+1} ---\n{truncated_doc}\n-----------\n")

        if not sampled_docs_formatted:
            logger.warning(f"No valid documents could be sampled or formatted for cluster {cluster_label}. Skipping LLM call.")
            results_list.append({
                'cluster_label': cluster_label,
                'cluster_size': cluster_size,
                'sample_size_used': 0,
                'sampled_doc_ids': sampled_ids,
                'generated_title': "Error: No documents sampled",
                'generated_description': "Error: No documents sampled",
                'llm_response_raw': ""
            })
            continue

        documents_string = "\n".join(sampled_docs_formatted)

        # Construct the prompt
        prompt = PROMPT_TEMPLATE.format(
            num_samples=len(sampled_docs_formatted), # Use actual number formatted
            document_samples=documents_string
        )
        logger.debug(f"Generated Prompt (first 300 chars): {prompt[:300]}...")

        
        # Call LLM
        logger.info(f"Sending request to LLM for cluster {cluster_label}...")
        llm_response = get_llm_completion(
            client=llm_client,
            prompt=prompt,
            deployment_name=azure_deployment,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens
        )

        # Parse response
        if llm_response:
            title, description = parse_llm_response(llm_response)
            logger.info(f"Cluster {cluster_label} | Title: {title} | Description: {description}")
        else:
            logger.error(f"LLM call failed for cluster {cluster_label}. Using error placeholders.")
            title = "Error: LLM call failed"
            description = "Error: LLM call failed"

        # Store results
        results_list.append({
            'cluster_label': cluster_label,
            'cluster_size': cluster_size,
            'sample_size_used': len(sampled_docs_formatted),
            'sampled_doc_ids': sampled_ids,
            'generated_title': title,
            'generated_description': description,
            'llm_response_raw': llm_response if llm_response else "" # Store raw response for debugging
        })

        # Optional: Add a small delay to avoid hitting rate limits aggressively
        time.sleep(3) # Adjust as needed

    # 6. Save Results
    if not results_list:
        logger.warning("No cluster labels were generated.")
    else:
        results_df = pd.DataFrame(results_list)
        output_dir = active_config.get("cluster_labels_path")
        if not output_dir:
            logger.error("Config key 'cluster_labels_path' is missing. Cannot save results.")
        else:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{active_config['name']}_cluster_labels_{timestamp}.csv"
            full_output_path = output_path / output_filename

            try:
                results_df.to_csv(full_output_path, index=False, encoding='utf-8')
                logger.info(f"Cluster labels saved successfully to: '{full_output_path}'\nwith filename: '{output_filename}'")
            except Exception as e:
                logger.error(f"Error saving cluster labels file: {e}", exc_info=True)

    logger.info(f"--- Cluster Labeling for '{dataset_name}' Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate labels and descriptions for clusters using an LLM.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset configuration to use."
    )
    parser.add_argument(
        "--assignments_file",
        type=str,
        default=None,
        help="Path to the specific cluster assignments CSV file (optional, finds latest otherwise)."
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of documents to sample per cluster (overrides config default)."
    )
    # Add other arguments if needed (e.g., --output_dir override)

    args = parser.parse_args()

    # Basic validation
    if args.sample_size is not None and args.sample_size <= 0:
        print("Error: --sample_size must be a positive integer.")
        sys.exit(1)

    main(args)