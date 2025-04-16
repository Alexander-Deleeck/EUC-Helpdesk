from datetime import datetime
import os
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


def setup_chromadb():
    """Initialize ChromaDB client and embedding function."""
    # Get the correct paths based on the project structure
    scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    helpdesk_dir = os.path.dirname(scripts_dir)
    embeddings_dir = os.path.join(helpdesk_dir, "helpdesk-data", "helpdesk-embeddings")
    db_path = os.path.join(embeddings_dir, "chroma_summaries")

    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_type="azure",
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        model_name=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
    )

    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_collection(
        name="helpdesk_summaries_embeddings",
        embedding_function=embedding_function
    )
    print(f"\nNumber of documents in collection: {collection.count()}\n")
    return collection


def query_database(collection, query_text, n_results=5):
    """Query the database and return the most relevant results."""
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    return results


def format_metadata(metadata):
    """Format metadata for display in a readable way."""
    relevant_fields = [
        "Issue key",
        "Issue Type",
        "Status",
        "Priority",
        "Category",
        "Classification",
        "Country",
        "User category",
        "token_count"
    ]
    
    formatted = []
    for field in relevant_fields:
        if field in metadata and metadata[field] is not None:
            formatted.append(f"{field}: {metadata[field]}")
    
    return "\n".join(formatted)


def main():
    collection = setup_chromadb()
    
    print("\nWelcome to the Helpdesk Embeddings Validator!")
    print("You can query the database to find relevant helpdesk entries.")
    print("Type 'q' to quit.\n")
    
    while True:
        query = input("\nEnter your search query: ").strip()
        
        if query.lower() == 'q':
            break
            
        results = query_database(collection, query)
        
        print("\nTop relevant results:")
        print("=" * 80)
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            print(f"\nResult {i+1} (Relevance: {1 - distance:.4f})")
            print("-" * 40)
            print(f"Summary:\n{doc}\n")
            print("Metadata:")
            print(format_metadata(metadata))
            print("=" * 80)


if __name__ == "__main__":
    main() 