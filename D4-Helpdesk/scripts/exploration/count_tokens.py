import tiktoken
import json
import os
from pathlib import Path
from typing import Dict, Tuple, Union

class TokenCountError(Exception):
    """Custom exception for token counting errors"""
    pass

def count_tokens_in_json(json_data: dict) -> int:
    """
    Count tokens in a single JSON file's content.
    
    Args:
        json_data (dict): Loaded JSON data
        
    Returns:
        int: Total token count for the file
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    total_tokens = 0
    
    for key, value in json_data.items():
        if not key.isnumeric():
            continue
            
        try:
            text = value['text']
            tokens = encoding.encode(text)
            total_tokens += len(tokens)
        except KeyError:
            raise TokenCountError(f"Missing 'text' field in segment {key}")
            
    return total_tokens

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

def process_files(path: Union[str, Path]) -> Tuple[int, float, int, Dict[str, int]]:
    """
    Process JSON files and calculate token counts and costs.
    
    Args:
        path (Union[str, Path]): Path to either a single JSON file or a directory containing JSON files
        
    Returns:
        Tuple containing:
            - total_tokens (int): Sum of all tokens across all processed files
            - total_cost (float): Total estimated cost in euros
            - num_files (int): Number of files processed
            - file_tokens (Dict[str, int]): Dictionary mapping filenames to their token counts
            
    Raises:
        FileNotFoundError: If the path doesn't exist
        TokenCountError: If there's an error processing the JSON content
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    
    files_to_process = []
    if path.is_file():
        if not path.suffix.lower() == '.json':
            raise ValueError(f"File must be a JSON file: {path}")
        files_to_process = [path]
    else:  # is directory
        files_to_process = list(path.glob('*.json'))
        if not files_to_process:
            raise FileNotFoundError(f"No JSON files found in directory: {path}")
    
    total_tokens = 0
    file_tokens = {}
    
    for file_path in files_to_process:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            file_token_count = count_tokens_in_json(json_data)
            total_tokens += file_token_count
            file_tokens[file_path.name] = file_token_count
            
        except json.JSONDecodeError:
            raise TokenCountError(f"Invalid JSON file: {file_path}")
        except Exception as e:
            raise TokenCountError(f"Error processing file {file_path}: {str(e)}")
    
    total_cost = calculate_embedding_cost(total_tokens)
    num_files = len(files_to_process)
    
    return total_tokens, total_cost, num_files, file_tokens

# Example usage:
if __name__ == "__main__":
    # Example path - replace with your actual path
    test_path = "path/to/your/json/files"
    
    try:
        tokens, cost, num_files, file_counts = process_files(test_path)
        print(f"\nProcessed {num_files} files:")
        for filename, count in file_counts.items():
            print(f"- {filename}: {count} tokens")
        print(f"\nTotal tokens: {tokens}")
        print(f"Estimated cost: €{cost:.4f}")
        
    except (FileNotFoundError, TokenCountError, ValueError) as e:
        print(f"Error: {str(e)}")