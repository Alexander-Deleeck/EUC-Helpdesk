from datetime import datetime
from typing import Optional, Tuple, List
from dotenv import load_dotenv
from openai import AzureOpenAI, APIError, RateLimitError
import time
import logging
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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