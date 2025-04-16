# D4-Helpdesk\scripts\scraping\scrape_ted_europa.py
import requests
from bs4 import BeautifulSoup
import os
import time
from urllib.parse import urljoin  # Keep for consistency, though less critical here

# Import necessary functions and Enum from your utils file
# Ensure utils_scraping.py is in the same directory or accessible via PYTHONPATH
from utils_scraping import (
    extract_page_title,
    save_section,
    convert_to_md,
    enrich_markdown,
    generate_filename,
    FileType,  # Import the Enum
)

# --- Configuration ---
# The specific help page URL to scrape
BASE_URL = "https://ted.europa.eu/en/help/"

# Output directory relative to the script location (Corrected Path)
OUTPUT_DIR = os.path.join("..", "results", "ted_europa", "ted_europa_docs_md")
FILE_TYPE_TO_SAVE = FileType.MD  # Choose MD or TXT

# CSS Selector for the main content container div
CONTENT_CONTAINER_SELECTOR = "div.ted-aside-list__content"
# Tag within the container holding the relevant pieces of content
CONTENT_SECTION_TAG = "section"

# Scraping politeness and robustness
REQUEST_DELAY_SECONDS = 1  # Still good practice even for one page
REQUEST_TIMEOUT_SECONDS = 30  # Timeout for requests
# --------------------

# Ensure the base output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Starting scrape for single page: {BASE_URL}")
print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
print(f"Content container selector: '{CONTENT_CONTAINER_SELECTOR}'")
print(f"Content section tag: '<{CONTENT_SECTION_TAG}>'")
print("-" * 30)
pagenames = [
    "about-ted",
    "ted-account",
    "notice-view",
    "search-results",
    "data-reuse",
    "search-browse",
]

for pagenames in pagenames:
    CURRENT_URL = urljoin(BASE_URL, pagenames)

    processed = False  # Flag to track if processing was successful
    try:
        # Add a delay before making the request
        print(f"Waiting {REQUEST_DELAY_SECONDS} second(s)...")
        time.sleep(REQUEST_DELAY_SECONDS)

        # --- Fetch the page ---
        print(f"Fetching {CURRENT_URL}...")
        response = requests.get(CURRENT_URL, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        print("Fetch successful.")
        page_html = response.text
        page_soup = BeautifulSoup(page_html, "lxml")

        # --- Extract Content ---
        print(
            f"Looking for content container with selector: '{CONTENT_CONTAINER_SELECTOR}'..."
        )
        # Find the main container div using the CSS selector
        content_container = page_soup.select_one(CONTENT_CONTAINER_SELECTOR)

        if content_container:
            print(
                f"Found content container. Looking for '<{CONTENT_SECTION_TAG}>' tags within it..."
            )
            # Find all <section> tags within that container
            # 'recursive=True' is the default, searching all descendants.
            # If sections are only direct children, use recursive=False.
            sections = content_container.find_all(CONTENT_SECTION_TAG)

            if sections:
                print(
                    f"Found {len(sections)} '<{CONTENT_SECTION_TAG}>' tag(s). Combining content..."
                )
                # Combine the HTML of all found sections into a single string
                combined_section_html = "".join(str(section) for section in sections)

                # --- Process Content ---
                print("Processing content...")
                page_title = extract_page_title(
                    page_soup
                )  # Get title from <title> or fallback
                filename = generate_filename(
                    CURRENT_URL, FILE_TYPE_TO_SAVE
                )  # Generate filename from URL

                if FILE_TYPE_TO_SAVE == FileType.MD:
                    print("Converting to Markdown...")
                    md_content = convert_to_md(combined_section_html)
                    print("Enriching Markdown...")
                    final_content = enrich_markdown(md_content, CURRENT_URL, page_title)
                else:  # FileType.TXT
                    print("Extracting text content...")
                    # Use BeautifulSoup to get text from the combined HTML
                    temp_soup = BeautifulSoup(combined_section_html, "html.parser")
                    text_content = temp_soup.get_text(separator="\n", strip=True)
                    # Add simple metadata for TXT
                    final_content = (
                        f"Title: {page_title}\nSource: {CURRENT_URL}\n\n---\n\n{text_content}"
                    )

                # --- Save Content ---
                print(f"Saving content to {filename} in {OUTPUT_DIR}...")
                save_section(
                    final_content, filename, OUTPUT_DIR
                )  # save_section handles printing success/failure
                processed = True

            else:
                print(
                    f"  Warning: No '<{CONTENT_SECTION_TAG}>' tags found within the container '{CONTENT_CONTAINER_SELECTOR}' on {CURRENT_URL}"
                )
        else:
            print(
                f"  Warning: Content container '{CONTENT_CONTAINER_SELECTOR}' not found on {CURRENT_URL}"
            )

    except requests.exceptions.HTTPError as e:
        print(
            f"  HTTP Error fetching {CURRENT_URL}: {e.response.status_code} {e.response.reason}"
        )
    except requests.exceptions.RequestException as e:
        print(f"  Network Error fetching {CURRENT_URL}: {e}")
    except Exception as e:
        print(f"  An unexpected error occurred while processing {CURRENT_URL}: {e}")
        import traceback

        traceback.print_exc()  # Print detailed traceback for unexpected errors

    # --- End of Script ---
    print("-" * 30)
    if processed:
        print("Scraping finished successfully for the page.")
    else:
        print("Scraping finished: Failed to process the page or find content.")
