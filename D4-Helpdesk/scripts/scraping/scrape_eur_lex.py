# D4-Helpdesk\scripts\scraping\scrape_eur_lex.py
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
import os
import time
from utils_scraping import (
    extract_help_links,
    extract_section,
    extract_page_title,
    save_section,
    convert_to_md,
    enrich_markdown,
    generate_filename,
    FileType,
)

# --- Configuration ---
EURLEX_BASE_URL = "https://eur-lex.europa.eu"
START_URL = urljoin(EURLEX_BASE_URL, "/content/help.html")  # Construct start URL safely
OUTPUT_DIR = os.path.join(
    "..", "results", "eurlex", "help_docs_md"
)  # Relative path to save urljoin
FILE_TYPE_TO_SAVE = FileType.MD  # Choose MD or TXT
START_MARKER = "<!-- Help static content start -->"
END_MARKER = "<!-- Help static content end -->"
REQUEST_DELAY_SECONDS = 1  # Be polite to the server
REQUEST_TIMEOUT_SECONDS = 20  # Timeout for requests
# --------------------

# Ensure the base output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Starting scrape from: {START_URL}")
print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")

try:
    # --- Get initial help links ---
    response = requests.get(START_URL, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
    initial_soup = BeautifulSoup(response.text, "html.parser")
    help_links = extract_help_links(initial_soup, EURLEX_BASE_URL)
    print(f"Found {len(help_links)} potential help links.")

    if not help_links:
        print("No help links found on the starting page. Exiting.")
        exit()

    # --- Process each help link ---
    processed_count = 0
    for i, help_link in enumerate(help_links):
        print(f"\nProcessing link {i+1}/{len(help_links)}: {help_link}")
        try:
            # Add a delay
            time.sleep(REQUEST_DELAY_SECONDS)

            # Fetch the individual help page
            page_response = requests.get(help_link, timeout=REQUEST_TIMEOUT_SECONDS)
            page_response.raise_for_status()
            page_html = page_response.text
            page_soup = BeautifulSoup(page_html, "html.parser")

            # Extract the relevant HTML section
            section_html = extract_section(page_html, START_MARKER, END_MARKER)

            if section_html:
                # Extract page title
                page_title = extract_page_title(page_soup)

                # Generate filename
                filename = generate_filename(help_link, FILE_TYPE_TO_SAVE)

                if FILE_TYPE_TO_SAVE == FileType.MD:
                    # Convert HTML section to Markdown
                    md_content = convert_to_md(section_html)
                    # Enrich Markdown with metadata
                    final_content = enrich_markdown(md_content, help_link, page_title)
                else:  # FileType.TXT
                    # For TXT, maybe just strip HTML tags using BeautifulSoup's get_text()
                    temp_soup = BeautifulSoup(section_html, "html.parser")
                    final_content = temp_soup.get_text(separator="\n", strip=True)
                    # Optionally, add simpler metadata for TXT
                    final_content = f"Title: {page_title}\nSource: {help_link}\n\n---\n\n{final_content}"

                # Save the processed content
                save_section(final_content, filename, OUTPUT_DIR)
                processed_count += 1
            else:
                print(f"Content section not found between markers for: {help_link}")

        except requests.exceptions.RequestException as e:
            print(f"Error fetching {help_link}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {help_link}: {e}")

    print(
        f"\nScraping finished. Successfully processed and saved {processed_count}/{len(help_links)} pages."
    )

except requests.exceptions.RequestException as e:
    print(f"Failed to fetch the initial page {START_URL}: {e}")
except Exception as e:
    print(f"An unexpected error occurred during initial setup: {e}")
