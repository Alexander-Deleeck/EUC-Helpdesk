# D4-Helpdesk\scripts\scraping\scrape_data_europa.py
import requests
from bs4 import BeautifulSoup
import os
import time
from urllib.parse import urljoin, urlparse
from utils_scraping import (
    extract_dataeuropa_links,  # Use the specific link extractor
    extract_page_title,
    save_section,
    convert_to_md,
    enrich_markdown,
    generate_filename,
    FileType,
)

# --- Configuration ---
# Base URL of the documentation site
DATAEUROPA_DOCS_BASE_URL = "https://dataeuropa.gitlab.io"
# The starting path for the documentation section we care about
START_PATH = "/data-provider-manual/"
START_URL = urljoin(DATAEUROPA_DOCS_BASE_URL, START_PATH)

# Output directory relative to the script location
OUTPUT_DIR = os.path.join("..", "results", "dataeuropa", "dataeuropa_docs_md")
FILE_TYPE_TO_SAVE = FileType.MD  # Choose MD or TXT

# HTML tag containing the main content
CONTENT_TAG = "article"

# Scraping politeness and robustness
REQUEST_DELAY_SECONDS = 1  # Be polite to the server
REQUEST_TIMEOUT_SECONDS = 30  # Timeout for requests
MAX_PAGES_TO_SCRAPE = 500  # Safety limit to prevent infinite loops
# --------------------

# Ensure the base output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Crawler Setup ---
urls_to_visit = {START_URL}  # Set of URLs to scrape
visited_urls = set()  # Set of URLs already scraped or attempted
processed_count = 0
error_count = 0
# --------------------

print(f"Starting scrape from: {START_URL}")
print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
print(f"Content tag: <{CONTENT_TAG}>")
print("-" * 30)

while urls_to_visit and processed_count < MAX_PAGES_TO_SCRAPE:
    # Get the next URL to visit
    current_url = urls_to_visit.pop()

    # Skip if already visited
    if current_url in visited_urls:
        continue

    print(f"Processing ({len(visited_urls)+1}): {current_url}")
    visited_urls.add(current_url)

    try:
        # Add a delay
        time.sleep(REQUEST_DELAY_SECONDS)

        # --- Fetch the page ---
        response = requests.get(current_url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        page_html = response.text
        page_soup = BeautifulSoup(page_html, "html.parser")

        # --- Extract Content ---
        content_area = page_soup.find(CONTENT_TAG)

        if content_area:
            # Convert the found <article> tag and its contents to string
            section_html = str(content_area)

            # --- Process Content ---
            page_title = extract_page_title(page_soup)  # Get title from <title> or <h1>
            filename = generate_filename(current_url, FILE_TYPE_TO_SAVE)

            if FILE_TYPE_TO_SAVE == FileType.MD:
                md_content = convert_to_md(section_html)
                final_content = enrich_markdown(md_content, current_url, page_title)
            else:  # FileType.TXT
                temp_soup = BeautifulSoup(section_html, "html.parser")
                text_content = temp_soup.get_text(separator="\n", strip=True)
                final_content = f"Title: {page_title}\nSource: {current_url}\n\n---\n\n{text_content}"

            # --- Save Content ---
            save_section(final_content, filename, OUTPUT_DIR)
            processed_count += 1

            # --- Find New Links ---
            # Pass the current URL as base for relative link resolution
            new_links = extract_dataeuropa_links(page_soup, current_url)
            print(f"  Found {len(new_links)} potential links on page.")

            valid_new_links_count = 0
            for link in new_links:
                # Add only links that haven't been visited or queued
                if link not in visited_urls and link not in urls_to_visit:
                    # Optional: Add further checks, e.g., ensure link starts with START_PATH
                    # if urlparse(link).path.startswith(START_PATH):
                    urls_to_visit.add(link)
                    valid_new_links_count += 1
            if valid_new_links_count > 0:
                print(f"  Added {valid_new_links_count} new unique links to the queue.")

        else:
            print(
                f"  Warning: Content tag '<{CONTENT_TAG}>' not found on {current_url}"
            )
            error_count += 1

    except requests.exceptions.HTTPError as e:
        print(
            f"  HTTP Error fetching {current_url}: {e.response.status_code} {e.response.reason}"
        )
        error_count += 1
    except requests.exceptions.RequestException as e:
        print(f"  Network Error fetching {current_url}: {e}")
        error_count += 1
    except Exception as e:
        print(f"  An unexpected error occurred while processing {current_url}: {e}")
        import traceback

        traceback.print_exc()  # Print detailed traceback for unexpected errors
        error_count += 1

# --- End of Crawl ---
print("-" * 30)
if processed_count >= MAX_PAGES_TO_SCRAPE:
    print(f"Scraping stopped: Reached maximum page limit ({MAX_PAGES_TO_SCRAPE}).")
print(f"Scraping finished.")
print(f"Successfully processed and saved: {processed_count} pages.")
print(f"URLs encountered errors: {error_count}")
print(f"Total unique URLs visited: {len(visited_urls)}")
print(f"URLs remaining in queue: {len(urls_to_visit)}")
