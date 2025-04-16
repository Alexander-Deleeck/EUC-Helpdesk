# D4-Helpdesk\scripts\scraping\utils_scraping.py
from enum import Enum
from urllib.parse import urljoin, urlparse
import os
import re
from bs4 import BeautifulSoup, Comment
from markdownify import markdownify as md  # Renamed import for brevity


class FileType(Enum):
    TXT = "txt"
    MD = "md"


def extract_help_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    """
    Extracts absolute URLs pointing to help pages from a BeautifulSoup object.

    Args:
        soup: BeautifulSoup object of the page containing links.
        base_url: The base URL to resolve relative links.

    Returns:
        A list of absolute URLs for help pages.
    """
    links = set()  # Use a set to avoid duplicates
    for link in soup.find_all("a", href=True):
        href = link.get("href")
        # Check if the link contains the specific help path segment
        if href and "/content/help/" in href:
            absolute_url = urljoin(base_url, href)
            # Optional: Add further filtering if needed (e.g., exclude specific patterns)
            links.add(absolute_url)
    return list(links)


def extract_section(
    html_content: str, start_marker: str, end_marker: str
) -> str | None:
    """Extracts HTML between comment markers (Used for EUR-Lex)."""
    # ... (implementation remains the same) ...
    try:
        start_index = html_content.index(start_marker) + len(start_marker)
        end_index = html_content.index(end_marker, start_index)
        return html_content[start_index:end_index].strip()
    except ValueError:
        print(f"Warning: Could not find markers '{start_marker}' or '{end_marker}'")
        return None


def extract_page_title(soup: BeautifulSoup) -> str:
    """Extracts the title from the page's <title> tag."""
    # ... (implementation remains the same) ...
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    # Fallback to h1 if title is generic or missing
    h1 = soup.find("h1")
    if h1 and h1.string:
        return h1.string.strip()
    return "Untitled"


def convert_to_md(html_section: str) -> str:
    """Converts an HTML string snippet to Markdown."""
    # ... (implementation remains the same) ...
    if not html_section:
        return ""
    # Consider adding 'strip=['script', 'style']' to markdownify options
    # if these tags sometimes appear within the <article>
    markdown_content = md(
        html_section, heading_style="ATX", bullets="-", strip=["script", "style"]
    ).strip()
    return markdown_content


def enrich_markdown(markdown_content: str, original_url: str, page_title: str) -> str:
    """Adds metadata (title, source URL) to the Markdown content."""
    # ... (implementation remains the same) ...
    header = f"# {page_title}\n\n"
    source_info = f"**Source:** <{original_url}>\n\n---\n\n"
    return header + source_info + markdown_content


def generate_filename(url: str, filetype: FileType) -> str:
    """Generates a sanitized filename from a URL."""
    # ... (implementation remains the same) ...
    parsed_url = urlparse(url)
    path_parts = [part for part in parsed_url.path.split("/") if part]
    if not path_parts:
        base_name = "index"
    else:
        # Use directory structure for uniqueness if paths are like /a/b/ and /c/b/
        # Join parts except the potential filename part
        dir_path = "_".join(path_parts[:-1])
        filename_part = path_parts[-1]
        # Remove common extensions like .html, .htm, or trailing slashes if they are file names
        base_name_part = (
            os.path.splitext(filename_part)[0]
            if "." in filename_part
            else filename_part
        )
        base_name = f"{dir_path}_{base_name_part}" if dir_path else base_name_part
        base_name = base_name.strip("_")  # Clean up if path starts with /

    # Sanitize the base name
    sanitized_name = re.sub(r"[^\w\-]+", "_", base_name)
    sanitized_name = sanitized_name.strip("_").strip("-")
    if not sanitized_name:
        sanitized_name = "page"  # Fallback if sanitization results in empty string

    # Prevent overly long filenames (optional)
    max_len = 100
    if len(sanitized_name) > max_len:
        sanitized_name = sanitized_name[:max_len]

    return f"{sanitized_name}.{filetype.value}"


def save_section(content: str, filename: str, output_dir: str):
    """Saves the content to a file in the specified directory."""
    # ... (implementation remains the same) ...
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    try:
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(content)
        print(f"Successfully saved: {filepath}")
    except IOError as e:
        print(f"Error saving file {filepath}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving {filepath}: {e}")


def extract_dataeuropa_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    """
    Extracts absolute URLs belonging to the same domain as the base_url
    (assumed to be data.europa.eu or related GitLab pages) found within a
    BeautifulSoup object. Handles relative links and ensures uniqueness.
    Excludes common non-HTML file links.
    """
    links = set()
    try:
        target_netloc = urlparse(base_url).netloc
        if not target_netloc:
            print(f"Warning: Could not determine domain from base_url: {base_url}")
            return []
    except ValueError as e:
        print(f"Error parsing base_url '{base_url}': {e}")
        return []

    for link in soup.find_all("a", href=True):
        href = link.get("href")

        # Clean and basic validation
        if not href:
            continue
        href = href.strip()
        if (
            not href
            or href.startswith("#")
            or href.startswith("mailto:")
            or href.startswith("tel:")
        ):
            continue

        # Skip links pointing directly to common non-page file types
        excluded_extensions = {
            ".pdf",
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".zip",
            ".xml",
            ".json",
            ".svg",
            ".ico",
            ".css",
            ".js",
        }
        parsed_href_path = urlparse(href).path.lower()
        if any(parsed_href_path.endswith(ext) for ext in excluded_extensions):
            continue

        try:
            absolute_url = urljoin(base_url, href)
            parsed_absolute_url = urlparse(absolute_url)

            # Check domain and scheme
            if (
                parsed_absolute_url.netloc == target_netloc
                and parsed_absolute_url.scheme in ["http", "https"]
            ):

                # Optional: Clean fragment from URL before adding
                cleaned_url = parsed_absolute_url._replace(fragment="").geturl()
                links.add(cleaned_url)

        except ValueError as e:
            print(f"Warning: Could not parse/join URL from href '{href}': {e}")
        except Exception as e:
            print(
                f"Warning: Error processing href '{href}' with base '{base_url}': {e}"
            )

    return list(links)
