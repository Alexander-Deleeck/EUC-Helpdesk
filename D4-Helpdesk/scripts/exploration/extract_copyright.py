import pandas as pd
import re
import extract_msg
from pathlib import Path
from datetime import datetime, timezone
from dateutil import parser as date_parser
import email.utils
import os
import json # Added for JSON output


# --- Configuration ---
# Define the email domains considered internal to your organization.
# Add all relevant internal domains here. Adjust as needed.
INTERNAL_DOMAINS = {'publications.europa.eu', 'another.internal.domain'} # *** ADJUST THESE DOMAINS ***

# --- Paths ---
REGISTER_PATH = r"C:\Users\deleeal\Documents\European Comission - EUC\Helpdesk-Butler\D2-Copyright\copyright-data\2023_REGISTER.xlsx" # *** ADJUST IF NEEDED ***
ARCHIVE_PATH = r"C:\Users\deleeal\Documents\European Comission - EUC\Helpdesk-Butler\D2-Copyright\copyright-data\copyright-archive-new" # *** ADJUST IF NEEDED ***
OUTPUT_DIR = Path("C:/Users/deleeal/Documents/European Comission - EUC/Helpdesk-Butler/D4-Helpdesk/data/processed") # *** ADJUST IF NEEDED ***
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

# --- Load Register Data ---
try:
    register_df = pd.read_excel(REGISTER_PATH, header=0, sheet_name="2023 COP")
    # Clean potential whitespace issues in column names
    register_df.columns = register_df.columns.str.strip()
    # Set 'igaro' as index for faster lookup
    if 'igaro' in register_df.columns:
        register_df.set_index('igaro', inplace=True)
        print("Register loaded successfully with 'igaro' as index.")
    else:
        print("Error: 'igaro' column not found in register file. Metadata lookup will fail.")
        register_df = None # Indicate failure
except FileNotFoundError:
    print(f"Error: Register file not found at {REGISTER_PATH}")
    register_df = None
except Exception as e:
    print(f"Error loading or processing register file: {e}")
    register_df = None


# --- Helper Functions ---

def parse_sender(from_line: str) -> str | None:
    """Extracts the email address from a 'From:' line."""
    if not from_line: return None
    real_name, email_address = email.utils.parseaddr(from_line.replace('From:', '', 1).strip())
    return email_address if '@' in email_address else None

def parse_sent_date(sent_line: str) -> datetime | None:
    """Parses the date string from a 'Sent:' line into a datetime object."""
    if not sent_line: return None
    # Extract date part after the header keyword (Sent, Date, etc.)
    # Be robust against different keywords (case-insensitive)
    match = re.match(r"^(Sent|Envoyé|Gesendet|Date):\s*(.*)", sent_line.strip(), re.IGNORECASE)
    if not match: return None
    date_str = match.group(2)
    try:
        dt = date_parser.parse(date_str)
        # Optional: Make timezone aware if needed, but naive comparison often works if consistent
        # if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        #     dt = dt.replace(tzinfo=timezone.utc) # Example: Assume UTC if naive
        return dt
    except (ValueError, OverflowError, TypeError) as e:
        print(f"Warning: Could not parse date: '{date_str}'. Error: {e}")
        return None

def parse_subject(subject_line: str) -> str | None:
    """Extracts the subject text from a 'Subject:' line."""
    if not subject_line: return None
    # Extract subject part after the header keyword (Subject, Objet, etc.)
    match = re.match(r"^(Subject|Objet|Betreff):\s*(.*)", subject_line.strip(), re.IGNORECASE)
    if not match: return None
    return match.group(2).strip()

def is_external_sender(email_address: str | None, internal_domains: set[str]) -> bool:
    """Checks if the email address belongs to an external domain."""
    if not email_address or '@' not in email_address: return False
    domain = email_address.split('@')[1].lower()
    return domain not in internal_domains

# --- Core Logic ---

def extract_first_external_email_details(msg_path: Path, internal_domains: set[str]) -> dict | None:
    """
    Opens a .msg file, parses the email thread in its body, identifies the earliest
    email sent by an external user, and returns its details (body, date, subject).

    Args:
        msg_path: Path to the .msg file.
        internal_domains: A set of lowercased internal email domains.

    Returns:
        A dictionary {'body': str, 'date': datetime, 'subject': str}
        or None if not found or error occurs.
    """
    try:
        msg = extract_msg.openMsg(str(msg_path))
        full_body = msg.body
        primary_subject = msg.subject # Get subject from main headers
        primary_date = msg.date # Get date from main headers (often already datetime)
        primary_sender = msg.sender # Get sender from main headers
    except Exception as e:
        print(f"Error opening or reading MSG file {msg_path}: {e}")
        return None

    if not full_body:
        # If body is empty, maybe it's an appointment or weird item?
        # Try using primary headers if sender is external
        if primary_sender and primary_date and is_external_sender(primary_sender, internal_domains):
             print(f"Warning: Empty body in {msg_path}, using primary headers.")
             return {
                 "body": "", # No body to extract
                 "date": primary_date,
                 "subject": primary_subject
             }
        print(f"Warning: Empty body and cannot determine sender/date from primary headers for {msg_path}")
        return None

    # Regex to split the body into individual email messages.
    splitter_pattern = re.compile(
        r"^\s*(?:[-_]+\s*)*"
        r"(?:From|De|Von):\s+.*"
        r"(?:\n^\s*(?:Sent|Envoyé|Gesendet|Date):\s+.*)?"
        , re.MULTILINE | re.IGNORECASE
    )
    parts = splitter_pattern.split(full_body)
    delimiters = splitter_pattern.findall(full_body)

    email_segments = []
    if parts:
        if len(parts) > 1 :
            for i in range(len(delimiters)):
                segment = delimiters[i] + parts[i+1]
                email_segments.append(segment.strip())
        # Handle edge case where the entire body might be the first email
        elif parts[0].strip():
             first_part_lower = parts[0].strip().lower()
             # Check if the *first part itself* contains standard headers
             if re.search(r"^(from|de|von):", first_part_lower, re.MULTILINE) and \
                re.search(r"^(sent|envoyé|gesendet|date):", first_part_lower, re.MULTILINE):
                 email_segments.append(parts[0].strip())
             # If no delimiters *at all*, consider the whole body as one segment potentially
             elif len(delimiters) == 0:
                 email_segments.append(parts[0].strip())


    # If parsing fails, check primary headers as a fallback
    if not email_segments:
        print(f"Warning: Could not parse distinct email segments in {msg_path}. Trying primary headers.")
        if primary_sender and primary_date and is_external_sender(primary_sender, internal_domains):
            print(f"Info: Using primary msg headers for {msg_path} as fallback.")
            # Need to determine the actual *first* message body in this case.
            # It might be the whole body, or just the part before the first reply signature/header.
            # Let's try returning the body up to the first common separator.
            separator_match = re.search(
                r"^\s*(?:From:|-----Original Message-----|_+|Sent:)", full_body,
                re.MULTILINE | re.IGNORECASE
            )
            body_to_return = full_body[:separator_match.start()].strip() if separator_match else full_body.strip()
            return {
                "body": body_to_return,
                "date": primary_date,
                "subject": primary_subject
            }
        print(f"Warning: Could not parse segments and primary headers are not from external sender or lack date for {msg_path}.")
        return None

    # --- Parsing loop ---
    earliest_details = None

    for segment in email_segments:
        lines = segment.splitlines()
        from_line, sent_line, subject_line = None, None, None
        body_lines = []
        header_parsing_done = False

        current_headers = {'from': None, 'sent': None, 'subject': None}
        temp_body_lines = []

        for i, line in enumerate(lines):
            if header_parsing_done:
                body_lines.append(line)
                continue

            line_strip = line.strip()
            line_lower = line_strip.lower()

            # Try to identify header lines
            if line_lower.startswith(("from:", "de:", "von:")):
                current_headers['from'] = line_strip
            elif line_lower.startswith(("sent:", "envoyé:", "gesendet:", "date:")):
                current_headers['sent'] = line_strip
            elif line_lower.startswith(("to:", "à:", "an:")):
                pass # Ignore
            elif line_lower.startswith("cc:"):
                pass # Ignore
            elif line_lower.startswith(("subject:", "objet:", "betreff:")):
                 current_headers['subject'] = line_strip
            # Detect end of headers (empty line or line not matching a header pattern)
            elif not line_strip or not re.match(r"^(From|De|Von|Sent|Envoyé|Gesendet|Date|To|À|An|Cc|Subject|Objet|Betreff):", line_strip, re.IGNORECASE):
                 # If we've seen at least From and Sent, assume headers are done
                 if current_headers['from'] and current_headers['sent']:
                     header_parsing_done = True
                     # Add the current line to body *if* it wasn't an empty separator line
                     if line_strip:
                         body_lines.append(line)
                 else:
                     # Headers haven't started properly or are incomplete, treat as body
                     temp_body_lines.append(line)
            # Keep accumulating potential header lines if none of the above conditions met yet

        # If header parsing never finished properly, use temp body lines
        if not header_parsing_done:
             body_lines = temp_body_lines

        # Reconstruct body for this segment
        current_body = "\n".join(body_lines).strip()

        # Extract details from found header lines
        sender_email = parse_sender(current_headers['from'])
        sent_date = parse_sent_date(current_headers['sent'])
        current_subject = parse_subject(current_headers['subject'])
        # If subject not found in segment headers, use the primary msg subject
        if current_subject is None:
            current_subject = primary_subject

        # --- Check if this is the earliest external email ---
        if sender_email and sent_date and current_body:
            if is_external_sender(sender_email, internal_domains):
                if earliest_details is None or sent_date < earliest_details['date']:
                    earliest_details = {
                        "body": current_body,
                        "date": sent_date,
                        "subject": current_subject # Store the subject found
                    }
                    # print(f"*** Found new earliest external: {sent_date} ***") # Debug

    # --- Fallback after parsing loop ---
    # If loop didn't find anything, but primary headers suggest external origin
    if earliest_details is None:
        if primary_sender and primary_date and is_external_sender(primary_sender, internal_domains):
             print(f"Info: Parsing failed, using primary msg headers for {msg_path} as final fallback.")
             separator_match = re.search(
                r"^\s*(?:From:|-----Original Message-----|_+|Sent:)", full_body,
                re.MULTILINE | re.IGNORECASE
             )
             body_to_return = full_body[:separator_match.start()].strip() if separator_match else full_body.strip()
             return {
                 "body": body_to_return,
                 "date": primary_date,
                 "subject": primary_subject
             }

    return earliest_details


# --- Filtering Functions (Using your provided versions) ---

def first_msg_file(filelist: list[str]) -> list[str] | None:
    """
    Filters a list of filenames to find those likely representing the
    original email conversation thread.
    Returns a list of matching filenames (often just one).
    """
    msg_files = [file for file in filelist if file.lower().endswith(".msg")]
    if not msg_files:
        return None

    # Filter out common reply/forward prefixes and specific patterns
    exclude_prefixes = ["fw_", "aw_", "re_", "wg_"] # Using common lowercase prefixes
    exclude_patterns = ["final -"]

    filtered_msg_files = []
    for file in msg_files:
        fname_lower = file.lower()
        # Check prefixes
        if any(fname_lower.startswith(prefix) for prefix in exclude_prefixes):
            continue
        # Check other patterns
        if any(pattern in fname_lower for pattern in exclude_patterns):
            continue
        filtered_msg_files.append(file)

    # Prefer files NOT ending with space + (number).msg
    pattern_no_number = r'^(?!.* \(\d+\)\.msg$).*\.msg$'
    original_files = [file for file in filtered_msg_files if re.match(pattern_no_number, file, re.IGNORECASE)]

    if original_files:
        return original_files # Return list of files without (number)
    elif filtered_msg_files:
        # Fallback: return files that passed prefix/pattern filter but might have (number)
        return filtered_msg_files
    else:
        # If all files were filtered out, maybe the original *did* have a prefix?
        # Return None as per previous logic, or consider returning msg_files as last resort.
        return None

def check_register(register: pd.DataFrame | None, folder_name: str) -> bool:
    """
    Check if the folder name is in the register index and meets filter criteria.
    """
    if register is None or not isinstance(register, pd.DataFrame):
        print("Warning: Register DataFrame is not available for checking.")
        return False # Cannot check if register is not loaded

    if folder_name not in register.index:
        # print(f"Debug: Folder '{folder_name}' not found in register index.")
        return False

    try:
        # Get the row using the index
        matching_row = register.loc[folder_name]

        # Apply filter: Country should not be "EU"
        # Handle potential multiple rows if 'igaro' isn't unique (use .iloc[0] if needed)
        country = matching_row['Country']
        if isinstance(country, pd.Series): # Handle case if index wasn't unique
             country = country.iloc[0]

        if pd.isna(country) or country != "EU":
             return True # Keep if country is not EU or is NaN
        else:
             # print(f"Debug: Skipping folder '{folder_name}' because Country is EU.")
             return False # Skip if country is EU

    except KeyError:
        # This case should be covered by `folder_name not in register.index`
        print(f"Warning: KeyError looking up '{folder_name}' in register index.")
        return False
    except Exception as e:
        print(f"Error checking register for folder '{folder_name}': {e}")
        return False

# --- Main Processing Loop ---

output_data = []
processed_count = 0
max_process =10 # float('inf') # Set to a number (e.g., 10) for testing, or float('inf') for all

print(f"Starting processing of archive: {ARCHIVE_PATH}")
print(f"Internal domains configured: {INTERNAL_DOMAINS}")

for folder_name in os.listdir(ARCHIVE_PATH):
    
    folder_path = Path(ARCHIVE_PATH) / folder_name
    if not folder_path.is_dir():
        continue # Skip items that are not directories

    # --- Apply Register Filter ---
    if not check_register(register_df, folder_name):
        # print(f"Skipping folder (failed register check): {folder_name}")
        continue

    # --- Limit Processing for Testing ---
    if processed_count >= max_process:
        print(f"Reached processing limit ({max_process}). Stopping.")
        break

    print(f"\n--- Processing Folder [{processed_count+1}]: {folder_name} ---")
    processed_count += 1

    filelist = os.listdir(folder_path)

    # Find candidate message file(s)
    candidate_files = first_msg_file(filelist)

    email_details = None
    target_msg_filename = None

    if candidate_files:
        # Process the first candidate file if multiple are returned
        target_msg_filename = candidate_files[0]
        target_msg_path = folder_path / target_msg_filename
        print(f"  Filtered to message file: {target_msg_filename}")
        email_details = extract_first_external_email_details(target_msg_path, INTERNAL_DOMAINS)
    else:
        print(f"  No suitable '.msg' file found after filtering.")

    # --- Prepare output dictionary ---
    result_item = {
        "folder": folder_name,
        "source_msg": target_msg_filename, # Will be None if no candidate found
        "first_external_email_body": None,
        "first_external_email_date": None,
        "first_external_email_subject": None,
        "date_received": None,
        "country": None,
        "title": None,
        "conclusion": None,
        "advice_reuse": None
    }

    # --- Populate with extracted email details ---
    if email_details:
        print(f"  Extracted First External Email Details:")
        print(f"    Date: {email_details['date']}")
        print(f"    Subject: {email_details['subject']}")
        print(f"    Body Preview: {email_details['body'][:200].strip()}...")
        result_item["first_external_email_body"] = email_details["body"]
        # Ensure date is serializable (e.g., ISO format string) for JSON
        result_item["first_external_email_date"] = email_details["date"].isoformat() if isinstance(email_details["date"], datetime) else email_details["date"]
        result_item["first_external_email_subject"] = email_details["subject"]
    else:
        print(f"  Could not extract first external email details.")

    # --- Populate with metadata from register ---
    if register_df is not None and folder_name in register_df.index:
        try:
            meta_row = register_df.loc[folder_name]
            # Handle potential Series if index wasn't unique (take first)
            if isinstance(meta_row, pd.DataFrame):
                meta_row = meta_row.iloc[0]

            # Use .get() for safer access in case columns are missing unexpectedly
            result_item["date_received"] = meta_row.get("Request received")
            result_item["country"] = meta_row.get("Country")
            result_item["title"] = meta_row.get("Title")
            result_item["conclusion"] = meta_row.get("Conclusion")
            result_item["advice_reuse"] = meta_row.get("Advice or Reuse?")

            # Convert 'Request received' date to ISO format if it's a datetime
            if isinstance(result_item["date_received"], datetime):
                 result_item["date_received"] = result_item["date_received"].isoformat()
            # Handle potential Pandas NaT or None before trying isoformat
            elif pd.isna(result_item["date_received"]):
                 result_item["date_received"] = None


            print(f"  Fetched metadata: Date Received={result_item['date_received']}, Country={result_item['country']}")

        except Exception as e:
            print(f"  Error fetching metadata for folder {folder_name}: {e}")
    else:
        print(f"  Metadata not found in register for folder {folder_name}.")

    output_data.append(result_item)

# --- Save results ---
if output_data:
    output_df = pd.DataFrame(output_data)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save CSV
    output_csv_path = OUTPUT_DIR / f"extracted_copyright_details_{timestamp}.csv"
    try:
        output_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n\nResults saved to CSV: {output_csv_path}")
    except Exception as e:
        print(f"\n\nError saving results to CSV: {e}")

    # Save JSON
    output_json_path = OUTPUT_DIR / f"extracted_copyright_details_{timestamp}.json"
    try:
        # Use to_json for better handling of data types like dates
        output_df.to_json(output_json_path, orient='records', date_format='iso', force_ascii=False, indent=4)
        # Alternative using json module:
        # with open(output_json_path, 'w', encoding='utf-8') as f:
        #     json.dump(output_data, f, ensure_ascii=False, indent=4)
        print(f"Results also saved to JSON: {output_json_path}")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")

else:
    print("\n\nNo data was processed or extracted.")

print("\nScript finished.")