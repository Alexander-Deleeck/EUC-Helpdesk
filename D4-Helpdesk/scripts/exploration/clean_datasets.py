import os
import pandas as pd
from pathlib import Path
from utils import clean_custom_fields, filter_summary_content, clean_description_content

def load_dataset(file_path: str | Path, load_columns: list[str]) -> pd.DataFrame:
    """
    Load CSV dataset with specified columns.
    
    Parameters:
        file_path: Path to the CSV file
        load_columns: List of column names to load
        
    Returns:
        pd.DataFrame: Loaded dataframe with specified columns
    """
    return pd.read_csv(file_path, sep=';', header=0, usecols=load_columns, encoding='utf-8')


def filter_english_content(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataframe to keep only English language content.
    
    Parameters:
        df: Input dataframe with 'Language' column
        
    Returns:
        pd.DataFrame: Filtered dataframe with only English content
    """
    return df[df['Language'] == 'EN - English'].copy()


def filter_related_issues(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out rows where 'Inward issue link (Relates)' is not empty.
    
    Parameters:
        df: Input dataframe
        
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    return df[df['Inward issue link (Relates)'].isna()].copy()


def clean_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Description column using clean_description_content function.
    
    Parameters:
        df: Input dataframe with 'Description' column
        
    Returns:
        pd.DataFrame: Dataframe with cleaned descriptions
    """
    df['Description'] = df['Description'].apply(clean_description_content)
    return df


def process_dataset(input_file: str | Path) -> pd.DataFrame:
    """
    Process a dataset through all cleaning steps.
    
    Parameters:
        input_file: Path to input CSV file
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    # Define columns to load
    load_columns = [
        "Summary",
        "Issue key",
        "Inward issue link (Relates)",
        "Issue id",
        "Parent id",
        "Issue Type",
        "Status",
        "Priority",
        "Resolution",
        "Creator",
        "Component/s",
        "Description",
        "Custom field (Category)",
        "Custom field (Classification)",
        "Custom field (Classification (Calculated))",
        "Custom field (Country)",
        "Custom field (Solution)",
        "Custom field (Language)",
        "Custom field (Summary Solution)",
        "Custom field (Summary in English)",
        "Custom field (User category)",
    ]

    # 1. Load dataset with specified columns
    print(f"Loading dataset from {input_file}")
    df = load_dataset(input_file, load_columns)
    initial_rows = len(df)
    print(f"Initial number of rows: {initial_rows}")

    # 2. Clean custom fields
    df = clean_custom_fields(df)
    print("Custom fields cleaned")
    
    # 3. Filter English content
    df = filter_english_content(df)
    print(f"Rows after English filter: {len(df)} ({len(df)/initial_rows:.1%} remaining)")

    # 4. Filter related issues
    df = filter_related_issues(df)
    print(f"Rows after related issues filter: {len(df)} ({len(df)/initial_rows:.1%} remaining)")

    # 5. Filter summary content
    df = filter_summary_content(df)
    print(f"Rows after summary content filter: {len(df)} ({len(df)/initial_rows:.1%} remaining)")

    # 6. Clean descriptions
    df = clean_descriptions(df)
    print("Descriptions cleaned")

    return df


def save_cleaned_dataset(df: pd.DataFrame, input_file: str | Path) -> None:
    """
    Save cleaned dataframe to a new CSV file.
    
    Parameters:
        df: Cleaned dataframe to save
        input_file: Original input file path (used to determine output path)
    """
    input_path = Path(input_file)
    output_path = input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}"
    df.to_csv(output_path, index=False, encoding="utf-8-sig", header=True)
    print(f"Cleaned dataset saved to {output_path}")
    return df

def main():
    """Main function to run the cleaning process."""
    
    scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    helpdesk_dir = os.path.dirname(scripts_dir)
    helpdesk_data_dir = os.path.join(helpdesk_dir, "helpdesk-data")
    
    csv_dir = os.path.join(helpdesk_data_dir, "JIRA FULL EXTRACT", "CSV")
    
    for idx, filenum in enumerate(range(1,5)):
        input_path = Path(f"{csv_dir}/Export {filenum}.csv")
    
        try:
            # Process the dataset
            df = process_dataset(input_path)
            
            # Save the cleaned dataset
            df = save_cleaned_dataset(df, input_path)
            print(f"\n{'='*50}{idx+1}{'='*50}\nCleaning pipeline completed. Sample of cleaned dataset:\n\n{df.head(5)}")
        except Exception as e:
            print(f"Error processing file: {e}")
            raise


if __name__ == "__main__":
    main()
