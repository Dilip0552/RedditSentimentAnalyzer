import os
import json
import logging
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directory structure
DATA_DIR = os.path.join(os.getcwd(), "data")
RESULTS_DIR = os.path.join(DATA_DIR, "preprocessed_results")

def get_dataset_analysis(dataset_id, include_emotions=False):
    """
    Get dataset analysis results, prioritizing full analysis if available.
    
    Args:
        dataset_id (str): Dataset identifier
        include_emotions (bool): Whether to include emotion data
        
    Returns:
        dict: Analysis results or None if not found
    """
    try:
        # Check if we have full analysis results
        dataset_dir = os.path.join(RESULTS_DIR, dataset_id)
        full_analysis_path = os.path.join(dataset_dir, "full_analysis.json")
        
        if os.path.exists(full_analysis_path):
            logging.info(f"Loading full analysis from {full_analysis_path}")
            with open(full_analysis_path, 'r') as f:
                results = json.load(f)
                
                # Check if we need to add emotions data
                if include_emotions and "emotions" not in results:
                    # Try to find emotions data in another file
                    emotions_path = os.path.join(dataset_dir, "subreddit_all_with_emotions.json")
                    if os.path.exists(emotions_path):
                        with open(emotions_path, 'r') as f2:
                            emotions_data = json.load(f2)
                            if "emotions" in emotions_data:
                                results["emotions"] = emotions_data["emotions"]
                
                return results
        
        # Try to find a cached analysis
        suffix = "subreddit_all_with_emotions.json" if include_emotions else "subreddit_all.json"
        path = os.path.join(dataset_dir, suffix)
        
        if os.path.exists(path):
            logging.info(f"Loading cached analysis from {path}")
            with open(path, 'r') as f:
                return json.load(f)
                
        # No analysis found
        logging.warning(f"No analysis found for dataset {dataset_id}")
        return None
        
    except Exception as e:
        logging.error(f"Error getting dataset analysis: {e}")
        return None

def safe_read_csv(file_path, chunk_size=None):
    """
    Safely read a CSV file with proper parameters to handle large datasets.
    
    Args:
        file_path (str): Path to CSV file
        chunk_size (int): Size of chunks to read, or None to read all at once
        
    Returns:
        pandas.DataFrame or iterator of DataFrames if chunk_size is provided
    """
    # Count lines in file for logging
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            line_count = sum(1 for _ in f)
        logging.info(f"CSV file contains {line_count} lines (including header)")
    except Exception as e:
        logging.warning(f"Could not count lines in file: {e}")
    
    # Parameters optimized for large files
    read_params = {
        'dtype': {'comment': str, 'subreddit': str},  # Ensure proper column types
        'low_memory': False,                          # Avoid mixed type inference issues
        'on_bad_lines': 'skip',                       # Skip problematic lines
        'encoding': 'utf-8',                          # UTF-8 encoding
        'escapechar': '\\',                           # Handle escaped characters
        'quotechar': '"'                              # Handle quoted text
    }
    
    # Add chunk_size parameter if provided
    if chunk_size:
        read_params['chunksize'] = chunk_size
    
    # Read CSV with parameters
    df = pd.read_csv(file_path, **read_params)
    
    if not chunk_size:
        logging.info(f"Successfully loaded {len(df)} rows into DataFrame")
    
    return df

if __name__ == "__main__":
    # Test functions
    dataset_id = "comments_Banning_TikTok_in_the_US_all_subreddits"
    results = get_dataset_analysis(dataset_id, include_emotions=True)
    
    if results:
        print(f"Found analysis with {results['total']} comments")
        print(f"Positive: {results['positive']['count']} ({results['positive']['percentage']}%)")
        print(f"Negative: {results['negative']['count']} ({results['negative']['percentage']}%)")
        print(f"Neutral: {results['neutral']['count']} ({results['neutral']['percentage']}%)")
        
        if 'emotions' in results:
            print(f"Emotions: {results['emotions']}")
    else:
        print(f"No analysis found for {dataset_id}")