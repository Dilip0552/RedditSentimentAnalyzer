import os
import sys
import json
import logging
import glob
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_all_datasets(max_rows=20000):
    """
    Process all CSV files in the attached_assets directory
    and create pre-processed analysis files for each.
    
    Args:
        max_rows (int): Maximum number of rows to process per dataset
    """
    # Get all CSV files in attached_assets
    attached_assets_dir = os.path.join(os.getcwd(), 'attached_assets')
    csv_files = glob.glob(os.path.join(attached_assets_dir, '*.csv'))
    
    # Check if we have any files
    if not csv_files:
        logging.error("No CSV files found in attached_assets directory")
        return
    
    logging.info(f"Found {len(csv_files)} CSV files to process")
    
    # Create datasets metadata file to track all datasets
    datasets_metadata = []
    
    # Process each CSV file
    for file_path in csv_files:
        try:
            # Get dataset name and create friendly display name
            file_name = os.path.basename(file_path)
            dataset_id = os.path.splitext(file_name)[0]
            
            # Parse the filename to extract topic
            parts = dataset_id.split('_')
            if len(parts) >= 3 and parts[0] == "comments":
                # Format: comments_<TOPIC>_<SUBREDDIT>.csv
                if "all_subreddits" in dataset_id:
                    # For "all_subreddits" format
                    topic_parts = parts[1:-1]  # Everything between 'comments_' and '_all_subreddits'
                    topic = ' '.join(topic_parts)
                    subreddit = "all"
                else:
                    # For specific subreddit format
                    topic_parts = parts[1:-1]  # Everything between 'comments_' and the last part
                    topic = ' '.join(topic_parts)
                    subreddit = parts[-1]
                
                # Create display name
                display_name = topic.replace('_', ' ')
                
                # Clean up display names with special characters
                display_name = display_name.replace('-', ' ')
                if "'" in display_name:
                    display_name = display_name.replace("'s", "'s")
                
                # Count lines in file to get comment count
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        comment_count = sum(1 for _ in f) - 1  # Subtract header
                except Exception as e:
                    logging.warning(f"Could not count lines in {file_path}: {e}")
                    comment_count = 0
                
                logging.info(f"Processing dataset: {display_name} ({comment_count} comments)")
                
                # Run the analysis script
                from fix_analysis import fix_dataset_analysis
                result_file = fix_dataset_analysis(file_path, max_rows)
                
                if result_file:
                    # Get summary data
                    dataset_dir = os.path.dirname(result_file)
                    summary_file = os.path.join(dataset_dir, "summary.json")
                    
                    subreddits = []
                    sentiment = {"positive": 0, "negative": 0, "neutral": 0}
                    
                    # If summary file exists, read sentiment data
                    if os.path.exists(summary_file):
                        with open(summary_file, 'r') as f:
                            summary = json.load(f)
                            if "positive_pct" in summary:
                                sentiment["positive"] = summary["positive_pct"]
                            if "negative_pct" in summary:
                                sentiment["negative"] = summary["negative_pct"]
                            if "neutral_pct" in summary:
                                sentiment["neutral"] = summary["neutral_pct"]
                    
                    # Try to get subreddits from full analysis
                    full_analysis_file = os.path.join(dataset_dir, "full_analysis.json")
                    if os.path.exists(full_analysis_file):
                        with open(full_analysis_file, 'r') as f:
                            analysis = json.load(f)
                            if "subreddits" in analysis and isinstance(analysis["subreddits"], list):
                                subreddits = [sub["name"] for sub in analysis["subreddits"]]
                    
                    # Add dataset metadata
                    datasets_metadata.append({
                        "id": dataset_id,
                        "display_name": display_name,
                        "topic": topic,
                        "subreddit": subreddit,
                        "comment_count": comment_count,
                        "processed_comments": max_rows if max_rows and max_rows < comment_count else comment_count,
                        "subreddits": subreddits,
                        "sentiment": sentiment
                    })
                
            else:
                logging.warning(f"File {file_name} doesn't match expected naming pattern, skipping")
                continue
                
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save metadata
    if datasets_metadata:
        metadata_dir = os.path.join(os.getcwd(), "data", "preprocessed_results")
        os.makedirs(metadata_dir, exist_ok=True)
        
        metadata_file = os.path.join(metadata_dir, "datasets_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(datasets_metadata, f, indent=2)
        
        logging.info(f"Saved metadata for {len(datasets_metadata)} datasets to {metadata_file}")
    else:
        logging.warning("No datasets were successfully processed")

if __name__ == "__main__":
    max_rows = 20000
    if len(sys.argv) > 1:
        try:
            max_rows = int(sys.argv[1])
        except ValueError:
            pass
    
    process_all_datasets(max_rows)