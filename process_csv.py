import os
import sys
import json
import logging
import pandas as pd
import re
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check if data directory exists
DATA_DIR = os.path.join(os.getcwd(), "data")
RESULTS_DIR = os.path.join(DATA_DIR, "preprocessed_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def process_csv(file_path):
    """Process a CSV file of Reddit comments for sentiment analysis"""
    try:
        # Verify file exists
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return
        
        logging.info(f"Processing file: {file_path}")
        
        # Count lines in file for reference
        line_count = sum(1 for _ in open(file_path, 'r', encoding='utf-8', errors='replace'))
        logging.info(f"File contains {line_count} lines (including header)")
        
        # Initialize VADER sentiment analyzer
        sia = SentimentIntensityAnalyzer()
        logging.info("VADER sentiment analyzer initialized")
        
        # Read CSV file (potentially large)
        # We'll read in chunks to manage memory
        dataset_name = os.path.basename(file_path).replace('.csv', '')
        dataset_dir = os.path.join(RESULTS_DIR, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Initialize results structure
        results = {
            "positive": {"count": 0, "percentage": 0, "samples": []},
            "negative": {"count": 0, "percentage": 0, "samples": []},
            "neutral": {"count": 0, "percentage": 0, "samples": []},
            "total": 0,
            "by_year": {},
            "by_month": {},
            "word_freq": {"positive": {}, "negative": {}, "neutral": {}},
            "emotions": {"joy": 0, "anger": 0, "sadness": 0, "fear": 0, "surprise": 0}
        }
        
        # Process file in chunks
        chunk_size = 5000
        chunk_count = 0
        processed_comments = 0
        
        # Use chunking to handle large files
        for chunk in pd.read_csv(
            file_path, 
            chunksize=chunk_size,
            encoding='utf-8',
            on_bad_lines='skip',
            low_memory=False,
            dtype={'comment': str}
        ):
            chunk_count += 1
            chunk_rows = len(chunk)
            logging.info(f"Processing chunk {chunk_count} with {chunk_rows} rows")
            
            # Process each comment in chunk
            for _, row in chunk.iterrows():
                if 'comment' not in row:
                    continue
                
                comment_text = row['comment']
                
                # Skip empty or invalid comments
                if not isinstance(comment_text, str) or not comment_text.strip():
                    continue
                
                # Analyze sentiment
                scores = sia.polarity_scores(comment_text)
                
                # Determine sentiment
                if scores["compound"] >= 0.05:
                    sentiment = "positive"
                elif scores["compound"] <= -0.05:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
                
                # Update counts
                results[sentiment]["count"] += 1
                results["total"] += 1
                
                # Add sample comment (max 5 per category)
                if len(results[sentiment]["samples"]) < 5:
                    sample = {
                        "text": comment_text[:200] + ("..." if len(comment_text) > 200 else ""),
                        "score": row.get('score', 1),
                        "date": datetime.now().strftime('%Y-%m-%d'),
                        "sentiment_score": scores["compound"]
                    }
                    results[sentiment]["samples"].append(sample)
                
                processed_comments += 1
                
                # Log progress periodically
                if processed_comments % 10000 == 0:
                    pos = results["positive"]["count"]
                    neg = results["negative"]["count"]
                    neu = results["neutral"]["count"]
                    logging.info(f"Processed {processed_comments} comments: {pos} positive, {neg} negative, {neu} neutral")
        
        # Calculate percentages
        if results["total"] > 0:
            for sentiment in ["positive", "negative", "neutral"]:
                results[sentiment]["percentage"] = round(
                    results[sentiment]["count"] / results["total"] * 100, 2
                )
        
        # Save results
        results_file = os.path.join(dataset_dir, "full_analysis.json")
        with open(results_file, 'w') as f:
            json.dump(results, f)
        
        logging.info("Analysis complete")
        logging.info(f"Total comments processed: {results['total']}")
        logging.info(f"Positive: {results['positive']['count']} ({results['positive']['percentage']}%)")
        logging.info(f"Negative: {results['negative']['count']} ({results['negative']['percentage']}%)")
        logging.info(f"Neutral: {results['neutral']['count']} ({results['neutral']['percentage']}%)")
        logging.info(f"Results saved to {results_file}")
        
        return results_file
        
    except Exception as e:
        logging.error(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_csv.py <path_to_csv_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    process_csv(file_path)