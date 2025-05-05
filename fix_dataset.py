import os
import sys
import json
import logging
import pandas as pd
import re
from datetime import datetime
from pathlib import Path
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download required NLTK resources
try:
    nltk.download('vader_lexicon')
    nltk.download('stopwords')
    nltk.download('punkt')
    logging.info("NLTK resources downloaded successfully")
except Exception as e:
    logging.error(f"Error downloading NLTK resources: {e}")

# Initialize VADER sentiment analyzer
try:
    sia = SentimentIntensityAnalyzer()
    logging.info("VADER sentiment analyzer initialized successfully")
except Exception as e:
    logging.error(f"Error initializing VADER: {e}")
    sys.exit(1)

# Directory structure
TEMP_DIR = os.path.join(os.getcwd(), "temp")
DATA_DIR = os.path.join(os.getcwd(), "data")
RESULTS_DIR = os.path.join(DATA_DIR, "preprocessed_results")

# Create necessary directories
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def clean_text(text):
    """Clean and preprocess text"""
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove Reddit usernames
    text = re.sub(r'u/\w+', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = text.split()
    filtered_text = [word for word in word_tokens if word.isalnum() and len(word) > 2 and word not in stop_words]
    
    return ' '.join(filtered_text)

def analyze_sentiment(text):
    """Analyze sentiment using VADER"""
    scores = sia.polarity_scores(text)
    
    # Determine sentiment category
    if scores["compound"] >= 0.05:
        sentiment = "positive"
    elif scores["compound"] <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    return {
        "compound": scores['compound'],
        "sentiment": sentiment
    }

def detect_emotions(text):
    """Detect emotions in text using rule-based approach"""
    if not text:
        return {}
    
    # Define emotion lexicons
    emotion_lexicons = {
        "joy": ["happy", "happiness", "joy", "delighted", "pleased", "glad", "excited", "thrilled", 
                "enjoy", "love", "great", "wonderful", "amazing", "fantastic", "awesome"],
        "anger": ["angry", "anger", "mad", "furious", "outraged", "annoyed", "irritated", "frustrated",
                 "hate", "despise", "resent", "rage", "hostile", "bitter"],
        "sadness": ["sad", "sadness", "unhappy", "depressed", "miserable", "gloomy", "disappointed",
                   "upset", "heartbroken", "grief", "sorrow", "regret", "melancholy"],
        "fear": ["afraid", "fear", "scared", "terrified", "anxious", "worried", "nervous", "dread",
                "panic", "frightened", "horrified", "alarmed", "concerned"],
        "surprise": ["surprised", "surprise", "shocked", "astonished", "amazed", "stunned", "unexpected",
                    "sudden", "wow", "whoa", "unexpected", "startled"]
    }
    
    # Convert text to lowercase and tokenize
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    # Count emotions
    emotion_counts = {"joy": 0, "anger": 0, "sadness": 0, "fear": 0, "surprise": 0}
    for word in words:
        for emotion, keywords in emotion_lexicons.items():
            if word in keywords:
                emotion_counts[emotion] += 1
    
    # Return emotions with non-zero counts
    return {emotion: count for emotion, count in emotion_counts.items() if count > 0}

def process_csv_file(file_path, output_dir, batch_size=5000):
    """Process CSV file in batches to handle large files"""
    logging.info(f"Starting processing of {file_path}")
    
    # Count lines in file
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        total_lines = sum(1 for _ in f)
    logging.info(f"File contains {total_lines} lines (including header)")
    
    # Create output directory
    dataset_name = os.path.basename(file_path).replace('.csv', '')
    result_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(result_dir, exist_ok=True)
    
    # Initialize results
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
    
    # Process file in chunks to avoid memory issues
    chunk_iterator = pd.read_csv(
        file_path,
        chunksize=batch_size,
        dtype={'comment': str, 'subreddit': str},
        quotechar='"',
        escapechar='\\',
        encoding='utf-8', 
        on_bad_lines='skip',
        low_memory=False
    )
    
    processed_rows = 0
    total_chunks = (total_lines - 1) // batch_size + 1
    
    for i, chunk in enumerate(chunk_iterator):
        logging.info(f"Processing chunk {i+1}/{total_chunks} ({len(chunk)} rows)")
        
        # Process each comment in the chunk
        for _, row in chunk.iterrows():
            if 'comment' not in row:
                continue
                
            comment_text = row['comment']
            
            # Skip empty or invalid comments
            if not isinstance(comment_text, str) or comment_text.strip() == "":
                continue
            
            # Clean and analyze comment
            cleaned_comment = clean_text(comment_text)
            if not cleaned_comment:
                continue
            
            sentiment_result = analyze_sentiment(cleaned_comment)
            sentiment = sentiment_result["sentiment"]
            
            # Update counts
            results[sentiment]["count"] += 1
            results["total"] += 1
            
            # Add sample comment (keeping max 5 samples per category)
            if len(results[sentiment]["samples"]) < 5:
                sample = {
                    "text": comment_text[:200] + ("..." if len(comment_text) > 200 else ""),
                    "score": row.get('score', 1),
                    "date": datetime.now().strftime('%Y-%m-%d'),
                    "sentiment_score": sentiment_result["compound"]
                }
                
                # Add emotion analysis
                emotions = detect_emotions(comment_text)
                sample["emotions"] = emotions
                
                results[sentiment]["samples"].append(sample)
            
            # Update emotion counts
            emotions = detect_emotions(cleaned_comment)
            for emotion, count in emotions.items():
                results["emotions"][emotion] += count
            
            # Update word frequency
            words = [word for word in cleaned_comment.split() if len(word) > 3]
            for word in words:
                if word not in results["word_freq"][sentiment]:
                    results["word_freq"][sentiment][word] = 0
                results["word_freq"][sentiment][word] += 1
            
            processed_rows += 1
            
            # Log progress periodically
            if processed_rows % 10000 == 0:
                logging.info(f"Processed {processed_rows} comments: {results['positive']['count']} positive, {results['negative']['count']} negative, {results['neutral']['count']} neutral")
    
    # Calculate percentages
    if results["total"] > 0:
        for sentiment in ["positive", "negative", "neutral"]:
            results[sentiment]["percentage"] = round(results[sentiment]["count"] / results["total"] * 100, 2)
    
    # Save results
    output_file = os.path.join(result_dir, "full_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(results, f)
    
    logging.info(f"Processing completed. Results saved to {output_file}")
    logging.info(f"Summary: {results['total']} total comments processed")
    logging.info(f"Positive: {results['positive']['count']} ({results['positive']['percentage']}%)")
    logging.info(f"Negative: {results['negative']['count']} ({results['negative']['percentage']}%)")
    logging.info(f"Neutral: {results['neutral']['count']} ({results['neutral']['percentage']}%)")
    
    return output_file

if __name__ == "__main__":
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python fix_dataset.py <csv_file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        sys.exit(1)
    
    # Process file
    output_file = process_csv_file(file_path, RESULTS_DIR)
    print(f"Analysis complete. Results saved to {output_file}")