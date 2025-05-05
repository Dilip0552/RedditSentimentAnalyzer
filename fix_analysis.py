import os
import sys
import json
import logging
import pandas as pd
import re
import string
import numpy as np
from datetime import datetime
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check if data directory exists, create if needed
DATA_DIR = os.path.join(os.getcwd(), "data")
RESULTS_DIR = os.path.join(DATA_DIR, "preprocessed_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Download NLTK resources if needed
try:
    stopwords.words('english')
except:
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')

# Initialize emotion keywords for detection
EMOTION_KEYWORDS = {
    'joy': ['happy', 'joy', 'excited', 'great', 'love', 'awesome', 'excellent', 'wonderful', 'glad', 'fantastic', 'amazing', 'delighted', 'pleased', 'thrilled', 'celebrated', 'cheered'],
    'anger': ['angry', 'mad', 'hate', 'rage', 'furious', 'annoyed', 'irritated', 'outraged', 'disgusting', 'frustrating', 'pissed', 'bitter', 'enraged', 'hostile', 'offended'],
    'sadness': ['sad', 'unhappy', 'depressed', 'disappointed', 'miserable', 'grief', 'heartbroken', 'gloomy', 'sorrow', 'crying', 'despair', 'regret', 'melancholy', 'hopeless'],
    'fear': ['afraid', 'scared', 'fear', 'terrified', 'worried', 'anxious', 'panic', 'nervous', 'frightened', 'dread', 'horror', 'concern', 'alarmed', 'threatened'],
    'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected', 'wow', 'startled', 'stunned', 'incredible', 'unbelievable', 'suddenly', 'unpredictable']
}

# Define stopwords
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """Clean and preprocess text for analysis"""
    if not isinstance(text, str):
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove Reddit-specific patterns
    text = re.sub(r'\[deleted\]|\[removed\]', '', text)
    text = re.sub(r'r/\w+', '', text)
    text = re.sub(r'u/\w+', '', text)
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_word_frequencies(comments, sentiment_type, top_n=50):
    """Extract word frequencies from comments of a specific sentiment"""
    all_words = []
    
    for comment in comments:
        if isinstance(comment, str) and comment.strip():
            # Clean text
            cleaned = clean_text(comment)
            
            # Simple tokenization (split by whitespace)
            tokens = cleaned.split()
            
            # Remove stopwords and very short words
            filtered_tokens = [word for word in tokens if word.lower() not in STOPWORDS and len(word) > 2]
            
            all_words.extend(filtered_tokens)
    
    # Get frequency distribution
    word_freq = Counter(all_words)
    
    # Return top N words
    return dict(word_freq.most_common(top_n))

def detect_emotions(text):
    """Detect emotions in text using rule-based approach"""
    text = text.lower()
    results = {emotion: 0 for emotion in EMOTION_KEYWORDS}
    
    # Count occurrences of emotion keywords
    for emotion, keywords in EMOTION_KEYWORDS.items():
        for keyword in keywords:
            if f" {keyword} " in f" {text} ":  # Add spaces to ensure whole word matching
                results[emotion] += 1
    
    # Convert to percentages based on total emotions found
    total_emotions = sum(results.values())
    if total_emotions > 0:
        for emotion in results:
            results[emotion] = round(results[emotion] / total_emotions, 4)
    
    return results

def extract_topics(comments, num_topics=3, num_words=5):
    """Extract main topics from comments using word frequency approach"""
    if len(comments) < 10:
        return []
    
    # For efficiency, limit the number of comments to process
    max_comments = min(1000, len(comments))
    sample_comments = comments[:max_comments]
    
    # Clean comments
    cleaned_comments = [clean_text(c) for c in sample_comments if isinstance(c, str) and c.strip()]
    
    if not cleaned_comments:
        return []
    
    try:
        # Combine all words from all comments
        all_words = []
        for comment in cleaned_comments:
            # Simple tokenization
            tokens = comment.split()
            # Filter out stopwords and short words
            filtered_tokens = [word for word in tokens if word.lower() not in STOPWORDS and len(word) > 2]
            all_words.extend(filtered_tokens)
        
        # Get most common words
        word_freq = Counter(all_words)
        top_words = word_freq.most_common(num_topics * num_words)
        
        # Create topics (simple approach - just group words)
        topics = []
        for i in range(0, min(num_topics, len(top_words) // num_words)):
            start_idx = i * num_words
            end_idx = start_idx + num_words
            if start_idx < len(top_words):
                words = [word for word, _ in top_words[start_idx:end_idx]]
                topics.append({
                    "id": i + 1,
                    "words": words
                })
        
        return topics
    except Exception as e:
        logging.error(f"Error extracting topics: {e}")
        return []

def fix_dataset_analysis(file_path, max_rows=None):
    """Process a CSV file of Reddit comments for sentiment analysis"""
    try:
        # Verify file exists
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return
        
        # Get dataset name from file path
        dataset_name = os.path.basename(file_path).replace('.csv', '')
        logging.info(f"Processing dataset: {dataset_name}")
        
        # Count lines in file for reference
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            line_count = sum(1 for _ in f)
        logging.info(f"File contains {line_count} lines (including header)")
        
        # Initialize VADER sentiment analyzer
        sia = SentimentIntensityAnalyzer()
        logging.info("VADER sentiment analyzer initialized")
        
        # Initialize results structure
        results = {
            "positive": {"count": 0, "percentage": 0, "samples": []},
            "negative": {"count": 0, "percentage": 0, "samples": []},
            "neutral": {"count": 0, "percentage": 0, "samples": []},
            "total": 0,
            "by_year": [],
            "by_month": [],
            "word_freq": {"positive": {}, "negative": {}, "neutral": {}},
            "subreddits": [],
            "emotions": {
                "joy": 0.0,
                "anger": 0.0,
                "sadness": 0.0,
                "fear": 0.0,
                "surprise": 0.0
            },
            "topics": []
        }
        
        # Read and process file in chunks to handle very large files
        chunk_size = 10000
        chunk_count = 0
        processed_comments = 0
        
        # Use chunking to handle large files efficiently
        for chunk in pd.read_csv(
            file_path, 
            chunksize=chunk_size,
            dtype={'comment': str, 'subreddit': str},
            encoding='utf-8',
            on_bad_lines='skip',
            low_memory=False,
            quotechar='"',
            escapechar='\\'
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
                
                # Get subreddit (if available) for subreddit stats
                subreddit = row.get('subreddit', 'unknown')
                
                # Check if this subreddit is already in our list
                subreddit_found = False
                for sub in results["subreddits"]:
                    if sub.get("name") == subreddit:
                        sub[sentiment] += 1
                        sub["total"] += 1
                        subreddit_found = True
                        break
                
                # If not found, add a new subreddit entry
                if not subreddit_found:
                    results["subreddits"].append({
                        "name": subreddit,
                        "positive": 1 if sentiment == "positive" else 0,
                        "negative": 1 if sentiment == "negative" else 0,
                        "neutral": 1 if sentiment == "neutral" else 0,
                        "total": 1
                    })
                
                # Get date info (for time-based analysis)
                created_year = datetime.now().year  # Default to current year
                created_month = f"{created_year}-01"  # Default month
                
                if 'created_utc' in row:
                    try:
                        created_time = datetime.fromtimestamp(float(row['created_utc']))
                        created_year = created_time.year
                        created_month = f"{created_year}-{created_time.month:02d}"
                    except:
                        pass  # If timestamp conversion fails, use default
                
                # Update yearly data
                year_found = False
                for year_data in results["by_year"]:
                    if year_data.get("year") == created_year:
                        year_data[sentiment] += 1
                        year_found = True
                        break
                
                if not year_found:
                    results["by_year"].append({
                        "year": created_year,
                        "positive": 1 if sentiment == "positive" else 0,
                        "negative": 1 if sentiment == "negative" else 0,
                        "neutral": 1 if sentiment == "neutral" else 0,
                        "total": 1
                    })
                
                # Update monthly data
                month_found = False
                for month_data in results["by_month"]:
                    if month_data.get("month") == created_month:
                        month_data[sentiment] += 1
                        month_data["total"] += 1
                        month_found = True
                        break
                
                if not month_found:
                    results["by_month"].append({
                        "month": created_month,
                        "positive": 1 if sentiment == "positive" else 0,
                        "negative": 1 if sentiment == "negative" else 0,
                        "neutral": 1 if sentiment == "neutral" else 0,
                        "total": 1
                    })
                
                # Add sample comment (max 5 per category)
                if len(results[sentiment]["samples"]) < 5:
                    sample = {
                        "text": comment_text[:200] + ("..." if len(comment_text) > 200 else ""),
                        "score": row.get('score', 1),
                        "date": row.get('date', datetime.now().strftime('%Y-%m-%d')),
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
                
                # Stop if max_rows limit is reached
                if max_rows and processed_comments >= max_rows:
                    logging.info(f"Reached max rows limit ({max_rows}), stopping processing")
                    break
            
            # Stop chunking if max_rows limit is reached
            if max_rows and processed_comments >= max_rows:
                break
        
        # Calculate percentages
        if results["total"] > 0:
            for sentiment in ["positive", "negative", "neutral"]:
                results[sentiment]["percentage"] = round(
                    results[sentiment]["count"] / results["total"] * 100, 2
                )
                
        # Calculate percentages for each subreddit
        for subreddit in results["subreddits"]:
            if subreddit["total"] > 0:
                subreddit["positive_percentage"] = round(subreddit["positive"] / subreddit["total"] * 100, 2)
                subreddit["negative_percentage"] = round(subreddit["negative"] / subreddit["total"] * 100, 2)
                subreddit["neutral_percentage"] = round(subreddit["neutral"] / subreddit["total"] * 100, 2)
        
        # Sort by total comments
        results["subreddits"].sort(key=lambda x: x["total"], reverse=True)
        
        # Sort time data chronologically
        results["by_month"].sort(key=lambda x: x["month"])
        results["by_year"].sort(key=lambda x: x["year"])
        
        # Collect comment text by sentiment for further analysis
        logging.info("Collecting comment texts for word frequency and topic analysis")
        sentiment_texts = {"positive": [], "negative": [], "neutral": []}
        all_comments = []
        
        # Limit the number of comments to process for efficiency
        max_comments_to_process = min(2000, processed_comments)
        comments_processed = 0
        
        # Read the file again to collect sample comments
        for chunk in pd.read_csv(
            file_path, 
            chunksize=chunk_size,
            dtype={'comment': str},
            encoding='utf-8',
            on_bad_lines='skip',
            low_memory=False
        ):
            for _, row in chunk.iterrows():
                if comments_processed >= max_comments_to_process:
                    break
                    
                if 'comment' not in row:
                    continue
                
                comment_text = row.get('comment', '')
                if not isinstance(comment_text, str) or not comment_text.strip():
                    continue
                
                # Analyze sentiment
                scores = sia.polarity_scores(comment_text)
                
                # Determine sentiment category
                if scores["compound"] >= 0.05:
                    # Only collect a reasonable number of comments per sentiment
                    if len(sentiment_texts["positive"]) < 700:
                        sentiment_texts["positive"].append(comment_text)
                elif scores["compound"] <= -0.05:
                    if len(sentiment_texts["negative"]) < 700:
                        sentiment_texts["negative"].append(comment_text)
                else:
                    if len(sentiment_texts["neutral"]) < 700:
                        sentiment_texts["neutral"].append(comment_text)
                
                if len(all_comments) < 1000:
                    all_comments.append(comment_text)
                
                # Detect emotions for overall emotion analysis
                # Only process a subset of comments for emotion analysis
                if comments_processed % 5 == 0:  # Process every 5th comment
                    emotion_scores = detect_emotions(comment_text)
                    for emotion, score in emotion_scores.items():
                        results["emotions"][emotion] += score
                
                comments_processed += 1
                
                # Status update
                if comments_processed % 1000 == 0:
                    logging.info(f"Processed {comments_processed} comments for word frequency and emotion analysis")
                    
            if comments_processed >= max_comments_to_process:
                break
                
        # Normalize emotion scores
        emotions_sum = sum(results["emotions"].values())
        if emotions_sum > 0:
            for emotion in results["emotions"]:
                results["emotions"][emotion] = round(results["emotions"][emotion] / emotions_sum, 4)
        
        # Extract word frequencies for each sentiment
        logging.info("Extracting word frequencies...")
        for sentiment in ["positive", "negative", "neutral"]:
            if sentiment_texts[sentiment]:
                results["word_freq"][sentiment] = get_word_frequencies(
                    sentiment_texts[sentiment], 
                    sentiment, 
                    top_n=50
                )
        
        # Extract topics
        logging.info("Extracting topics...")
        topics = extract_topics(all_comments, num_topics=3, num_words=5)
        results["topics"] = topics
        
        # Save results to JSON file
        dataset_dir = os.path.join(RESULTS_DIR, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        results_file = os.path.join(dataset_dir, "full_analysis.json")
        with open(results_file, 'w') as f:
            json.dump(results, f)
        
        # Save summary to a separate file for quick access
        summary_file = os.path.join(dataset_dir, "summary.json")
        summary = {
            "dataset": dataset_name,
            "total_comments": results["total"],
            "positive": results["positive"]["count"],
            "negative": results["negative"]["count"],
            "neutral": results["neutral"]["count"],
            "positive_pct": results["positive"]["percentage"],
            "negative_pct": results["negative"]["percentage"],
            "neutral_pct": results["neutral"]["percentage"],
            "processed_at": datetime.now().isoformat()
        }
        with open(summary_file, 'w') as f:
            json.dump(summary, f)
        
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
        print("Usage: python fix_analysis.py <path_to_csv_file> [max_rows]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    max_rows = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    fix_dataset_analysis(file_path, max_rows)