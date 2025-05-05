#!/usr/bin/env python3
import os
import pandas as pd
import json
import nltk
import logging
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from datetime import datetime
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download NLTK resources if needed
nltk_data_dir = os.path.expanduser('~/nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.download('vader_lexicon', download_dir=nltk_data_dir, quiet=True)
nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Define paths
DATA_DIR = Path('data')
COMMENTS_DIR = DATA_DIR / 'reddit_comments'
RESULTS_DIR = DATA_DIR / 'preprocessed_results'

# Create directories if they don't exist
os.makedirs(COMMENTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Create a JSON file with metadata about available datasets
DATASETS_METADATA_FILE = RESULTS_DIR / 'datasets_metadata.json'

def clean_text(text):
    """Clean and preprocess text"""
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove Reddit username mentions
    text = re.sub(r'u/\w+', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize using simple splitting to avoid NLTK dependency issues
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
    
    # Calculate emotion intensities
    emotions = {}
    for emotion, keywords in emotion_lexicons.items():
        count = sum(1 for word in words if word in keywords)
        if count > 0:
            # Normalize by text length to get intensity
            intensity = min(1.0, count / (len(words) * 0.3))  # Cap at 1.0
            emotions[emotion] = round(intensity, 2)
    
    return emotions

def extract_topics(comments, num_topics=3, num_words=5):
    """Extract main topics from comments using TF-IDF"""
    if not comments or len(comments) < 5:
        return []
    
    try:
        # Initialize vectorizer
        vectorizer = TfidfVectorizer(
            max_df=0.7,      # Ignore terms that appear in more than 70% of documents
            min_df=2,        # Ignore terms that appear in fewer than 2 documents
            stop_words='english',
            ngram_range=(1, 2)  # Consider unigrams and bigrams
        )
        
        # Fit and transform comments
        tfidf_matrix = vectorizer.fit_transform(comments)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Analyze term importance
        importance = np.argsort(np.asarray(tfidf_matrix.sum(axis=0)).ravel())[::-1]
        
        # Extract top topics
        topics = []
        current_topic = []
        topic_count = 0
        
        for idx in importance:
            term = feature_names[idx]
            if len(term.split()) > 1 or len(term) > 3:  # Only consider meaningful terms
                current_topic.append(term)
                
                if len(current_topic) >= num_words:
                    topics.append(current_topic)
                    current_topic = []
                    topic_count += 1
                    
                    if topic_count >= num_topics:
                        break
        
        return topics
    except Exception as e:
        logging.error(f"Error extracting topics: {e}")
        return []

def analyze_dataset(file_path, file_name):
    """Analyze a dataset and save results to JSON files"""
    try:
        logging.info(f"Processing dataset: {file_name}")
        
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Check required columns
        if 'comment' not in df.columns:
            logging.error(f"Missing required column 'comment' in {file_name}")
            return False
        
        # Extract subreddit info if available
        subreddits = []
        if 'subreddit' in df.columns:
            subreddits = df['subreddit'].unique().tolist()
        
        # Extract topic from filename (assume filename format)
        topic_match = re.search(r'comments_(.+?)_all', file_name)
        topic = topic_match.group(1).replace('_', ' ') if topic_match else "Unknown"
        
        # Create display name
        display_name = f"{topic} ({len(subreddits)} subreddits)"
        
        # Create dataset metadata
        dataset_metadata = {
            "id": file_name,
            "file_name": file_name,
            "display_name": display_name,
            "topic": topic,
            "subreddit": 'all',
            "processed_at": datetime.now().isoformat(),
            "available_subreddits": subreddits if subreddits else ["all"]
        }
        
        # Create the base directory for this dataset's results
        dataset_dir = RESULTS_DIR / file_name.replace('.csv', '')
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Process without filters first (all data)
        analyze_and_save(df, dataset_dir, "", "all", False)
        
        # Process with emotion detection
        analyze_and_save(df, dataset_dir, "", "all", True)
        
        # Process for each subreddit if available
        if 'subreddit' in df.columns and len(subreddits) > 1:
            for subreddit in subreddits:
                subreddit_df = df[df['subreddit'] == subreddit]
                if len(subreddit_df) < 5:  # Skip small subreddits
                    continue
                    
                analyze_and_save(subreddit_df, dataset_dir, "", subreddit, False)
                analyze_and_save(subreddit_df, dataset_dir, "", subreddit, True)
        
        return dataset_metadata
        
    except Exception as e:
        logging.error(f"Error preprocessing dataset {file_name}: {e}")
        return None

def analyze_and_save(df, dataset_dir, topic_filter, subreddit_filter, include_emotion):
    """Analyze DataFrame with filters and save results to JSON file"""
    logging.info(f"Analyzing with filters - Topic: '{topic_filter}', Subreddit: '{subreddit_filter}', Emotion: {include_emotion}")
    
    # Apply topic filter if provided
    filtered_df = df.copy()
    logging.info(f"Initial DataFrame size: {len(filtered_df)} rows")
    
    if topic_filter:
        filtered_df = filtered_df[filtered_df['comment'].str.lower().str.contains(topic_filter.lower(), na=False)]
    
    # Apply subreddit filter if not 'all'
    if subreddit_filter != 'all' and 'subreddit' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['subreddit'] == subreddit_filter]
    
    logging.info(f"DataFrame size after filtering: {len(filtered_df)} rows")
    
    # Skip if no comments after filtering
    if filtered_df.empty:
        logging.warning(f"No comments left after filtering: topic='{topic_filter}', subreddit='{subreddit_filter}'")
        return
    
    # Process comments for sentiment and emotion
    results = {
        "positive": {"count": 0, "percentage": 0, "samples": []},
        "negative": {"count": 0, "percentage": 0, "samples": []},
        "neutral": {"count": 0, "percentage": 0, "samples": []},
        "total": 0,
        "by_year": {},
        "by_month": {}
    }
    
    word_freq = {"positive": {}, "negative": {}, "neutral": {}}
    all_cleaned_comments = []
    
    for _, row in filtered_df.iterrows():
        comment_text = row['comment']
        
        # Skip empty or deleted comments
        if not isinstance(comment_text, str) or comment_text == "[deleted]" or comment_text == "[removed]":
            continue
        
        # Clean and analyze comment
        cleaned_comment = clean_text(comment_text)
        if not cleaned_comment:
            continue
            
        all_cleaned_comments.append(cleaned_comment)
        
        sentiment_result = analyze_sentiment(cleaned_comment)
        sentiment = sentiment_result["sentiment"]
        
        # Add date info (use current date if not available)
        created_time = datetime.now()
        if 'created_utc' in row:
            created_time = datetime.fromtimestamp(row['created_utc'])
        elif 'date' in row:
            try:
                created_time = datetime.fromisoformat(row['date'])
            except (ValueError, TypeError):
                pass
        
        created_year = created_time.year
        created_month = f"{created_year}-{created_time.month:02d}"
        
        # Update counts
        results[sentiment]["count"] += 1
        results["total"] += 1
        
        # Add sample comment (keeping max 5 samples per category)
        if len(results[sentiment]["samples"]) < 5:
            sample = {
                "text": comment_text[:200] + ("..." if len(comment_text) > 200 else ""),
                "score": row.get('score', 1),
                "date": created_time.strftime('%Y-%m-%d'),
                "sentiment_score": sentiment_result["compound"]
            }
            
            # Add emotion analysis if requested
            if include_emotion:
                sample["emotions"] = detect_emotions(comment_text)
                
            results[sentiment]["samples"].append(sample)
        
        # Update year data
        if created_year not in results["by_year"]:
            results["by_year"][created_year] = {
                "positive": 0, "negative": 0, "neutral": 0
            }
        results["by_year"][created_year][sentiment] += 1
        
        # Update month data
        if created_month not in results["by_month"]:
            results["by_month"][created_month] = {
                "positive": 0, "negative": 0, "neutral": 0
            }
        results["by_month"][created_month][sentiment] += 1
        
        # Add words to sentiment frequency data
        words = [word for word in cleaned_comment.split() if len(word) > 3]
        for word in words:
            if word not in word_freq[sentiment]:
                word_freq[sentiment][word] = 0
            word_freq[sentiment][word] += 1
    
    # Calculate percentages
    if results["total"] > 0:
        for sentiment in ["positive", "negative", "neutral"]:
            results[sentiment]["percentage"] = round(results[sentiment]["count"] / results["total"] * 100, 2)
    
    # Convert by_year dict to a list format for easier charting
    year_data = []
    for year, sentiments in results["by_year"].items():
        year_data.append({
            "year": year,
            "positive": sentiments["positive"],
            "negative": sentiments["negative"],
            "neutral": sentiments["neutral"]
        })
    year_data.sort(key=lambda x: x["year"])
    results["by_year"] = year_data
    
    # Convert by_month dict to a list format
    month_data = []
    for month, sentiments in results["by_month"].items():
        month_data.append({
            "month": month,
            "positive": sentiments["positive"],
            "negative": sentiments["negative"],
            "neutral": sentiments["neutral"],
            "total": sentiments["positive"] + sentiments["negative"] + sentiments["neutral"]
        })
    month_data.sort(key=lambda x: x["month"])
    results["by_month"] = month_data
    
    # Add word frequency data
    if any(word_freq.values()):
        # Remove words occurring only once
        for sentiment_type in ["positive", "negative", "neutral"]:
            word_freq[sentiment_type] = {
                word: count for word, count in word_freq[sentiment_type].items() 
                if count > 1  # Only keep words that appear more than once
            }
        results["word_freq"] = word_freq
    
    # Add topic modeling
    if len(all_cleaned_comments) > 10:
        topics = extract_topics(all_cleaned_comments)
        if topics:
            results["topics"] = topics
    
    # Add emotion analysis for overall dataset if requested
    if include_emotion:
        all_text = " ".join(all_cleaned_comments)
        results["emotions"] = detect_emotions(all_text)
    
    # Log the final number of comments that were successfully processed
    logging.info(f"Successfully processed {results['total']} comments out of {len(filtered_df)} after cleaning")
    
    # Generate a filename based on parameters
    filename_parts = []
    if topic_filter:
        filename_parts.append(f"topic_{topic_filter}")
    filename_parts.append(f"subreddit_{subreddit_filter}")
    if include_emotion:
        filename_parts.append("with_emotions")
    
    result_filename = "_".join(filename_parts) + ".json"
    result_path = dataset_dir / result_filename
    
    # Save results to JSON file
    with open(result_path, 'w') as f:
        json.dump(results, f)
    
    logging.info(f"Saved analysis results to {result_path}")

def main():
    """Main function to process all datasets"""
    logging.info("Starting dataset preprocessing")
    
    # Look for CSV files in data directory and attached_assets directory
    csv_files = list(COMMENTS_DIR.glob('*.csv'))
    
    # Also check attached_assets directory if it exists
    attached_assets_dir = Path('attached_assets')
    if attached_assets_dir.exists():
        csv_files.extend(attached_assets_dir.glob('*.csv'))
    
    # If no CSV files found, use sample data in root directory
    if not csv_files:
        csv_files = list(Path('.').glob('*.csv'))
        
    if not csv_files:
        logging.warning("No CSV files found")
        return
    
    # Process each CSV file and collect metadata
    all_datasets = []
    for file_path in csv_files:
        file_name = file_path.name
        metadata = analyze_dataset(file_path, file_name)
        if metadata:
            all_datasets.append(metadata)
    
    # Save datasets metadata
    with open(DATASETS_METADATA_FILE, 'w') as f:
        json.dump(all_datasets, f)
    
    logging.info(f"Preprocessing completed. {len(all_datasets)}/{len(csv_files)} datasets processed successfully.")
    logging.info(f"Metadata saved to {DATASETS_METADATA_FILE}")

if __name__ == "__main__":
    main()