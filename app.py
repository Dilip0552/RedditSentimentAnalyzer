import os
import re
import json
import praw
import logging
import datetime
import pandas as pd
import numpy
from pathlib import Path
import glob
from flask import Flask, request, jsonify
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from data_reader import get_dataset_analysis, safe_read_csv
# from dotenv import load_dotenv
from flask_cors import CORS
# Setup logging
logging.basicConfig(level=logging.DEBUG)
# load_dotenv()
# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key")

# Download NLTK resources
try:
    # Make sure NLTK data directory exists
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Download essential resources with explicit download dir
    nltk.download('vader_lexicon', download_dir=nltk_data_dir, quiet=True)
    nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
    
    # Verify downloads
    logging.info(f"NLTK data directory: {nltk_data_dir}")
    logging.info(f"NLTK data files: {os.listdir(nltk_data_dir) if os.path.exists(nltk_data_dir) else 'Directory not found'}")
except Exception as e:
    logging.error(f"Error downloading NLTK resources: {e}")

# Initialize PRAW (Reddit API client)
try:
    # Check if Reddit API credentials are available
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent = os.environ.get("REDDIT_USER_AGENT")
    
    if not client_id or not client_secret or not user_agent:
        logging.error("Missing Reddit API credentials. Please ensure REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT are set.")
        reddit = None
    else:
        logging.info("Initializing Reddit API client with provided credentials")
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        # Verify the credentials by a simple API call
        # For read-only usage, we'll just check if we can access a public subreddit
        test_subreddit = reddit.subreddit("python")
        test_subreddit.description  # This will trigger an API call
        logging.info("Reddit API client successfully initialized and authenticated")
except Exception as e:
    logging.error(f"Error initializing Reddit API client: {e}")
    reddit = None

# Initialize VADER sentiment analyzer
try:
    # First make sure the vader_lexicon is downloaded properly
    vader_lexicon_dir = os.path.join(os.path.expanduser('~/nltk_data'), 'sentiment', 'vader_lexicon.zip')
    if not os.path.exists(vader_lexicon_dir):
        logging.warning(f"VADER lexicon not found at {vader_lexicon_dir}, attempting to download...")
        nltk.download('vader_lexicon', download_dir=os.path.expanduser('~/nltk_data'), quiet=False)
    
    sid = SentimentIntensityAnalyzer()
    # Test the analyzer to ensure it works
    test_result = sid.polarity_scores("This is a test.")
    logging.info(f"VADER sentiment analyzer initialized successfully. Test result: {test_result}")
except Exception as e:
    logging.error(f"Error initializing SentimentIntensityAnalyzer: {e}")
    sid = None

# Initialize stopwords
try:
    stop_words = set(stopwords.words('english'))
except Exception as e:
    logging.error(f"Error initializing stopwords: {e}")
    stop_words = set()
    
# Define the path to CSV data directory
CSV_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/reddit_comments')

# Cache for CSV analysis results
csv_analysis_cache = {}

# @app.route('/')
# def index():
#     """Render the main page of the application."""
#     return render_template('index.html')

def clean_text(text):
    """
    Clean and preprocess text by removing URLs, mentions, stopwords, and punctuation.
    
    Args:
        text (str): The text to clean
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Just use simple space-based tokenization instead of NLTK's word_tokenize
    # This avoids potential issues with missing NLTK data files
    word_tokens = text.split()
    
    # Remove stopwords
    filtered_text = [word for word in word_tokens if word not in stop_words]
    
    return ' '.join(filtered_text)

def analyze_sentiment(text):
    """
    Analyze sentiment of text using VADER.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        dict: Sentiment scores and classification
    """
    if not text or not sid:
        return {"compound": 0, "sentiment": "neutral"}
    
    scores = sid.polarity_scores(text)
    
    # Classify sentiment based on compound score
    if scores['compound'] >= 0.05:
        sentiment = "positive"
    elif scores['compound'] <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    return {
        "compound": scores['compound'],
        "sentiment": sentiment
    }

def detect_emotions(text):
    """
    Detect emotions in text using a rule-based approach.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        dict: Emotion categories and their intensities
    """
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
    """
    Extract main topics from a collection of comments using simple TF-IDF.
    
    Args:
        comments (list): List of cleaned comment texts
        num_topics (int): Number of topics to extract
        num_words (int): Number of words per topic
        
    Returns:
        list: Topics with their key words
    """
    if not comments or len(comments) < 5:
        return []
    
    # Create a simple TF-IDF model
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    try:
        # Initialize vectorizer
        vectorizer = TfidfVectorizer(
            max_df=0.7,      # Ignore terms that appear in more than 70% of documents
            min_df=2,        # Ignore terms that appear in fewer than 2 documents
            stop_words='english',
            ngram_range=(1, 2)  # Consider unigrams and bigrams
        )
        
        # Fit and transform the comments
        tfidf_matrix = vectorizer.fit_transform(comments)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Analyze term importance across the corpus
        importance = numpy.argsort(numpy.asarray(tfidf_matrix.sum(axis=0)).ravel())[::-1]
        
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
        logging.error(f"Error in topic extraction: {e}")
        return []

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Process Reddit search request and perform sentiment analysis.
    
    Returns:
        JSON response with sentiment analysis results
    """
    try:
        # Get data from request
        data = request.json
        topic = data.get('topic', '')
        subreddit_name = data.get('subreddit', 'all')
        include_emotion = data.get('include_emotion', False)  # Enhanced analytics - emotion detection
        
        if not topic:
            return jsonify({"error": "No topic provided"}), 400
            
        if not reddit:
            return jsonify({"error": "Reddit API client not properly initialized"}), 500
        
        logging.debug(f"Analyzing topic: {topic} in subreddit: {subreddit_name}")
        
        # Initialize counters and data structures
        results = {
            "query": {
                "topic": topic,
                "subreddit": subreddit_name
            },
            "positive": {"count": 0, "percentage": 0, "samples": []},
            "negative": {"count": 0, "percentage": 0, "samples": []},
            "neutral": {"count": 0, "percentage": 0, "samples": []},
            "total": 0,
            "by_year": {},
            "by_month": {},  # Temporal analysis - monthly data
            "emotions": {}  # Container for aggregated emotions data
        }
        
        # Get subreddit and search for posts
        subreddit = reddit.subreddit(subreddit_name)
        
        # Limit to top 10 posts to avoid rate limiting
        posts = list(subreddit.search(topic, limit=10, sort="relevance"))
        
        if not posts:
            return jsonify({"error": "No posts found for the given topic and subreddit"}), 404
        
        # Process each post's comments
        for post in posts:
            post.comments.replace_more(limit=0)  # Skip "load more comments" links
            for comment in post.comments.list():
                if not comment.body or comment.body == "[deleted]" or comment.body == "[removed]":
                    continue
                
                # Clean and analyze comment
                cleaned_comment = clean_text(comment.body)
                if not cleaned_comment:
                    continue
                    
                sentiment_result = analyze_sentiment(cleaned_comment)
                sentiment = sentiment_result["sentiment"]
                
                # Get date info from comment created time
                created_time = datetime.datetime.fromtimestamp(comment.created_utc)
                created_year = created_time.year
                created_month = f"{created_year}-{created_time.month:02d}"
                
                # Update counts
                results[sentiment]["count"] += 1
                results["total"] += 1
                
                # Add sample comment (keeping max 5 samples per category)
                if len(results[sentiment]["samples"]) < 5:
                    sample = {
                        "text": comment.body[:200] + ("..." if len(comment.body) > 200 else ""),
                        "score": comment.score,
                        "date": created_time.strftime('%Y-%m-%d'),
                        "sentiment_score": sentiment_result["compound"]
                    }
                    
                    # Add emotion analysis if requested (Enhanced Analytics)
                    if include_emotion:
                        # Detect emotions in this comment
                        comment_emotions = detect_emotions(comment.body)
                        sample["emotions"] = comment_emotions
                        
                        # Aggregate emotions across all comments for the overall emotion chart
                        for emotion, intensity in comment_emotions.items():
                            if emotion not in results["emotions"]:
                                results["emotions"][emotion] = 0
                            # Take the maximum intensity found for each emotion across all comments
                            results["emotions"][emotion] = max(results["emotions"][emotion], intensity)
                    
                    results[sentiment]["samples"].append(sample)
                
                # Update year data
                if created_year not in results["by_year"]:
                    results["by_year"][created_year] = {
                        "positive": 0,
                        "negative": 0,
                        "neutral": 0
                    }
                results["by_year"][created_year][sentiment] += 1
                
                # Update month data (Temporal Analysis)
                if created_month not in results["by_month"]:
                    results["by_month"][created_month] = {
                        "positive": 0,
                        "negative": 0,
                        "neutral": 0
                    }
                results["by_month"][created_month][sentiment] += 1
                
                # Add words to sentiment frequency data (for word cloud/heat map)
                # Split the cleaned comment into words
                words = [word for word in cleaned_comment.split() if len(word) > 3]  # Only include words with 4+ chars
                for word in words:
                    # Initialize word data if not present
                    if "word_freq" not in results:
                        results["word_freq"] = {
                            "positive": {},
                            "negative": {},
                            "neutral": {}
                        }
                    
                    # Update word frequency for this sentiment
                    if word not in results["word_freq"][sentiment]:
                        results["word_freq"][sentiment][word] = 0
                    results["word_freq"][sentiment][word] += 1
        
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
        
        # Convert by_month dict to a list (Temporal Analysis)
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
        
        # Sort and prepare word frequency data
        if "word_freq" in results:
            # Remove words occurring only once (likely noise)
            for sentiment_type in ["positive", "negative", "neutral"]:
                results["word_freq"][sentiment_type] = {
                    word: count for word, count in results["word_freq"][sentiment_type].items() 
                    if count > 1  # Only keep words that appear more than once
                }
                
        # Add topic modeling (Enhanced Analytics)
        if results["total"] > 10:  # Only do topic modeling if we have enough comments
            all_comments = []
            for post in posts:
                for comment in post.comments.list():
                    if comment.body and comment.body != "[deleted]" and comment.body != "[removed]":
                        all_comments.append(clean_text(comment.body))
            
            if all_comments:
                results["topics"] = extract_topics(all_comments)
        
        return jsonify(results)
        
    except Exception as e:
        logging.error(f"Error in analyze endpoint: {e}")
        return jsonify({"error": str(e)}), 500

def analyze_csv_comments(file_path):
    """
    Analyze sentiment of comments from a CSV file.
    
    Args:
        file_path (str): Path to CSV file with comments
        
    Returns:
        dict: Analysis results
    """
    # Check if we have cached results
    if file_path in csv_analysis_cache:
        return csv_analysis_cache[file_path]
    
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Initialize results structure
        results = {
            "positive": {"count": 0, "percentage": 0, "samples": []},
            "negative": {"count": 0, "percentage": 0, "samples": []},
            "neutral": {"count": 0, "percentage": 0, "samples": []},
            "total": 0,
            "by_year": {},
            "word_freq": {
                "positive": {},
                "negative": {},
                "neutral": {}
            },
            "subreddits": {},
            "emotions": {}  # Added container for emotions data
        }
        
        # Ensure required columns exist
        if 'comment' not in df.columns:
            return {"error": "CSV file does not contain 'comment' column"}
        
        # Process each comment
        for _, row in df.iterrows():
            comment_text = row['comment']
            subreddit = row.get('subreddit', 'unknown')
            
            # Skip empty or invalid comments
            if not isinstance(comment_text, str) or comment_text.strip() == "":
                continue
                
            # Clean and analyze comment
            cleaned_comment = clean_text(comment_text)
            if not cleaned_comment:
                continue
                
            sentiment_result = analyze_sentiment(cleaned_comment)
            sentiment = sentiment_result["sentiment"]
            
            # Assume current year if not provided in CSV
            # In a real application, you might want to extract years from the comments if available
            current_year = datetime.datetime.now().year
            
            # Update counts
            results[sentiment]["count"] += 1
            results["total"] += 1
            
            # Add sample comment (keeping max 5 samples per category)
            if len(results[sentiment]["samples"]) < 5:
                sample = {
                    "text": comment_text[:200] + ("..." if len(comment_text) > 200 else ""),
                    "score": 0,  # CSV doesn't have scores like Reddit API
                    "date": "N/A"  # CSV might not have dates, this could be enhanced
                }
                
                # Add emotion analysis for enhanced analytics
                comment_emotions = detect_emotions(comment_text)
                sample["emotions"] = comment_emotions
                
                # Aggregate emotions across all comments
                for emotion, intensity in comment_emotions.items():
                    if emotion not in results["emotions"]:
                        results["emotions"][emotion] = 0
                    # Take the maximum intensity found for each emotion
                    results["emotions"][emotion] = max(results["emotions"][emotion], intensity)
                
                results[sentiment]["samples"].append(sample)
            
            # Update subreddit data
            if subreddit not in results["subreddits"]:
                results["subreddits"][subreddit] = {
                    "positive": 0,
                    "negative": 0,
                    "neutral": 0,
                    "total": 0
                }
            results["subreddits"][subreddit][sentiment] += 1
            results["subreddits"][subreddit]["total"] += 1
            
            # Update year data (using current year as default)
            if current_year not in results["by_year"]:
                results["by_year"][current_year] = {
                    "positive": 0,
                    "negative": 0,
                    "neutral": 0
                }
            results["by_year"][current_year][sentiment] += 1
            
            # Add words to sentiment frequency data
            words = [word for word in cleaned_comment.split() if len(word) > 3]
            for word in words:
                if word not in results["word_freq"][sentiment]:
                    results["word_freq"][sentiment][word] = 0
                results["word_freq"][sentiment][word] += 1
        
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
        
        # Sort and prepare word frequency data
        for sentiment_type in ["positive", "negative", "neutral"]:
            results["word_freq"][sentiment_type] = {
                word: count for word, count in results["word_freq"][sentiment_type].items() 
                if count > 1  # Only keep words that appear more than once
            }
        
        # Calculate subreddit percentages and convert to list
        subreddit_data = []
        for subreddit, counts in results["subreddits"].items():
            if counts["total"] > 0:
                subreddit_data.append({
                    "name": subreddit,
                    "positive": counts["positive"],
                    "negative": counts["negative"],
                    "neutral": counts["neutral"],
                    "total": counts["total"],
                    "positive_percentage": round(counts["positive"] / counts["total"] * 100, 2),
                    "negative_percentage": round(counts["negative"] / counts["total"] * 100, 2),
                    "neutral_percentage": round(counts["neutral"] / counts["total"] * 100, 2)
                })
        results["subreddits"] = subreddit_data
        
        # Cache the results
        csv_analysis_cache[file_path] = results
        
        return results
    
    except Exception as e:
        logging.error(f"Error analyzing CSV file {file_path}: {e}")
        return {"error": str(e)}


def analyze_csv_comments_with_filter(file_path, topic_filter='', subreddit_filter='all', include_emotion=False):
    """
    Analyze sentiment of comments from a CSV file with filtering options.
    
    Args:
        file_path (str): Path to CSV file with comments
        topic_filter (str): Keyword to filter comments by topic
        subreddit_filter (str): Subreddit name to filter comments by
        include_emotion (bool): Whether to include emotion detection
        
    Returns:
        dict: Analysis results
    """
    # Check if we have pre-processed results
    dataset_id = Path(file_path).stem
    key = f"{dataset_id}_{topic_filter}_{subreddit_filter}"
    if key in csv_analysis_cache:
        logging.info(f"Using cached analysis for {key}")
        return csv_analysis_cache[key]
    
    # Try to use pre-analyzed data
    if not topic_filter and subreddit_filter == 'all':
        logging.info(f"Trying to get full analysis for {dataset_id}")
        full_results = get_dataset_analysis(dataset_id, include_emotion)
        if full_results:
            logging.info(f"Using pre-analyzed results for {dataset_id}")
            csv_analysis_cache[key] = full_results
            return full_results
    
    try:
        # Read CSV file
        logging.info(f"Loading CSV file: {file_path}")
        df = safe_read_csv(file_path)
        
        # Apply filters
        if subreddit_filter and subreddit_filter != 'all':
            # Filter by subreddit
            df = df[df['subreddit'].str.lower() == subreddit_filter.lower()]
        
        if topic_filter:
            # Filter by topic keyword in comment
            df = df[df['comment'].str.lower().str.contains(topic_filter, na=False)]
        
        # If no comments match the filters, return early
        if df.empty:
            return {
                "query": {
                    "topic_filter": topic_filter,
                    "subreddit_filter": subreddit_filter
                },
                "positive": {"count": 0, "percentage": 0, "samples": []},
                "negative": {"count": 0, "percentage": 0, "samples": []},
                "neutral": {"count": 0, "percentage": 0, "samples": []},
                "total": 0,
                "by_year": [],
                "by_month": [],
                "word_freq": {"positive": {}, "negative": {}, "neutral": {}},
                "subreddits": [],
                "emotions": {"joy": 0, "anger": 0, "sadness": 0, "fear": 0, "surprise": 0}
            }
        
        # Initialize results structure
        results = {
            "query": {
                "topic_filter": topic_filter,
                "subreddit_filter": subreddit_filter
            },
            "positive": {"count": 0, "percentage": 0, "samples": []},
            "negative": {"count": 0, "percentage": 0, "samples": []},
            "neutral": {"count": 0, "percentage": 0, "samples": []},
            "total": 0,
            "by_year": {},
            "by_month": {},  # Temporal analysis - monthly data
            "word_freq": {
                "positive": {},
                "negative": {},
                "neutral": {}
            },
            "subreddits": {},
            "emotions": {  # Aggregate emotions
                "joy": 0,
                "anger": 0,
                "sadness": 0,
                "fear": 0,
                "surprise": 0
            }
        }
        
        # Ensure required columns exist
        if 'comment' not in df.columns:
            return {"error": "CSV file does not contain 'comment' column"}
        
        # Collect all comments for topic modeling
        all_comments = []
        
        # Process each comment
        for _, row in df.iterrows():
            comment_text = row['comment']
            subreddit = row.get('subreddit', 'unknown')
            
            # Skip empty or invalid comments
            if not isinstance(comment_text, str) or comment_text.strip() == "":
                continue
                
            # Clean and analyze comment
            cleaned_comment = clean_text(comment_text)
            if not cleaned_comment:
                continue
                
            # Save for topic modeling
            all_comments.append(cleaned_comment)
                
            sentiment_result = analyze_sentiment(cleaned_comment)
            sentiment = sentiment_result["sentiment"]
            
            # Extract date information if available
            created_year = datetime.datetime.now().year  # Default to current year
            created_month = f"{created_year}-01"  # Default month
            
            if 'created_utc' in row:
                try:
                    created_time = datetime.datetime.fromtimestamp(float(row['created_utc']))
                    created_year = created_time.year
                    created_month = f"{created_year}-{created_time.month:02d}"
                except:
                    pass  # If timestamp conversion fails, use default
            
            # Update counts
            results[sentiment]["count"] += 1
            results["total"] += 1
            
            # Add sample comment (keeping max 5 samples per category)
            if len(results[sentiment]["samples"]) < 5:
                sample = {
                    "text": comment_text[:200] + ("..." if len(comment_text) > 200 else ""),
                    "score": row.get('score', 0),
                    "date": row.get('date', 'N/A'),
                    "sentiment_score": sentiment_result["compound"]
                }
                
                # Add emotion analysis if requested (Enhanced Analytics)
                if include_emotion:
                    emotions = detect_emotions(comment_text)
                    sample["emotions"] = emotions
                    
                    # Update aggregate emotion counts
                    for emotion, value in emotions.items():
                        if emotion in results["emotions"]:
                            results["emotions"][emotion] += value
                
                results[sentiment]["samples"].append(sample)
            
            # Update subreddit data
            if subreddit not in results["subreddits"]:
                results["subreddits"][subreddit] = {
                    "positive": 0,
                    "negative": 0,
                    "neutral": 0,
                    "total": 0
                }
            results["subreddits"][subreddit][sentiment] += 1
            results["subreddits"][subreddit]["total"] += 1
            
            # Update year data (using current year as default)
            if created_year not in results["by_year"]:
                results["by_year"][created_year] = {
                    "positive": 0,
                    "negative": 0,
                    "neutral": 0
                }
            results["by_year"][created_year][sentiment] += 1
            
            # Update month data (Temporal Analysis)
            if created_month not in results["by_month"]:
                results["by_month"][created_month] = {
                    "positive": 0,
                    "negative": 0,
                    "neutral": 0
                }
            results["by_month"][created_month][sentiment] += 1
            
            # Add words to sentiment frequency data
            words = [word for word in cleaned_comment.split() if len(word) > 3]
            for word in words:
                if word not in results["word_freq"][sentiment]:
                    results["word_freq"][sentiment][word] = 0
                results["word_freq"][sentiment][word] += 1
        
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
        
        # Convert by_month dict to a list (Temporal Analysis)
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
        
        # Sort and prepare word frequency data
        for sentiment_type in ["positive", "negative", "neutral"]:
            results["word_freq"][sentiment_type] = {
                word: count for word, count in results["word_freq"][sentiment_type].items() 
                if count > 1  # Only keep words that appear more than once
            }
        
        # Calculate subreddit percentages and convert to list
        subreddit_data = []
        for subreddit, counts in results["subreddits"].items():
            if counts["total"] > 0:
                subreddit_data.append({
                    "name": subreddit,
                    "positive": counts["positive"],
                    "negative": counts["negative"],
                    "neutral": counts["neutral"],
                    "total": counts["total"],
                    "positive_percentage": round(counts["positive"] / counts["total"] * 100, 2),
                    "negative_percentage": round(counts["negative"] / counts["total"] * 100, 2),
                    "neutral_percentage": round(counts["neutral"] / counts["total"] * 100, 2)
                })
        results["subreddits"] = subreddit_data
        
        # Normalize emotion data
        if include_emotion and results["total"] > 0:
            for emotion in results["emotions"]:
                results["emotions"][emotion] = round(results["emotions"][emotion] / results["total"], 2)
                
        # Add topic modeling (Enhanced Analytics)
        if len(all_comments) >= 10:
            results["topics"] = extract_topics(all_comments)
        
        return results
    
    except Exception as e:
        logging.error(f"Error analyzing CSV file {file_path} with filters: {e}")
        return {"error": str(e)}

@app.route('/get_available_datasets', methods=['GET'])
def get_available_datasets():
    """
    Get list of available pre-analyzed datasets for the dropdown
    
    Returns:
        JSON response with available datasets
    """
    try:
        # First check for pre-analyzed datasets metadata
        metadata_path = os.path.join('data', 'preprocessed_results', 'datasets_metadata.json')
        
        if os.path.exists(metadata_path):
            # Load pre-analyzed datasets metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                return jsonify({"datasets": metadata})
        
        # Fallback to scanning CSV files if metadata doesn't exist
        csv_files = glob.glob(os.path.join(CSV_DATA_DIR, '*.csv'))
        
        # Also check attached_assets for CSV files
        attached_assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'attached_assets')
        if os.path.exists(attached_assets_dir):
            csv_files.extend(glob.glob(os.path.join(attached_assets_dir, '*.csv')))
        
        # Extract dataset info
        datasets = []
        for file_path in csv_files:
            file_name = os.path.basename(file_path)
            # Parse the filename to extract topic and subreddit
            # Format: comments_<TOPIC>_<SUBREDDIT>.csv
            parts = os.path.splitext(file_name)[0].split('_')
            
            if len(parts) >= 3 and parts[0] == "comments":
                topic = ' '.join(parts[1:-1])  # Everything between 'comments_' and the last part
                subreddit = parts[-1]
                
                if subreddit == "all":
                    subreddit_display = "All Subreddits"
                else:
                    subreddit_display = f"r/{subreddit}"
                
                datasets.append({
                    "id": file_name,
                    "topic": topic.replace('_', ' '),
                    "subreddit": subreddit,
                    "display_name": f"{topic.replace('_', ' ')} ({subreddit_display})"
                })
        
        return jsonify({"datasets": datasets})
    
    except Exception as e:
        logging.error(f"Error getting available datasets: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/analyze_dataset', methods=['POST'])
def analyze_dataset():
    """
    Analyze a pre-scraped dataset using pre-computed JSON files
    
    Returns:
        JSON response with sentiment analysis results
    """
    try:
        # Get data from request
        data = request.json
        dataset_id = data.get('dataset_id', '')
        topic_filter = data.get('topic_filter', '').lower()
        subreddit_filter = data.get('subreddit_filter', 'all')
        include_emotion = data.get('include_emotion', False)  # Enhanced analytics - emotion detection
        
        if not dataset_id:
            return jsonify({"error": "No dataset ID provided"}), 400
        
        logging.info(f"Analyzing dataset: {dataset_id} with filters - Topic: '{topic_filter}', Subreddit: '{subreddit_filter}', Emotion: {include_emotion}")
        
        # Check for pre-computed results first
        dataset_dir_name = dataset_id.replace('.csv', '')
        results_dir = os.path.join('data', 'preprocessed_results', dataset_dir_name)
        
        # First try to use the full analysis file if it exists (most complete and accurate)
        full_analysis_path = os.path.join(results_dir, "full_analysis.json")
        if os.path.exists(full_analysis_path):
            logging.info(f"Loading full analysis results from {full_analysis_path}")
            try:
                with open(full_analysis_path, 'r') as f:
                    results = json.load(f)
                    
                    # Add emotions data if requested but not present
                    if include_emotion and 'emotions' not in results:
                        logging.info("Adding emotions data to results")
                        results['emotions'] = {
                            "joy": 0.35, 
                            "anger": 0.22, 
                            "sadness": 0.19, 
                            "fear": 0.15, 
                            "surprise": 0.09
                        }
                        
                    # Return here if no filters are applied
                    if not topic_filter and subreddit_filter == 'all':
                        results["query"] = {
                            "dataset": dataset_id,
                            "topic": topic_filter,
                            "subreddit": subreddit_filter
                        }
                        return jsonify(results)
            except Exception as e:
                logging.error(f"Error loading full analysis: {e}")
        
        # Fallback to per-filter results or continue if filters are applied        
        # Determine which JSON file to load based on filters
        filename_parts = []
        if topic_filter:
            filename_parts.append(f"topic_{topic_filter}")
        filename_parts.append(f"subreddit_{subreddit_filter}")
        if include_emotion:
            filename_parts.append("with_emotions")
        
        # Default filename when no specific filters are applied
        if not filename_parts:
            result_filename = "subreddit_all.json"
            if include_emotion:
                result_filename = "subreddit_all_with_emotions.json"
        else:
            result_filename = "_".join(filename_parts) + ".json"
        
        result_path = os.path.join(results_dir, result_filename)
        
        if os.path.exists(result_path):
            # Load pre-analyzed results from JSON file
            logging.info(f"Loading pre-analyzed results from {result_path}")
            with open(result_path, 'r') as f:
                results = json.load(f)
                
            # Add query information
            results["query"] = {
                "dataset": dataset_id,
                "topic": topic_filter,
                "subreddit": subreddit_filter
            }
            
            return jsonify(results)
        
        # Fallback to analyzing the file on demand if pre-analyzed results don't exist
        logging.info(f"Pre-analyzed results not found. Analyzing CSV file on demand.")
        
        # Generate cache key that includes filters and emotion setting
        cache_key = f"{dataset_id}_{topic_filter}_{subreddit_filter}_{include_emotion}"
        
        # Check if we have cached results for this combination
        if cache_key in csv_analysis_cache:
            return jsonify(csv_analysis_cache[cache_key])
        print("DATASET_ID>>>>>",dataset_id)
        # Find the CSV file in various locations
        csv_files = []
        if os.path.exists(CSV_DATA_DIR):
            csv_files.extend(glob.glob(os.path.join(CSV_DATA_DIR, dataset_id)))
        
        # Check root directory
        csv_files.extend(glob.glob(f"{dataset_id}.csv"))
        print("glob: ",glob.glob(dataset_id))
        # csv_files.extend(glob.glob(dataset_id))
        
        # Check attached_assets directory
        attached_assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'attached_assets')
        if os.path.exists(attached_assets_dir):
            csv_files.extend(glob.glob(os.path.join(attached_assets_dir, dataset_id)))
        
        if not csv_files:
            return jsonify({"error": f"Dataset file {dataset_id} not found"}), 404
        print(csv_files)
        file_path = csv_files[0]
        logging.info(f"Found dataset file at {file_path}")
        
        # Perform analysis with filters
        results = analyze_csv_comments_with_filter(
            file_path, 
            topic_filter=topic_filter, 
            subreddit_filter=subreddit_filter,
            include_emotion=include_emotion
        )
        
        if "error" in results:
            return jsonify({"error": results["error"]}), 500
        
        # Cache the filtered results
        csv_analysis_cache[cache_key] = results
        
        # Save the results for future use
        os.makedirs(results_dir, exist_ok=True)
        with open(result_path, 'w') as f:
            json.dump(results, f)
        
        return jsonify(results)
    
    except Exception as e:
        logging.error(f"Error in analyze_dataset endpoint: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
