import pandas as pd
import numpy as np
import re
import string
import sys
import os
import joblib
import json
from urllib.parse import urlparse

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

from flask import Flask, request, jsonify
from flask_cors import CORS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from vectorizer_utils import clean_text, create_pipeline

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables
models = {}
tfidf_vectorizer = None
dataset_tfidf = None
df = None

SOURCE_CREDIBILITY = {
    "cnn.com": "High",
    "nytimes.com": "High",
    "washingtonpost.com": "High",
    "reuters.com": "High",
    "bbc.com": "High",
    "npr.org": "High",
    "apnews.com": "High",
    "foxnews.com": "Medium",
    "usatoday.com": "Medium",
    "cbsnews.com": "Medium",
    "nbcnews.com": "Medium",
    "abcnews.go.com": "Medium",
    "breitbart.com": "Low",
    "infowars.com": "Low",
    "naturalnews.com": "Low",
    "theonion.com": "Satire",
    "babylonbee.com": "Satire"
}

def train_and_load_models():
    """Train and load ML models"""
    global models, tfidf_vectorizer, dataset_tfidf, df

    model_paths = {
        "Naive Bayes": 'models/nb_pipeline.pkl',
        "Random Forest": 'models/rf_pipeline.pkl'
    }
    data_paths = ['models/dataset_tfidf.pkl', 'models/preprocessed_df.pkl']

    if all(os.path.exists(path) for path in list(model_paths.values()) + data_paths):
        print("Loading pre-trained models and data...")
        for name, path in model_paths.items():
            models[name] = joblib.load(path)
        tfidf_vectorizer = models["Naive Bayes"].named_steps['tfidf']
        dataset_tfidf = joblib.load('models/dataset_tfidf.pkl')
        df = joblib.load('models/preprocessed_df.pkl')
        print("Models loaded successfully!")
    else:
        print("Training models from scratch...")
        try:
            fake_news = pd.read_csv("data/Fake-1.csv", usecols=['text'])
            true_news = pd.read_csv("data/True-1.csv", usecols=['text'])
        except FileNotFoundError:
            print("Error: CSV files not found in 'data/' directory.")
            return False

        fake_news['label'] = 0
        true_news['label'] = 1
        df = pd.concat([fake_news, true_news]).sample(n=39998, random_state=42)
        df['text'] = df['text'].apply(clean_text)
        
        X_train, _, y_train, _ = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

        nb_pipeline = create_pipeline(MultinomialNB())
        rf_pipeline = create_pipeline(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))

        print("Training models...")
        nb_pipeline.fit(X_train, y_train)
        rf_pipeline.fit(X_train, y_train)

        models = {
            "Naive Bayes": nb_pipeline,
            "Random Forest": rf_pipeline
        }
        tfidf_vectorizer = nb_pipeline.named_steps['tfidf']
        dataset_tfidf = tfidf_vectorizer.transform(df['text'])

        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        joblib.dump(nb_pipeline, model_paths['Naive Bayes'])
        joblib.dump(rf_pipeline, model_paths['Random Forest'])
        joblib.dump(dataset_tfidf, 'models/dataset_tfidf.pkl')
        joblib.dump(df, 'models/preprocessed_df.pkl')
        print("Models trained and saved successfully!")
    
    return True

def check_source_credibility(url):
    """Check source credibility based on URL"""
    try:
        domain = urlparse(url).netloc.lower()
        # Remove www. prefix
        domain = domain.replace('www.', '')
        
        for known_source, credibility in SOURCE_CREDIBILITY.items():
            if known_source in domain:
                return {
                    "credibility": credibility,
                    "source": known_source,
                    "domain": domain
                }
        return {
            "credibility": "Unknown",
            "source": "Unknown",
            "domain": domain
        }
    except:
        return {
            "credibility": "Unknown",
            "source": "Unknown",
            "domain": "Unknown"
        }

def extract_top_words(text, model_choice, top_n=5):
    """Extract top contributing words for fake news prediction"""
    if not models:
        return []

    try:
        cleaned_text = clean_text(text)
        pipeline = models[model_choice]
        
        tfidf = pipeline.named_steps['tfidf']
        model = pipeline.named_steps['model']
        tfidf_vector = tfidf.transform([cleaned_text])
        feature_names = tfidf.get_feature_names_out()
        tfidf_scores = tfidf_vector.toarray()[0]

        word_contributions = {}

        if model_choice == "Naive Bayes":
            probs = model.feature_log_prob_[0]
        else:
            if hasattr(model, "feature_importances_"):
                probs = model.feature_importances_
            elif hasattr(model, "coef_"):
                probs = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            else:
                return []

        for idx, score in enumerate(tfidf_scores):
            if score > 0 and idx < len(probs):
                word = feature_names[idx]
                word_contributions[word] = float(score * probs[idx])

        top_words = sorted(word_contributions.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [{"word": word, "contribution": contrib} for word, contrib in top_words]
    
    except Exception as e:
        print(f"Error in extract_top_words: {e}")
        return []

def find_similar_articles(text, top_n=3):
    """Find similar articles in dataset"""
    if not tfidf_vectorizer or dataset_tfidf is None or df is None:
        return []

    try:
        cleaned_text = clean_text(text)
        input_vec = tfidf_vectorizer.transform([cleaned_text])
        similarities = cosine_similarity(input_vec, dataset_tfidf)[0]
        top_indices = similarities.argsort()[-top_n:][::-1]

        similar_articles = []
        for i, idx in enumerate(top_indices):
            article_text = df.iloc[idx]['text']
            similarity_score = float(similarities[idx] * 100)
            label = "Real" if df.iloc[idx]['label'] == 1 else "Fake"
            
            similar_articles.append({
                "rank": i + 1,
                "similarity": round(similarity_score, 2),
                "text_preview": article_text[:200] + "..." if len(article_text) > 200 else article_text,
                "label": label
            })
        
        return similar_articles
    except Exception as e:
        print(f"Error in find_similar_articles: {e}")
        return []

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models_loaded": len(models) > 0
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    """Main API endpoint for analyzing text"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "error": "Missing 'text' field in request body"
            }), 400

        text = data.get('text', '')
        url = data.get('url', '')
        model_choice = data.get('model', 'Naive Bayes')
        threshold = float(data.get('threshold', 0.5))

        if not text.strip():
            return jsonify({
                "error": "Empty text provided"
            }), 400

        if model_choice not in models:
            return jsonify({
                "error": f"Invalid model choice. Available models: {list(models.keys())}"
            }), 400

        # Clean and analyze text
        cleaned_text = clean_text(text)
        pipeline = models[model_choice]
        
        # Get predictions
        probabilities = pipeline.predict_proba([cleaned_text])[0]
        prob_fake = float(probabilities[0] * 100)
        prob_real = float(probabilities[1] * 100)
        prediction = 0 if probabilities[0] > threshold else 1
        label = "Real" if prediction == 1 else "Fake"

        # Sentiment analysis
        sentiment_analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = sentiment_analyzer.polarity_scores(text)
        sentiment_compound = float(sentiment_scores['compound'])
        
        if sentiment_compound > 0.05:
            sentiment_label = "Positive"
        elif sentiment_compound < -0.05:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"

        # Source credibility
        credibility_info = check_source_credibility(url) if url else {
            "credibility": "Unknown",
            "source": "No URL provided",
            "domain": "Unknown"
        }

        # Top contributing words
        top_words = extract_top_words(text, model_choice)

        # Similar articles
        similar_articles = find_similar_articles(text)

        # Prepare response
        response = {
            "prediction": {
                "label": label,
                "confidence": {
                    "fake": round(prob_fake, 2),
                    "real": round(prob_real, 2)
                },
                "threshold_used": threshold
            },
            "sentiment": {
                "label": sentiment_label,
                "compound_score": round(sentiment_compound, 2),
                "scores": {
                    "positive": float(sentiment_scores['pos']),
                    "negative": float(sentiment_scores['neg']),
                    "neutral": float(sentiment_scores['neu'])
                }
            },
            "source_credibility": credibility_info,
            "analysis": {
                "top_contributing_words": top_words,
                "similar_articles": similar_articles,
                "text_length": len(text),
                "word_count": len(text.split())
            },
            "metadata": {
                "model_used": model_choice,
                "url": url,
                "timestamp": pd.Timestamp.now().isoformat()
            }
        }

        return jsonify(response)

    except Exception as e:
        print(f"Error in analyze_text: {e}")
        return jsonify({
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route('/api/models', methods=['GET'])
def get_available_models():
    """Get list of available models"""
    return jsonify({
        "available_models": list(models.keys()),
        "default_model": "Naive Bayes"
    })

@app.route('/api/sources', methods=['GET'])
def get_source_credibility_db():
    """Get the source credibility database"""
    return jsonify({
        "sources": SOURCE_CREDIBILITY,
        "total_sources": len(SOURCE_CREDIBILITY)
    })

# Initialize models on startup
if not train_and_load_models():
    print("Failed to load models. Exiting...")
    sys.exit(1)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)