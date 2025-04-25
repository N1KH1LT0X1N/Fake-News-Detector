import pandas as pd
import numpy as np
import re
import string
import sys
import os
import webbrowser
import threading
import time
import socket
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

from flask import Flask, request, make_response, render_template
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from vectorizer_utils import clean_text, create_pipeline

app = Flask(__name__)

# Global variables
models = {}
tfidf_vectorizer = None
dataset_tfidf = None
df = None

SOURCE_CREDIBILITY = {
    "cnn.com": "High",
    "nytimes.com": "High",
    "foxnews.com": "Medium",
    "infowars.com": "Low",
    "breitbart.com": "Low"
}

def find_available_port(start_port=7860):
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', port))
                return port
            except OSError:
                port += 1

def train_and_load_models():
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
    else:
        print("Precomputed files not found. Training models...")
        try:
            fake_news = pd.read_csv("data/Fake-1.csv", usecols=['text'])
            true_news = pd.read_csv("data/True-1.csv", usecols=['text'])
        except FileNotFoundError:
            print("Error: CSV files not found in 'data/' directory.")
            sys.exit(1)

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

        joblib.dump(nb_pipeline, model_paths['Naive Bayes'])
        joblib.dump(rf_pipeline, model_paths['Random Forest'])
        joblib.dump(dataset_tfidf, 'models/dataset_tfidf.pkl')
        joblib.dump(df, 'models/preprocessed_df.pkl')
        print("Models and data trained and saved successfully.")

def check_source_credibility(text):
    url_pattern = r'(https?://)?([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'
    matches = re.findall(url_pattern, text)
    for _, domain in matches:
        domain = domain.lower()
        for known, credibility in SOURCE_CREDIBILITY.items():
            if known in domain:
                return f"Source Credibility: {credibility} (Detected: {known})"
    return "Source Credibility: Unknown (No recognizable source detected)"

def predict_news(news_article, model_choice, threshold=0.5):
    if not models:
        train_and_load_models()

    cleaned_text = clean_text(news_article)
    pipeline = models[model_choice]
    probabilities = pipeline.predict_proba([cleaned_text])[0]

    prob_fake = probabilities[0] * 100
    prob_real = probabilities[1] * 100
    prediction = 0 if probabilities[0] > threshold else 1

    label = "Real" if prediction == 1 else "Fake"

    tfidf = pipeline.named_steps['tfidf']
    model = pipeline.named_steps['model']
    tfidf_vector = tfidf.transform([cleaned_text])
    feature_names = tfidf.get_feature_names_out()
    tfidf_scores = tfidf_vector.toarray()[0]

    word_contributions = {}

    try:
        if model_choice == "Naive Bayes":
            probs = model.feature_log_prob_[0]
        else:
            if hasattr(model, "feature_importances_"):
                probs = model.feature_importances_
            elif hasattr(model, "coef_"):
                probs = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            else:
                probs = []

        for idx, score in enumerate(tfidf_scores):
            if score > 0 and idx < len(probs):
                word = feature_names[idx]
                word_contributions[word] = score * probs[idx]
    except Exception as e:
        print(f"[Error] Word contribution failed for model '{model_choice}': {e}")

    top_fake_words = sorted(word_contributions.items(), key=lambda x: x[1], reverse=True)[:5]
    fake_words = set(word for word, _ in top_fake_words)

    highlighted_text = " ".join(
        f'<span class="highlight-fake">{word}</span>' if clean_text(word) in fake_words else word
        for word in news_article.split()
    )

    sentiment = SentimentIntensityAnalyzer().polarity_scores(news_article)['compound']
    sentiment_label = "Positive" if sentiment > 0.05 else "Negative" if sentiment < -0.05 else "Neutral"

    credibility = check_source_credibility(news_article)

    input_vec = tfidf_vectorizer.transform([cleaned_text])
    similarities = cosine_similarity(input_vec, dataset_tfidf)[0]
    top_indices = similarities.argsort()[-3:][::-1]

    similar_articles = [
        f"<p><strong>Similar Article #{i+1} (Similarity: {similarities[idx]*100:.2f}%):</strong> {df.iloc[idx]['text'][:200]}...<br><em>Label: {'Real' if df.iloc[idx]['label'] == 1 else 'Fake'}</em></p>"
        for i, idx in enumerate(top_indices)
    ]

    word_data = [{"word": w, "contribution": float(c)} for w, c in top_fake_words]

    output = f"""
        <p><strong>Prediction:</strong> {label}</p>
        <p><strong>Probability of being Fake:</strong> {prob_fake:.2f}%</p>
        <p><strong>Probability of being Real:</strong> {prob_real:.2f}%</p>
        <p><strong>Threshold Used:</strong> {threshold}</p>
        <p><strong>Sentiment:</strong> {sentiment_label} (Compound Score: {sentiment:.2f})</p>
        <p><strong>{credibility}</strong></p>
        <p><strong>Highlighted Article (highlighted parts indicate potentially fake content):</strong></p>
        <p>{highlighted_text}</p>
        <h3>Top Contributing Words:</h3>
        <div id=\"wordChart\" style=\"width: 100%; height: 300px;\"></div>
        <script>var wordData = {json.dumps(word_data)};</script>
        <h3>Similar Articles in Dataset:</h3>
        {''.join(similar_articles)}
    """
    return output

@app.route('/', methods=['GET', 'POST'])
def index():
    news_article = model_choice = output = None
    threshold = 0.5
    if request.method == 'POST':
        news_article = request.form.get('news_article', '')
        model_choice = request.form.get('model_choice', 'Naive Bayes')
        try:
            threshold = float(request.form.get('threshold', 0.5))
        except ValueError:
            threshold = 0.5
        if news_article:
            output = predict_news(news_article, model_choice, threshold)
    response = make_response(render_template("index.html", news_article=news_article, model_choice=model_choice, output=output, threshold=threshold))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response

def run_flask():
    port = find_available_port(start_port=7860)
    url = f"http://127.0.0.1:{port}"
    print(f"Starting Flask server on {url}...")
    threading.Timer(0.5, lambda: webbrowser.open_new_tab(url)).start()
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

# Execute everything
train_and_load_models()
threading.Thread(target=run_flask, daemon=True).start()

# Keep the script running in Jupyter (optional)
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopped by user.")

if __name__ == "__main__":
    train_and_load_models()
    app.run(host='0.0.0.0', port=7860)
