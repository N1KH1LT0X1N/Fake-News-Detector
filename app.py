import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template_string, make_response
import sys
import os
import webbrowser
import threading
import time
import socket
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
from flask import render_template
from vectorizer_utils import clean_text, create_pipeline


print("Running the fake news detector script (version 2025-03-25 v8)")

app = Flask(__name__)

# Global variables
models = {}
tfidf_vectorizer = None
dataset_tfidf = None
df = None

def find_available_port(start_port=5000):
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
    model_files = {
        "Naive Bayes": 'models/nb_pipeline.pkl',
        "Random Forest": 'models/rf_pipeline.pkl',
        "SVM": 'models/svm_pipeline.pkl'
    }
    data_files = ['models/dataset_tfidf.pkl', 'models/preprocessed_df.pkl']

    if all(os.path.exists(f) for f in list(model_files.values()) + data_files):
        print("Loading pre-trained models and data...")
        for name, file in model_files.items():
            models[name] = joblib.load(file)
        tfidf_vectorizer = models["Naive Bayes"].named_steps['tfidf']
        dataset_tfidf = joblib.load('models/dataset_tfidf.pkl')
        df = joblib.load('models/preprocessed_df.pkl')
    else:
        print("Precomputed files not found. Training models...")
        try:
            fake_news = pd.read_csv(r"data\Fake-1.csv", usecols=['text'])
            true_news = pd.read_csv(r"data\True-1.csv", usecols=['text'])
        except FileNotFoundError:
            print("Error: 'data\Fake-1.csv' or 'data\True-1.csv' not found. Please ensure they are in the provided directory.")
            sys.exit(1)
        
        fake_news['label'] = 0
        true_news['label'] = 1
        df = pd.concat([fake_news, true_news])
        df = df.sample(n=5000, random_state=42)
        df["text"] = df["text"].apply(clean_text)
        X_train, _, y_train, _ = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

        nb_pipeline = create_pipeline(MultinomialNB())
        rf_pipeline = create_pipeline(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        svm_pipeline = create_pipeline(LinearSVC())

        print("Training Naive Bayes...")
        nb_pipeline.fit(X_train, y_train)
        print("Training Random Forest...")
        rf_pipeline.fit(X_train, y_train)
        print("Training SVM...")
        svm_pipeline.fit(X_train, y_train)

        models = {"Naive Bayes": nb_pipeline, "Random Forest": rf_pipeline, "SVM": svm_pipeline}
        tfidf_vectorizer = models["Naive Bayes"].named_steps['tfidf']
        dataset_tfidf = tfidf_vectorizer.transform(df["text"])

        joblib.dump(nb_pipeline, 'models/nb_pipeline.pkl')
        joblib.dump(rf_pipeline, 'models/rf_pipeline.pkl')
        joblib.dump(svm_pipeline, 'models/svm_pipeline.pkl')
        joblib.dump(dataset_tfidf, 'models/dataset_tfidf.pkl')
        joblib.dump(df, 'models/preprocessed_df.pkl')
        print("Models and data trained and saved successfully!")

SOURCE_CREDIBILITY = {
    "cnn.com": "High",
    "nytimes.com": "High",
    "foxnews.com": "Medium",
    "infowars.com": "Low",
    "breitbart.com": "Low"
}

def check_source_credibility(text):
    url_pattern = r'(https?://)?([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'
    matches = re.findall(url_pattern, text)
    for _, domain in matches:
        domain = domain.lower()
        for known_domain, credibility in SOURCE_CREDIBILITY.items():
            if known_domain in domain:
                return f"Source Credibility: {credibility} (Detected: {known_domain})"
    return "Source Credibility: Unknown (No recognizable source detected)"

def predict_news(news_article, model_choice):
    if not models:
        train_and_load_models()
    cleaned_text = clean_text(news_article)
    pipeline = models[model_choice]
    prediction = pipeline.predict([cleaned_text])[0]
    probabilities = pipeline.predict_proba([cleaned_text])[0] if model_choice != "SVM" else [0.5, 0.5]

    label = "Real" if prediction == 1 else "Fake"
    prob_fake = probabilities[0] * 100
    prob_real = probabilities[1] * 100

    tfidf = pipeline.named_steps['tfidf']
    model = pipeline.named_steps['model']
    tfidf_vector = tfidf.transform([cleaned_text])
    feature_names = tfidf.get_feature_names_out()
    tfidf_scores = tfidf_vector.toarray()[0]

    if model_choice == "Naive Bayes":
        fake_log_probs = model.feature_log_prob_[0]
        word_contributions = {}
        for idx, score in enumerate(tfidf_scores):
            if score > 0:
                word = feature_names[idx]
                contribution = score * fake_log_probs[idx]
                word_contributions[word] = contribution
    else:
        if model_choice == "Random Forest":
            coef = model.feature_importances_
        else:
            coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
        word_contributions = {}
        for idx, score in enumerate(tfidf_scores):
            if score > 0:
                if idx >= len(coef):
                    continue
                word = feature_names[idx]
                contribution = score * coef[idx]
                word_contributions[word] = contribution

    top_fake_words = sorted(word_contributions.items(), key=lambda x: x[1], reverse=True)[:5]
    fake_words = set(word for word, _ in top_fake_words)

    words = news_article.split()
    highlighted_article = []
    for word in words:
        cleaned_word = clean_text(word)
        if cleaned_word in fake_words:
            highlighted_article.append(f'<span class="highlight-fake">{word}</span>')
        else:
            highlighted_article.append(word)
    highlighted_text = " ".join(highlighted_article)

    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(news_article)
    compound_score = sentiment_scores['compound']
    sentiment = "Positive" if compound_score > 0.05 else "Negative" if compound_score < -0.05 else "Neutral"

    credibility = check_source_credibility(news_article)

    input_tfidf = tfidf_vectorizer.transform([cleaned_text])
    similarities = cosine_similarity(input_tfidf, dataset_tfidf)[0]
    top_indices = similarities.argsort()[-3:][::-1]
    similar_articles = []
    for idx in top_indices:
        sim_score = similarities[idx] * 100
        article_text = df.iloc[idx]["text"][:200] + "..."
        article_label = "Real" if df.iloc[idx]["label"] == 1 else "Fake"
        similar_articles.append(f"<p><strong>Similar Article #{len(similar_articles)+1} (Similarity: {sim_score:.2f}%):</strong> {article_text} <br><em>Label: {article_label}</em></p>")

    word_contribution_data = [{"word": word, "contribution": float(contribution)} for word, contribution in top_fake_words]

    output = f"<p><strong>Prediction:</strong> {label}</p>"
    output += f"<p><strong>Probability of being Fake:</strong> {prob_fake:.2f}%</p>"
    output += f"<p><strong>Probability of being Real:</strong> {prob_real:.2f}%</p>"
    output += f"<p><strong>Sentiment:</strong> {sentiment} (Compound Score: {compound_score:.2f})</p>"
    output += f"<p><strong>{credibility}</strong></p>"
    output += "<p><strong>Highlighted Article (highlighted parts indicate potentially fake content):</strong></p>"
    output += f"<p>{highlighted_text}</p>"
    output += "<h3>Top Contributing Words:</h3>"
    output += '<div id="wordChart" style="width: 100%; height: 300px;"></div>'
    output += f'<script>var wordData = {json.dumps(word_contribution_data)};</script>'
    output += "<h3>Similar Articles in Dataset:</h3>"
    output += "".join(similar_articles)

    return output


@app.route('/', methods=['GET', 'POST'])
def index():
    news_article = None
    model_choice = "Naive Bayes"
    output = None
    if request.method == 'POST':
        news_article = request.form.get('news_article', '')
        model_choice = request.form.get('model_choice', 'Naive Bayes')
        if news_article:
            output = predict_news(news_article, model_choice)
    response = make_response(render_template("index.html", news_article=news_article, model_choice=model_choice, output=output))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response

def run_flask():
    port = find_available_port(start_port=5000)
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



