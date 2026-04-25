from flask import Flask, jsonify, render_template
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

# === Load datasets ===
listings = pd.read_csv("listings.csv", low_memory=False)
reviews = pd.read_csv("reviews.csv", low_memory=False)

# === Sentiment functions ===
def sentiment_textblob(text):
    return TextBlob(text).sentiment.polarity if pd.notnull(text) else 0

analyzer = SentimentIntensityAnalyzer()
def sentiment_vader(text):
    return analyzer.polarity_scores(text)['compound'] if pd.notnull(text) else 0

# === Compute both sentiments ===
reviews['textblob'] = reviews['comments'].astype(str).apply(sentiment_textblob)
reviews['vader'] = reviews['comments'].astype(str).apply(sentiment_vader)

# === Hybrid score function ===
def hybrid_score(x, alpha=0.6):
    median_sent = x.median()
    pos_ratio = (x > 0).mean()
    return alpha * median_sent + (1 - alpha) * pos_ratio

# === Aggregate to listing level ===
textblob_scores = reviews.groupby("listing_id")['textblob'].apply(hybrid_score).reset_index()
textblob_scores.rename(columns={'textblob': 'hybrid_sentiment'}, inplace=True)

vader_scores = reviews.groupby("listing_id")['vader'].apply(hybrid_score).reset_index()
vader_scores.rename(columns={'vader': 'hybrid_sentiment'}, inplace=True)

# === Add listing details ===
list_cols = ['id', 'name', 'neighbourhood_cleansed']
textblob_scores = textblob_scores.merge(listings[list_cols], left_on="listing_id", right_on="id", how="left")
vader_scores = vader_scores.merge(listings[list_cols], left_on="listing_id", right_on="id", how="left")

# === API endpoints ===
@app.route("/api/top_positive/<model>")
def top_positive(model):
    df = textblob_scores if model == "textblob" else vader_scores
    top_pos = df.sort_values(by="hybrid_sentiment", ascending=False).head(10)
    return jsonify(top_pos[['listing_id', 'name', 'neighbourhood_cleansed', 'hybrid_sentiment']].to_dict(orient="records"))

@app.route("/api/top_negative/<model>")
def top_negative(model):
    df = textblob_scores if model == "textblob" else vader_scores
    top_neg = df.sort_values(by="hybrid_sentiment", ascending=True).head(10)
    return jsonify(top_neg[['listing_id', 'name', 'neighbourhood_cleansed', 'hybrid_sentiment']].to_dict(orient="records"))

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
