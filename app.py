from flask import Flask, render_template, request
import os
import requests
from transformers import pipeline
from functools import lru_cache

app = Flask(__name__)

# 環境変数からTwitterのBearer Tokenを取得
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

@lru_cache(maxsize=1)
def get_classifier():
    return pipeline("text-classification", model="unitary/toxic-bert")

def create_headers(token):
    return {"Authorization": f"Bearer {token}"}

def get_user_id(username, headers):
    url = f"https://api.twitter.com/2/users/by/username/{username}"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()['data']['id']

def get_latest_tweets(user_id, headers, max_results=10):
    url = f"https://api.twitter.com/2/users/{user_id}/tweets"
    params = {
        "max_results": max_results,
        "tweet.fields": "created_at,text"
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    tweets = response.json().get('data', [])
    return [tweet["text"] for tweet in tweets]

def analyze_tweets(tweets):
    classifier = get_classifier()
    results = []
    total_score = 0
    for tweet in tweets:
        output = classifier(tweet)[0]
        score = output['score'] if output['label'] == 'TOXIC' else 1 - output['score']
        total_score += score
    average_score = total_score / len(tweets) if tweets else 0.0
    return average_score

def score_to_rgb(score):
    r = int(score * 255)
    g = int((1 - score) * 255)
    b = int(100 + (1 - abs(score - 0.5)) * 100)
    return f"rgb({r}, {g}, {b})"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        username = request.form["username"].strip()
        headers = create_headers(BEARER_TOKEN)
        try:
            user_id = get_user_id(username, headers)
            tweets = get_latest_tweets(user_id, headers)
            average_score = analyze_tweets(tweets)
            background_color = score_to_rgb(average_score)
            return render_template("result.html", username=username, score=average_score, color=background_color)
        except Exception as e:
            return f"エラーが発生しました: {e}"
    return render_template("form.html")

if __name__ == "__main__":
    # ローカル実行用（Renderでは使わないけど、開発では必要）
    app.run(host="0.0.0.0", port=10000, debug=False)