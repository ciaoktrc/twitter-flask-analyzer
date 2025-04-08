import os
import time
import requests
from flask import Flask, render_template, request
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

BEARER_TOKEN = os.getenv("BEARER_TOKEN")
classifier = pipeline("text-classification", model="unitary/toxic-bert")

headers = {
    "Authorization": f"Bearer {BEARER_TOKEN}"
}

last_username = None
last_access_time = 0

@app.route("/", methods=["GET", "POST"])
def index():
    global last_username, last_access_time

    if request.method == "POST":
        username = request.form["username"].strip().lower()

        # 同一ユーザーの短期連続リクエスト防止（5秒以内）
        current_time = time.time()
        if username == last_username and current_time - last_access_time < 5:
            return render_template("error.html", message="短時間に同じユーザーへの再アクセスはできません。少し時間を空けてください。")

        # ユーザーIDを取得
        user_url = f"https://api.twitter.com/2/users/by/username/{username}"
        user_response = requests.get(user_url, headers=headers)

        # Rate Limit エラーハンドリング（ユーザー取得）
        if user_response.status_code == 429:
            return render_template("error.html", message="Twitter APIの利用制限がかかっています。しばらく時間を空けてからお試しください。")

        if user_response.status_code != 200:
            return render_template("error.html", message="ユーザーが見つからないか、Twitterとの通信に問題が発生しました。")

        user_id = user_response.json()["data"]["id"]


        # スリープで間引き
        time.sleep(2)

        # ツイート取得
        tweet_url = f"https://api.twitter.com/2/users/{user_id}/tweets"
        params = {
            "max_results": 5,
            "tweet.fields": "created_at,text"
        }
        tweet_response = requests.get(tweet_url, headers=headers, params=params)

        # Rate Limitの残数を確認
        remaining = int(tweet_response.headers.get("x-rate-limit-remaining", "1"))
        if remaining < 3:
            return render_template("error.html", message="Twitter APIの利用制限が近づいています。しばらく時間を置いて再度お試しください。")

        if tweet_response.status_code != 200:
            return render_template("error.html", message=f"エラーが発生しました: {tweet_response.status_code}")

        tweets = tweet_response.json().get("data", [])
        if not tweets:
            return render_template("error.html", message="投稿が見つかりませんでした。")

        scores = []
        for tweet in tweets:
            text = tweet["text"]
            result = classifier(text)[0]
            score = result["score"] if result["label"] == "TOXIC" else 1 - result["score"]
            scores.append(score)

        average_score = sum(scores) / len(scores)

        last_username = username
        last_access_time = current_time

        return render_template("result.html", scores=scores, score=average_score, username=username)

    return render_template("form.html")
