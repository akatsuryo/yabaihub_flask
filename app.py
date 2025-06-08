
from flask import Flask, render_template, request, jsonify
import random
import numpy as np
import json
import openai
import os
from dotenv import load_dotenv


load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



app = Flask(__name__)

DINNERS = [
    "カレー", "ラーメン", "パスタ", "焼き魚", "鍋", "うどん", "寿司", "ハンバーグ", "オムライス", "サラダチキン"
]

# コサイン類似度計算関数
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/search", methods=["GET", "POST"])
def search():
    results = []
    query = ""
    if request.method == "POST":
        query = request.form["query"]

        # 入力文のEmbedding取得
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = response.data[0].embedding

        # 保存済みの機能一覧と比較
        with open("functions_with_embeddings.json", "r", encoding="utf-8") as f:
            functions = json.load(f)

        # 類似度スコア付きでソート
        scored = []
        for func in functions:
            score = cosine_similarity(query_embedding, func["embedding"])
            scored.append((score, func))
        scored.sort(reverse=True, key=lambda x: x[0])

        # スコア上位3件を返す（しきい値0.75程度も可）
        results = [item[1] for item in scored[:3]]

    return render_template("search.html", results=results, query=query)

@app.route("/dinner")
def dinner():
    return render_template("dinner.html")  # ← dinner.html を使っていること


@app.route("/random_dinner", methods=["GET"])
def random_dinner():
    result = random.choice(DINNERS)
    return jsonify({"result": result})

@app.route("/sign")
def sign():
    return render_template("sign.html")

@app.route("/generate_sign", methods=["GET"])
def generate_sign():
    name = request.args.get("name", "")
    fancy_map = str.maketrans({
        'A': '𝓐', 'B': '𝓑', 'C': '𝓒', 'D': '𝓓',
        'E': '𝓔', 'F': '𝓕', 'G': '𝓖', 'H': '𝓗',
        'I': '𝓘', 'J': '𝓙', 'K': '𝓚', 'L': '𝓛',
        'M': '𝓜', 'N': '𝓝', 'O': '𝓞', 'P': '𝓟',
        'Q': '𝓠', 'R': '𝓡', 'S': '𝓢', 'T': '𝓣',
        'U': '𝓤', 'V': '𝓥', 'W': '𝓦', 'X': '𝓧',
        'Y': '𝓨', 'Z': '𝓩',
        'a': '𝓪', 'b': '𝓫', 'c': '𝓬', 'd': '𝓭',
        'e': '𝓮', 'f': '𝓯', 'g': '𝓰', 'h': '𝓱',
        'i': '𝓲', 'j': '𝓳', 'k': '𝓴', 'l': '𝓵',
        'm': '𝓶', 'n': '𝓷', 'o': '𝓸', 'p': '𝓹',
        'q': '𝓺', 'r': '𝓻', 's': '𝓼', 't': '𝓽',
        'u': '𝓾', 'v': '𝓿', 'w': '𝔀', 'x': '𝔁',
        'y': '𝔂', 'z': '𝔃'
    })
    signature = name.translate(fancy_map)
    return jsonify({"signature": signature})


if __name__ == "__main__":
    app.run(debug=True)
