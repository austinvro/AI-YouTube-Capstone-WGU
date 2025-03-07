import os
import random
import re
from collections import Counter
from flask import Flask, request, Response, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import openai

# Set up Flask to serve the React build as static files.
app = Flask(__name__, static_folder="../frontend/build", static_url_path="")
CORS(app)

# ----------------------------------------------------------------
# 1) SET YOUR OPENAI KEY
# For the purpose of this evaluation, an OPENAI key is provided.
# In a real-world scenario, this would be stored as an environment variable.
# ----------------------------------------------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")

# ----------------------------------------------------------------
# Set up relative file paths based on the project structure.
# ----------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "backend",
                          "multi_output_predictor.pkl")
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "youtube.csv")
CLEANED_DATA_PATH = os.path.join(
    PROJECT_ROOT, "dataset", "cleaned_youtube_data.csv")

# ----------------------------------------------------------------
# 2) LOAD THE MULTI-OUTPUT MODEL (e.g., Decision Tree)
# ----------------------------------------------------------------
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# ----------------------------------------------------------------
# 3) READ THE DATASET (WITH REAL YOUTUBE TITLES) FOR GPT TITLE GENERATION
# ----------------------------------------------------------------
try:
    all_titles = pd.read_csv(DATASET_PATH)["title"].dropna().tolist()
except Exception as e:
    print(f"Error loading dataset: {e}")
    all_titles = []

# ----------------------------------------------------------------
# Define stop words for filtering in analytics (expanded list)
# ----------------------------------------------------------------
STOP_WORDS = set([
    "the", "and", "a", "of", "to", "in", "is", "you", "that", "it", "for",
    "on", "with", "as", "this", "are", "was", "but", "be", "at", "by", "an",
    "or", "if", "from", "so", "we", "can", "not", "all", "your", "our",
    "i", "me", "my", "he", "she", "they", "them", "his", "her", "its", "their",
    "what", "which", "who", "whom", "when", "where", "why", "how", "s", "d", "t", "ll",
    "official", "video", "trailer", "episode", "full", "new", "les"
])

# ----------------------------------------------------------------
# Define a mapping from category_id to category name.
# ----------------------------------------------------------------
YOUTUBE_CATEGORY_MAPPING = {
    1: "Film & Animation",
    2: "Autos & Vehicles",
    10: "Music",
    15: "Pets & Animals",
    17: "Sports",
    18: "Short Movies",
    19: "Travel & Events",
    20: "Gaming",
    21: "Videoblogging",
    22: "People & Blogs",
    23: "Comedy",
    24: "Entertainment",
    25: "News & Politics",
    26: "Howto & Style",
    27: "Education",
    28: "Science & Technology",
    29: "Nonprofits & Activism",
    30: "Movies",
    31: "Anime/Animation",
    32: "Action/Adventure",
    33: "Classics",
    34: "Comedy",
    35: "Documentary",
    36: "Drama",
    37: "Family",
    38: "Foreign",
    39: "Horror",
    40: "Sci-Fi/Fantasy",
    41: "Thriller",
    42: "Shorts",
    43: "Shows",
    44: "Trailers"
}

# ----------------------------------------------------------------
# Helper: Find relevant titles (keyword search) for GPT Title Generation
# ----------------------------------------------------------------


def find_relevant_titles(topic: str, max_results=3):
    t_lower = topic.lower()
    matched = []
    for t in all_titles:
        count = t.lower().count(t_lower)
        if count > 0:
            matched.append((t, count))
    matched.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in matched[:max_results]]

# ----------------------------------------------------------------
# GPT-3.5: Title Generation Anchored to Dataset Titles
# ----------------------------------------------------------------


def generate_dataset_based_title(topic: str) -> str:
    examples = find_relevant_titles(topic, max_results=3)
    if not examples:
        examples_text = "No similar titles found in dataset."
    else:
        examples_text = "\n".join(
            f"{i+1}. {ex}" for i, ex in enumerate(examples))
    system_msg = (
        "You are an AI that writes short, catchy YouTube titles. "
        "No disclaimers, no partial text, and no links. Focus on the topic and stay concise."
    )
    user_msg = (
        f"Below are real YouTube titles from my dataset related to the topic:\n{examples_text}\n\n"
        f"Now, write ONLY a short, catchy YouTube title about '{topic}'. "
        "Show the full text of what you generate (no disclaimers)."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.7,
            max_tokens=100
        )
        generated_text = response.choices[0].message.content.strip()
        return generated_text if generated_text else "Error generating title"
    except Exception as e:
        print("Error generating dataset-based title (GPT-3.5):", str(e))
        return "Error generating title"

# ----------------------------------------------------------------
# GPT-3.5: Description Generation
# ----------------------------------------------------------------


def generate_description_gpt35(title: str) -> str:
    system_msg = (
        "You are an AI that writes concise, engaging YouTube video descriptions. "
        "Include relevant hashtags. Keep it appealing and professional."
    )
    user_msg = f"Write a short YouTube description for the video title: '{title}'. Include relevant hashtags."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.7,
            max_tokens=150
        )
        desc_text = response.choices[0].message.content.strip()
        return desc_text if desc_text else "Error generating description"
    except Exception as e:
        print("Error generating description (GPT-3.5):", str(e))
        return "Error generating description"

# ----------------------------------------------------------------
# Simple “Best Posting Time” Recommendation
# ----------------------------------------------------------------


def recommend_posting_time():
    days = ["Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday", "Sunday"]
    times = ["9:00 AM", "12:00 PM", "3:00 PM", "6:00 PM", "8:00 PM"]
    return f"{random.choice(days)} at {random.choice(times)}"

# ----------------------------------------------------------------
# Endpoint: Generate Title
# ----------------------------------------------------------------


@app.route('/generate_title_gpt35', methods=['POST'])
def generate_title_gpt35_endpoint():
    try:
        data = request.get_json()
        if not data or "topic" not in data or not data["topic"].strip():
            return Response("Topic is required", status=400)
        topic = data["topic"].strip()
        result = generate_dataset_based_title(topic)
        return Response(result, mimetype="text/plain", status=200)
    except Exception as e:
        return Response(str(e), mimetype="text/plain", status=400)

# ----------------------------------------------------------------
# Endpoint: Generate Description
# ----------------------------------------------------------------


@app.route('/generate_description_gpt35', methods=['POST'])
def generate_description_endpoint():
    try:
        data = request.get_json()
        if not data or "title" not in data or not data["title"].strip():
            return Response("Title is required", status=400)
        title_text = data["title"].strip()
        desc = generate_description_gpt35(title_text)
        return Response(desc, mimetype="text/plain", status=200)
    except Exception as e:
        return Response(str(e), mimetype="text/plain", status=400)

# ----------------------------------------------------------------
# Endpoint: Predict Video Performance
# ----------------------------------------------------------------


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Empty JSON request"}), 400
        category_id = int(data.get("category_id", 0))
        publish_year = int(data.get("publish_year", 2024))
        publish_month = int(data.get("publish_month", 1))
        publish_day = int(data.get("publish_day", 1))
        input_data = pd.DataFrame([[category_id, publish_year, publish_month, publish_day]],
                                  columns=["category_id", "publish_year", "publish_month", "publish_day"])
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
        predicted_values = model.predict(input_data)[0]
        predicted_views = int(predicted_values[0])
        predicted_engagement_rate = float(predicted_values[1])
        predicted_comment_rate = float(predicted_values[2])
        posting_time = recommend_posting_time()
        return jsonify({
            "predicted_views": predicted_views,
            "predicted_engagement_rate": predicted_engagement_rate,
            "comment_rate": predicted_comment_rate,
            "best_posting_time": posting_time
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ----------------------------------------------------------------
# Endpoint: Analytics
# ----------------------------------------------------------------


@app.route('/analytics', methods=['GET'])
def analytics():
    try:
        # Load the cleaned dataset using the relative path
        df = pd.read_csv(CLEANED_DATA_PATH)

        # Compute trending keywords from titles by counting word frequencies.
        words = []
        for title in df["title"].dropna().tolist():
            tokens = re.findall(r'\w+', title.lower())
            filtered_tokens = [token for token in tokens if token.isalpha(
            ) and token not in STOP_WORDS and len(token) > 2 and token.isascii()]
            words.extend(filtered_tokens)
        word_counts = Counter(words)
        trending_keywords = word_counts.most_common(10)

        # Compute overall engagement metrics: average views, engagement rate, and comment rate.
        df["engagement_rate"] = df["likes"] / df["views"]
        df["comment_rate"] = df["comment_count"] / df["views"]
        avg_views = df["views"].mean()
        avg_engagement_rate = df["engagement_rate"].mean()
        avg_comment_rate = df["comment_rate"].mean()

        # Group dataset by category_id to determine top performing videos per category.
        df["category_id"] = df["category_id"].astype(int)
        grouped = df.groupby("category_id").agg({
            "views": "max",
            "engagement_rate": "max",
            "comment_count": "max"
        }).reset_index()

        ranking_by_views = grouped.sort_values(
            "views", ascending=False).to_dict(orient="records")
        ranking_by_engagement = grouped.sort_values(
            "engagement_rate", ascending=False).to_dict(orient="records")
        ranking_by_comments = grouped.sort_values(
            "comment_count", ascending=False).to_dict(orient="records")

        def add_category_name(record):
            cid = int(record["category_id"])
            record["category_name"] = YOUTUBE_CATEGORY_MAPPING.get(
                cid, "Unknown")
            return record

        ranking_by_views = [add_category_name(r) for r in ranking_by_views]
        ranking_by_engagement = [add_category_name(
            r) for r in ranking_by_engagement]
        ranking_by_comments = [add_category_name(
            r) for r in ranking_by_comments]

        analytics_data = {
            "trending_keywords": trending_keywords,
            "average_views": avg_views,
            "average_engagement_rate": avg_engagement_rate,
            "average_comment_rate": avg_comment_rate,
            "ranking_by_views": ranking_by_views,
            "ranking_by_engagement": ranking_by_engagement,
            "ranking_by_comments": ranking_by_comments
        }
        return jsonify(analytics_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------------------------------------------
# Serve React App for all other routes
# ----------------------------------------------------------------


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")


# ----------------------------------------------------------------
# Run the Flask App
# ----------------------------------------------------------------
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
