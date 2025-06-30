from flask import Flask, render_template, request, jsonify
import requests
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import spacy
import os
import traceback
from sklearn.metrics import silhouette_score, davies_bouldin_score, ndcg_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

app = Flask(__name__)

# Hugging Face API Setup
HF_TOKEN = ""
API_URL = ""
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

TXT_FILE = "t.txt"
DATASET_PATH = "E:/DS_GENRE.csv"

# Load dataset
df = pd.read_csv(DATASET_PATH)
genre_columns = df.columns[2:-2]  # Exclude 'movie_name', 'description', and 'Cluster'

# Scaling dataset
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df[genre_columns])
df_scaled = pd.DataFrame(X_scaled, columns=genre_columns)

# KMeans Clustering
def apply_kmeans(X_scaled, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    return kmeans

# Apply KMeans and get cluster assignments
kmeans = apply_kmeans(X_scaled, n_clusters=5)  # You can tune the number of clusters (n_clusters)

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Stopword-like genre/generic terms
GENERIC_WORDS = {"movie", "film", "cinema", "show", "series", "story"}
GENRE_WORDS = {"thriller", "action", "comedy", "drama", "horror", "romance", "sci-fi", "fantasy", "adventure"}

def read_metrics():
    if not os.path.exists(TXT_FILE):
        return 0, 0.0, 0.0, 0.0
    with open(TXT_FILE, "r") as file:
        lines = file.readlines()
    if not lines:
        return 0, 0.0, 0.0, 0.0
    last_line = lines[-1].strip()
    parts = last_line.split(",")
    if len(parts) != 4:
        return 0, 0.0, 0.0, 0.0
    return int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])

def update_metrics(new_silhouette, new_davies_bouldin, new_ndgc):
    current_run, _, _, _ = read_metrics()
    current_run += 1
    with open(TXT_FILE, "a") as f:
        f.write(f"{current_run}, {new_silhouette}, {new_davies_bouldin}, {new_ndgc}\n")

def classify_genres(description):
    half = len(genre_columns) // 2

    def call_api(genres):
        payload = {"inputs": description, "parameters": {"candidate_labels": genres, "multi_label": True}}
        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload))
        return response.json() if response.status_code == 200 else {}

    result_1 = call_api(list(genre_columns[:half]))
    result_2 = call_api(list(genre_columns[half:]))

    genre_scores = {}
    for result in (result_1, result_2):
        if isinstance(result, dict) and "labels" in result and "scores" in result:
            genre_scores.update({label: round(score * 100, 2) for label, score in zip(result["labels"], result["scores"])})

    return genre_scores

def extract_keywords(description):
    doc = nlp(description)
    keywords, current_phrase = [], []

    for token in doc:
        if token.pos_ in {"ADJ", "NOUN", "PROPN"} and not token.is_stop:
            if token.text.lower() not in GENRE_WORDS:
                current_phrase.append(token.text)
        else:
            if current_phrase:
                keywords.append(" ".join(current_phrase))
                current_phrase = []

    if current_phrase:
        keywords.append(" ".join(current_phrase))

    return list(set(keywords))

def find_top_keyword_movies(keywords):
    if not keywords:
        return []
    keywords = [kw.lower() for kw in keywords]
    return [row["movie_name"] for _, row in df.iterrows()
            if any(kw in str(row["description"]).lower() or kw in row["movie_name"].lower() for kw in keywords)]

def find_closest_movie(genre_percentages, filtered_movies=None):
    filtered_df = df[df["movie_name"].isin(filtered_movies)] if filtered_movies else df
    if filtered_df.empty:
        return "No matching movies found", float("inf")

    input_vector = [genre_percentages.get(genre, 0) for genre in genre_columns]
    input_df = pd.DataFrame([input_vector], columns=genre_columns)
    input_scaled = scaler.transform(input_df)

    X_filtered = scaler.transform(filtered_df[genre_columns])
    nbrs = NearestNeighbors(n_neighbors=min(5, len(filtered_df)), metric='cosine')
    nbrs.fit(X_filtered)
    dist, idx = nbrs.kneighbors(input_scaled)

    best_match = filtered_df.iloc[idx[0][0]]
    best_distance = dist[0][0]

    return (best_match["movie_name"], best_distance) if best_distance <= 0.6 else ("No strong match found", best_distance)

def compute_ndcg(input_genre_scores):
    input_vector = np.array([[input_genre_scores.get(g, 0) for g in genre_columns]])
    input_df = pd.DataFrame(input_vector, columns=genre_columns)
    input_vector_scaled = scaler.transform(input_df)
    
    similarities = cosine_similarity(input_vector_scaled, X_scaled)[0]

    # Normalize similarities to range [0, 1]
    similarities = (similarities - similarities.min()) / (similarities.max() - similarities.min() + 1e-8)
    ideal = sorted(similarities, reverse=True)

    return ndcg_score([ideal], [similarities])

@app.route("/")
def home():
    return render_template("index.html", genre_predictions={}, top_movies=[], filtered_movies=[], keywords=[])

@app.route("/predict", methods=["POST"])
def predict():
    try:
        description = request.form.get("description", "").strip()
        if not description:
            return jsonify({"error": "No description provided"}), 400

        genre_percentages = classify_genres(description)
        if not genre_percentages:
            return jsonify({"error": "Failed to classify genres"}), 500

        keywords = extract_keywords(description)
        top_keyword_movies = find_top_keyword_movies(keywords)
        closest_movie, distance = find_closest_movie(genre_percentages, top_keyword_movies) if top_keyword_movies else find_closest_movie(genre_percentages)

        genre_matrix = df[genre_columns].values
        silhouette = silhouette_score(genre_matrix, df["Cluster"])
        davies_bouldin = davies_bouldin_score(genre_matrix, df["Cluster"])
        ndgc = compute_ndcg(genre_percentages)

        update_metrics(silhouette, davies_bouldin, ndgc)

        return jsonify({
            "entered_description": description,
            "genre_scores": genre_percentages,
            "closest_movie": closest_movie,
            "distance": round(distance, 2),
            "keywords": keywords,
            "top_keyword_movies": top_keyword_movies,
            "silhouette_score": round(silhouette, 4),
            "davies_bouldin_index": round(davies_bouldin, 4),
            "ndgc_score": round(ndgc, 4)
        })

    except Exception as e:
        print("Error Traceback:\n", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
