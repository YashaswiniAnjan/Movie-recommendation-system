import pandas as pd
import requests
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.recommendation import ALS


# -----------------------------
# 1. LOAD MOVIE DATA
# -----------------------------

movies = pd.read_csv("data/movielens_100k.csv")

ratings = pd.read_csv(
    "u.data.txt",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)


# -----------------------------
# 2. CONTENT-BASED RECOMMENDER
# -----------------------------

movies["content"] = (
    movies["title"].fillna("") + " " +
    movies["directors"].fillna("") + " " +
    movies["actors"].fillna("") + " " +
    movies["genres"].fillna("")
)

tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(movies["content"])


def recommend_movies(title, n=5):

    matches = movies[
        movies["title"].str.contains(
            title,
            case=False,
            na=False
        )
    ]

    if matches.empty:
        return movies.iloc[0:0]

    idx = matches.index[0]

    similarity_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    indices = similarity_scores.argsort()[-n-1:-1][::-1]

    return movies.iloc[indices]


# -----------------------------
# 3. PYSPARK ALS
# -----------------------------

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("MovieRecommendation")
    .getOrCreate()
)

ratings_spark = spark.createDataFrame(
    ratings[["user_id", "movie_id", "rating"]]
)

ratings_spark = ratings_spark.select(
    col("user_id").cast("int"),
    col("movie_id").cast("int"),
    col("rating").cast("float")
)


als = ALS(
    userCol="user_id",
    itemCol="movie_id",
    ratingCol="rating",
    rank=10,
    maxIter=10,
    regParam=0.1,
    coldStartStrategy="drop",
    nonnegative=True
)

als_model = als.fit(ratings_spark)

print("ALS MODEL TRAINED SUCCESSFULLY")


# -----------------------------
# 4. ALS RECOMMENDATIONS
# -----------------------------

user_df = (
    ratings_spark
    .filter(col("user_id") == 1)
    .select("user_id")
    .distinct()
)

user_recommendations = als_model.recommendForUserSubset(
    user_df,
    5
)

recommended_ids = [
    r["movie_id"]
    for r in user_recommendations
    .select("recommendations")
    .collect()[0]["recommendations"]
]

recommended_movies = movies[
    movies["movie_id"].isin(recommended_ids)
]


# -----------------------------
# 5. TMDB API
# -----------------------------

TMDB_TOKEN = st.secrets["TMDB_TOKEN"]
headers = {
    "Authorization": f"Bearer {TMDB_TOKEN}",
    "accept": "application/json"
}


def get_tmdb_info(title, year=None):

    response = requests.get(
        "https://api.themoviedb.org/3/search/movie",
        headers=headers,
        params={"query": title}
    )

    if response.status_code != 200:
        return None

    results = response.json().get("results", [])

    if not results:
        return None

    movie = results[0]

    return {
        "title": movie["title"],
        "overview": movie.get("overview", ""),
        "rating": movie.get("vote_average", 0),
        "poster_url": (
            "https://image.tmdb.org/t/p/w500"
            + movie["poster_path"]
            if movie.get("poster_path")
            else None
        )
    }


# -----------------------------
# 6. TMDB DETAILS FOR ALS RESULTS
# -----------------------------

tmdb_recommendations = []

for title in recommended_movies["title"]:

    info = get_tmdb_info(title)

    if info:
        tmdb_recommendations.append(info)
