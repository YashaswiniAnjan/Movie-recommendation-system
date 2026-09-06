import streamlit as st
from recommendation import (
    recommend_movies,
    get_tmdb_info,
    get_als_recommendations
)

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")
st.write("Find movies similar to a movie you like and get personalized recommendations.")

movie_name = st.text_input(
    "🔎 Search for a movie",
    placeholder="e.g., Toy Story"
)

user_id = st.number_input(
    "👤 Enter User ID",
    min_value=1,
    max_value=943,
    value=1
)

if st.button("🎯 Recommend Movies"):

    if not movie_name:
        st.warning("Please enter a movie name.")

    else:
        recommendations = recommend_movies(movie_name)
        als_recommendations = get_als_recommendations(user_id)

        if recommendations.empty:
            st.warning("Movie not found. Try another movie.")

        else:

            # ==========================================
            # CONTENT-BASED RECOMMENDATIONS
            # ==========================================

            st.header("🎬 Content-Based Recommendations")
            st.write(f"Movies similar to **{movie_name}**")

            content_movies = recommendations.head(5)

            content_cols = st.columns(5)

            for i, (_, movie) in enumerate(content_movies.iterrows()):

                with content_cols[i]:

                    info = get_tmdb_info(movie["title"])

                    if info:

                        if info["poster_url"]:
                            st.image(
                                info["poster_url"],
                                width=160
                            )

                        st.subheader(info["title"])

                        st.write(
                            f"⭐ {info['rating']}"
                        )

                        if info["overview"]:
                            st.write(info["overview"])
                        else:
                            st.write(
                                "No description available."
                            )

            st.divider()

            # ==========================================
            # PYSPARK ALS RECOMMENDATIONS
            # ==========================================

            st.header("🤖 PySpark ALS Personalized Recommendations")
            st.write(f"Personalized for **User {user_id}**")

            als_movies = als_recommendations.head(5)

            als_cols = st.columns(5)

            for i, (_, movie) in enumerate(als_movies.iterrows()):

                with als_cols[i]:

                    info = get_tmdb_info(movie["title"])

                    if info:

                        if info["poster_url"]:
                            st.image(
                                info["poster_url"],
                                width=160
                            )

                        st.subheader(info["title"])

                        st.write(
                            f"⭐ {info['rating']}"
                        )

                        if info["overview"]:
                            st.write(info["overview"])
                        else:
                            st.write(
                                "No description available."
                            )
