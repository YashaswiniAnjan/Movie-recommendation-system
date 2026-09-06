import streamlit as st
from recommendation import recommend_movies, get_tmdb_info

st.title("🎬 Movie Recommendation System")
st.write("Find movies similar to a movie you like.")

movie_name = st.text_input(
    "🔎 Search for a movie",
    placeholder="e.g., Star Wars"
)

if st.button("🎯 Recommend Movies"):
    if movie_name:

        recommendations = recommend_movies(movie_name)

        if recommendations.empty:
            st.warning("Movie not found. Try another movie.")
        else:
            st.subheader(f"Movies similar to {movie_name}")

            for _, movie in recommendations.iterrows():

                info = get_tmdb_info(movie["title"])

                if info:
                    if info["poster_url"]:
                        st.image(info["poster_url"], width=180)

                    st.subheader(info["title"])
                    st.write(f"⭐ TMDB Rating: {info['rating']}")
                    st.write(info["overview"])
                    st.divider()

    else:
        st.warning("Please enter a movie name.")