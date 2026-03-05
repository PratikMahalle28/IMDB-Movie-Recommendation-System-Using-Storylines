# app.py

import streamlit as st
from recommender_backend import StorylineRecommender

st.set_page_config(
    page_title="IMDb 2024 Storyline Recommender",
    page_icon="🎬",
    layout="wide"
)

@st.cache_resource
def load_recommender():
    rec = StorylineRecommender("imdb_2024_movies_storylines_5000.csv")
    rec.load_and_prepare()
    return rec

def main():
    st.title("IMDb 2024 Movie Recommendation System (Storyline-based)")
    st.write(
        "Enter a **storyline** and get similar 2024 movies based on their plot summaries."
    )

    recommender = load_recommender()

    user_story = st.text_area(
        "Enter movie storyline / plot description:",
        height=200,
        placeholder="Example: A young wizard begins his journey at a magical school where he makes friends and enemies, facing dark forces along the way."
    )

    top_n = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)

    if st.button("Get Recommendations"):
        if not user_story.strip():
            st.warning("Please enter a storyline.")
        else:
            with st.spinner("Finding similar movies..."):
                results = recommender.recommend_from_story(user_story, top_n=top_n)

            if len(results) == 0:
                st.info("No recommendations found. Try a different storyline.")
            else:
                st.subheader("Recommended Movies")
                for i, r in enumerate(results, start=1):
                    st.markdown(f"### {i}. {r['Movie Name']}")
                    st.markdown(f"**Similarity score:** {r['Similarity']:.3f}")
                    st.write(r["Storyline"])
                    st.markdown("---")

if __name__ == "__main__":
    main()
