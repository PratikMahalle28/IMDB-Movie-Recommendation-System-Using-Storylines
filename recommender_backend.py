# recommender_backend.py (NLTK-FREE version)

import pandas as pd
import numpy as np
import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Simple stopwords (no NLTK needed)
STOPWORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not',
    'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from',
    'they', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their'
}

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

class StorylineRecommender:
    def __init__(self, csv_path="imdb_2024_movies_storylines_5000.csv"):
        self.csv_path = csv_path
        self.df = None
        self.vectorizer = None
        self.tfidf_matrix = None

    def load_and_prepare(self):
        df = pd.read_csv(self.csv_path)

        if "Movie Name" not in df.columns or "Storyline" not in df.columns:
            raise ValueError("CSV must have columns 'Movie Name' and 'Storyline'.")

        df = df.dropna(subset=["Storyline"])
        df = df.drop_duplicates(subset=["Movie Name"])

        df["clean_story"] = df["Storyline"].apply(clean_text)
        df = df[df["clean_story"].str.strip() != ""]

        if df.empty:
            raise ValueError("No valid storylines after cleaning. Check CSV content.")

        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(df["clean_story"])
        self.df = df.reset_index(drop=True)

    def recommend_from_story(self, input_story, top_n=5):
        if self.df is None or self.vectorizer is None:
            raise ValueError("Model not loaded. Call load_and_prepare() first.")

        cleaned_input = clean_text(input_story)
        if not cleaned_input.strip():
            return []

        input_vec = self.vectorizer.transform([cleaned_input])
        sims = cosine_similarity(input_vec, self.tfidf_matrix)[0]

        top_indices = np.argsort(sims)[::-1][:top_n]

        results = []
        for idx in top_indices:
            results.append({
                "Movie Name": self.df.loc[idx, "Movie Name"],
                "Storyline": self.df.loc[idx, "Storyline"],
                "Similarity": float(sims[idx])
            })
        return results

if __name__ == "__main__":
    rec = StorylineRecommender("imdb_2024_movies_storylines_5000.csv")
    rec.load_and_prepare()
    test_story = "A young wizard begins his journey at a magical school where he makes friends and enemies, facing dark forces along the way."
    recs = rec.recommend_from_story(test_story, top_n=5)
    for r in recs:
        print(r["Movie Name"], "->", round(r["Similarity"], 3))
