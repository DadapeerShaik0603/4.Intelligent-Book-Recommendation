import streamlit as st
import joblib
import pandas as pd
import numpy as np
from PIL import Image

# Load models and data
vectorizer = joblib.load("vectorizer.pkl")
similarity_matrix = joblib.load("similarity_matrix.pkl")
kmeans = joblib.load("kmeans_model.pkl")
processed_books = pd.read_csv("processed_books.csv")

# Ensure the 'cluster' column exists in the processed_books dataframe
if 'cluster' not in processed_books.columns:
    processed_books['cluster'] = kmeans.predict(vectorizer.transform(processed_books['Book Name']))

def recommend_content_based(book_title, min_rating=0, num_recommendations=5):
    idx = processed_books[processed_books['Book Name'] == book_title].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    filtered_books = processed_books.iloc[[i[0] for i in sim_scores]][['Book Name', 'Author', 'Rating']]
    return filtered_books[filtered_books['Rating'] >= min_rating].head(num_recommendations)

def recommend_clustering(book_title, min_rating=0, num_recommendations=5):
    idx = processed_books[processed_books['Book Name'] == book_title].index[0]
    cluster_label = processed_books.iloc[idx]['cluster']
    cluster_books = processed_books[processed_books['cluster'] == cluster_label]
    return cluster_books[cluster_books['Rating'] >= min_rating].sample(min(num_recommendations, len(cluster_books)))[['Book Name', 'Author', 'Rating']]

def recommend_hybrid(book_title, min_rating=0, num_recommendations=5):
    content_recs = recommend_content_based(book_title, min_rating, num_recommendations * 2)
    clustering_recs = recommend_clustering(book_title, min_rating, num_recommendations * 2)
    hybrid_recs = pd.concat([content_recs, clustering_recs]).drop_duplicates().nlargest(num_recommendations, 'Rating')
    return hybrid_recs

# Streamlit UI
st.title("📚 Book Recommendation System")
st.sidebar.image("a-bookshelf-with-many-books-ai-generated-photo.jpg", use_container_width=True)

st.sidebar.subheader("Select Recommendation Model")
model_option = st.sidebar.radio("Choose a model:", ["Content-Based", "Clustering-Based",  "Hybrid Model"])

min_rating = st.sidebar.slider("Select Minimum Rating", 0.0, 5.0, 3.0, 0.1)

if model_option == "Content-Based":
    book_choice = st.selectbox("Select a book for recommendations:", processed_books['Book Name'].unique())
    if st.button("Get Recommendations"):
        results = recommend_content_based(book_choice, min_rating)
        for _, row in results.iterrows():
            st.write(f"**{row['Book Name']}** by {row['Author']} (Rating: {row['Rating']})")

elif model_option == "Clustering-Based":
    book_choice = st.selectbox("Select a book for recommendations:", processed_books['Book Name'].unique())
    if st.button("Get Recommendations"):
        results = recommend_clustering(book_choice, min_rating)
        for _, row in results.iterrows():
            st.write(f"**{row['Book Name']}** by {row['Author']} (Rating: {row['Rating']})")


elif model_option == "Hybrid Model":
    book_choice = st.selectbox("Select a book for recommendations:", processed_books['Book Name'].unique())
    if st.button("Get Hybrid Recommendations"):
        results = recommend_hybrid(book_choice, min_rating)
        for _, row in results.iterrows():
            st.write(f"**{row['Book Name']}** by {row['Author']} (Rating: {row['Rating']})")

st.sidebar.info("Select a model and input your preferences to receive book recommendations!")
