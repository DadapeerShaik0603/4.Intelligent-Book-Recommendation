import streamlit as st
import joblib
import torch
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset
from PIL import Image

# Load models and data
vectorizer = joblib.load("vectorizer.pkl")
similarity_matrix = joblib.load("similarity_matrix.pkl")
kmeans = joblib.load("kmeans_model.pkl")
svd = joblib.load("svd_model.pkl")
knn = joblib.load("knn_model.pkl")
processed_books = pd.read_csv("processed_books.csv")

# Get top 5 authors and books
top_authors = processed_books['Author'].value_counts().head(5)
top_books = processed_books.nlargest(5, 'Rating')[['Book Name', 'Author', 'Rating']]

def recommend_content_based(book_title, min_rating=0, num_recommendations=5):
    idx = processed_books[processed_books['Book Name'] == book_title].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    filtered_books = processed_books.iloc[[i[0] for i in sim_scores]][['Book Name', 'Author', 'Description', 'Rating']]
    return filtered_books[filtered_books['Rating'] >= min_rating].head(num_recommendations)

def recommend_clustering(book_title, model_type="KMeans", min_rating=0, num_recommendations=5):
    idx = processed_books[processed_books['Book Name'] == book_title].index[0]
    cluster_label = processed_books.iloc[idx]['cluster']
    
    if model_type == "KMeans":
        cluster_books = processed_books[processed_books['cluster'] == cluster_label]
    
    return cluster_books[cluster_books['Rating'] >= min_rating].sample(min(num_recommendations, len(cluster_books)))[['Book Name', 'Author', 'Description', 'Rating']]

def recommend_collaborative(user_id, min_rating=0, num_recommendations=5):
    all_books = processed_books[['Book Name', 'Author', 'Rating']].reset_index()
    predictions = [svd.predict(user_id, i).est for i in range(len(all_books))]
    all_books['Predicted Rating'] = predictions
    top_books = all_books[all_books['Predicted Rating'] >= min_rating].nlargest(num_recommendations, 'Predicted Rating')
    return top_books[['Book Name', 'Author', 'Rating']]

# Streamlit UI
st.title("📚 Book Recommendation System")
st.sidebar.image("a-bookshelf-with-many-books-ai-generated-photo.jpg", use_container_width=True)

st.sidebar.subheader("Top 5 Authors")
for author, count in top_authors.items():
    st.sidebar.write(f"{author} ({count} books)")

st.sidebar.subheader("Top 5 Books")
for _, row in top_books.iterrows():
    st.sidebar.write(f"**{row['Book Name']}** by {row['Author']} (Rating: {row['Rating']})")

st.sidebar.subheader("Select Recommendation Model")
model_option = st.sidebar.radio("Choose a model:", ["Content-Based", "Clustering-Based", "Collaborative Filtering"])

min_rating = st.sidebar.slider("Select Minimum Rating", 0.0, 5.0, 3.0, 0.1)

if model_option == "Content-Based":
    book_choice = st.selectbox("Select a book for recommendations:", processed_books['Book Name'].unique())
    if st.button("Get Recommendations"):
        results = recommend_content_based(book_choice, min_rating)
        for idx, row in results.iterrows():
            st.write(f"**{row['Book Name']}** by {row['Author']} (Rating: {row['Rating']})")
            st.write(f"_Description:_ {row['Description']}")

elif model_option == "Clustering-Based":
    book_choice = st.selectbox("Select a book for recommendations:", processed_books['Book Name'].unique())
    clustering_type = st.selectbox("Choose Clustering Method:", ["KMeans"])
    if st.button("Get Recommendations"):
        results = recommend_clustering(book_choice, clustering_type, min_rating)
        for idx, row in results.iterrows():
            st.write(f"**{row['Book Name']}** by {row['Author']} (Rating: {row['Rating']})")
            st.write(f"_Description:_ {row['Description']}")

elif model_option == "Collaborative Filtering":
    user_id = st.number_input("Enter User ID", min_value=1, step=1)
    if st.button("Get Recommendations"):
        results = recommend_collaborative(user_id, min_rating)
        for idx, row in results.iterrows():
            st.write(f"**{row['Book Name']}** by {row['Author']} (Rating: {row['Rating']})")

st.sidebar.info("Select a model and input your preferences to receive book recommendations!")
