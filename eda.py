import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load datasets
audiob = pd.read_csv("Cleaned_audiob.csv")
audiob_adv = pd.read_csv("Cleaned_audiob_adv.csv")

# Merge datasets
df = audiob_adv.copy()

# Streamlit App Title
st.title("📊 Audible Catalog - Exploratory Data Analysis (EDA)")

# Display dataset preview
if st.checkbox("Show raw data"):
    st.write(df.head())

# Most Popular Authors
st.subheader("Most Popular Authors")
if 'Author' in audiob_adv.columns:
    author_counts = audiob_adv['Author'].value_counts().head(20)
    st.bar_chart(author_counts)

# Most Expensive Books by Different Authors
st.subheader("Most Expensive Books by Different Authors")
if 'Author' in audiob_adv.columns and 'Price' in audiob_adv.columns:
    author_price = audiob_adv.groupby('Author')['Price'].max().sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.barplot(x=author_price.values, y=author_price.index, ax=ax)
    ax.set_title('Most Expensive Book by Author')
    ax.set_ylabel('Author')
    ax.set_xlabel('')
    st.pyplot(fig)

# Highest Rated Book by Author
st.subheader("Highest Rated Book by Author")
if 'Author' in audiob_adv.columns and 'Rating' in audiob_adv.columns:
    author_rating = audiob_adv.groupby('Author')['Rating'].max().sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.barplot(x=author_rating.values, y=author_rating.index, ax=ax)
    ax.set_title('Highest Rated Book by Author')
    ax.set_ylabel('Author')
    ax.set_xlabel('')
    st.pyplot(fig)

# Lowest Rated Book by Author
st.subheader("Lowest Rated Book by Author")
if 'Author' in audiob_adv.columns and 'Rating' in audiob_adv.columns:
    author_rating = audiob_adv.groupby('Author')['Rating'].max().sort_values(ascending=True).head(20)
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.barplot(x=author_rating.values, y=author_rating.index, ax=ax)
    ax.set_title('Lowest Rated Book by Author')
    ax.set_ylabel('Author')
    ax.set_xlabel('')
    st.pyplot(fig)

# Shortest Book by Author (in minutes)
st.subheader("Shortest Book by Author (in minutes)")
trial = audiob_adv[~(audiob_adv['Time'] < 20)]
if 'Author' in trial.columns and 'Time' in trial.columns:
    author_time = trial.groupby('Author')['Time'].max().sort_values(ascending=True).head(20)
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.barplot(x=author_time.values, y=author_time.index, ax=ax)
    ax.set_title('Shortest Book by Author (in minutes)')
    ax.set_ylabel('Author')
    ax.set_xlabel('')
    st.pyplot(fig)

# Distribution of Ratings
st.subheader("Distribution of Ratings")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['Rating'], bins=20, kde=True, color='blue', ax=ax)
ax.set_xlabel("Rating")
ax.set_ylabel("Count")
st.pyplot(fig)

# Ratings vs. Review Counts
st.subheader("Ratings vs. Review Counts")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=df['Number_of_Reviews'], y=df['Rating'], alpha=0.6, ax=ax)
ax.set_xscale('log')
ax.set_xlabel("Number of Reviews")
ax.set_ylabel("Rating")
st.pyplot(fig)

# Joint Plot of Rating vs. Time
st.subheader("Joint Plot of Rating vs. Time")
if 'Rating' in audiob_adv.columns and 'Time' in audiob_adv.columns:
    fig = sns.jointplot(x="Rating", y="Time", data=audiob_adv, color='crimson')
    fig.set_axis_labels("Rating", "Time")
    st.pyplot(fig)

# Joint Plot of Rating vs. Price
trial = audiob_adv[~(audiob_adv['Price'] > 3000)]
st.subheader("Joint Plot of Rating vs. Price")
if 'Rating' in trial.columns and 'Price' in trial.columns:
    fig = sns.jointplot(x="Rating", y="Price", data=trial, color='crimson')
    fig.set_axis_labels("Rating", "Price")
    st.pyplot(fig)

# Joint Plot of Price vs. Time
st.subheader("Joint Plot of Price vs. Time")
if 'Price' in trial.columns and 'Time' in trial.columns:
    fig = sns.jointplot(x="Price", y="Time", data=trial, color='crimson')
    fig.set_axis_labels("Price", "Time")
    st.pyplot(fig)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
numeric_columns = df.select_dtypes(include='number')
corr_matrix = numeric_columns.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, vmin=-1, vmax=1, annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Correlation Heatmap')
st.pyplot(fig)

# Hidden Gems: Highly Rated but Low Popularity
st.subheader("Hidden Gems: Highly Rated but Low Popularity")
threshold_reviews = df['Number_of_Reviews'].quantile(0.25)
hidden_gems = df[(df['Rating'] >= 4.5) & (df['Number_of_Reviews'] < threshold_reviews)]
st.write(hidden_gems[['Book Name', 'Author', 'Rating', 'Number_of_Reviews']].head(10))


st.write("EDA Completed ✅")
