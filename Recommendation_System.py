import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Movie dataset with movie titles and genres
movies = pd.DataFrame({
    'title': ['The Matrix', 'Titanic', 'Toy Story', 'The Avengers', 'Gladiator'],
    'genre': ['Action|Sci-Fi', 'Romance|Drama', 'Animation|Adventure|Comedy', 'Action|Adventure|Sci-Fi', 'Action|Adventure|Drama']
})

# User ratings dataset
user_ratings = pd.DataFrame({
    'user': ['User1', 'User2', 'User3', 'User1', 'User2', 'User3', 'User1', 'User2', 'User3'],
    'title': ['The Matrix', 'Titanic', 'Toy Story', 'Gladiator', 'The Avengers', 'The Matrix', 'The Avengers', 'Gladiator', 'Titanic'],
    'rating': [5, 4, 3, 4, 5, 4, 5, 4, 2]
})



# Create a TF-IDF vectorizer to convert genres into vectors
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genre'])

# Calculate cosine similarity between movies based on genre
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get recommendations based on movie title
def content_based_recommendations(title, cosine_sim=cosine_sim):
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# Get recommendations based on a movie
print("Content-Based Recommendations for 'The Matrix':")
print(content_based_recommendations('The Matrix'))

#Collaborative Filtering

# Create a user-item matrix
user_item_matrix = user_ratings.pivot_table(index='user', columns='title', values='rating')

# Fill NaN values with 0 (indicating no rating)
user_item_matrix = user_item_matrix.fillna(0)

# Calculate cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix)

# Convert into DataFrame
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Function to get movie recommendations for a user based on similar users
def collaborative_recommendations(user, num_recommendations=3):
    similar_users = user_similarity_df[user].sort_values(ascending=False).index[1:num_recommendations+1]
    similar_user_ratings = user_item_matrix.loc[similar_users]
    avg_ratings = similar_user_ratings.mean(axis=0)
    movies_watched = user_item_matrix.loc[user][user_item_matrix.loc[user] > 0].index
    avg_ratings = avg_ratings.drop(movies_watched)
    return avg_ratings.sort_values(ascending=False).head(num_recommendations)

# Get recommendations for a user
print("Collaborative Filtering Recommendations for User1:")
print(collaborative_recommendations('User1'))
