import streamlit as st
import pandas as pd
import requests
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


ratings_data = pd.read_csv('u.data', encoding='latin1', header=0, engine='python',sep='\t', names=["User_Id", "Movie_Id", "Rating", "Timestamp"])
ratings_df = pd.DataFrame(ratings_data)
ratings_df = ratings_df.drop('Timestamp', axis=1)
user_item_matrix = ratings_df.pivot(index='User_Id', columns='Movie_Id', values='Rating')
user_item_matrix.fillna(0, inplace=True)
matrix_sparse_csr = csr_matrix(user_item_matrix)

k = 10

u, s, vt = svds(matrix_sparse_csr, k=k)

final_matrix = np.dot(u, np.dot(np.diag(s), vt))
matrix_sparse_csr = csr_matrix(user_item_matrix)

k = 10

u, s, vt = svds(matrix_sparse_csr, k=k)

final_matrix = np.dot(u, np.dot(np.diag(s), vt))
user_item_matrix_np = user_item_matrix.values
zero_indices = np.where(user_item_matrix_np == 0)

user_item_matrix_updated = user_item_matrix_np.copy()

user_item_matrix_updated[zero_indices] = final_matrix[zero_indices]
num_zeros_original = np.sum(user_item_matrix_np == 0)

user_item_matrix_updated = user_item_matrix_np.copy()

zero_indices = np.where(user_item_matrix_np == 0)

user_item_matrix_updated[zero_indices] = final_matrix[zero_indices]

num_replaced = np.sum(user_item_matrix_np != user_item_matrix_updated)
normalized_array = (user_item_matrix_updated - np.min(user_item_matrix_updated)) / (np.max(user_item_matrix_updated) - np.min(user_item_matrix_updated))
min_range = 1
max_range = 5
scaled_array = min_range + normalized_array * (max_range - min_range)

# Standardize the array
scaled_array = StandardScaler().fit_transform(scaled_array)

# Compute the mean of each item
item_means = scaled_array.mean(axis=0)

# Compute the adjusted scaled array by subtracting the mean from each item
adjusted_scaled_array = scaled_array - item_means

# Compute the adjusted cosine similarity matrix
item_similarity_matrix = cosine_similarity(adjusted_scaled_array.T)


popularity_ratings = np.mean(scaled_array, axis=0)
# print(popularity_ratings)
# print(popularity_ratings.shape)
similarity_weight = 0.5
popularity_weight = 0.5

weighted_similarity_matrix = similarity_weight * item_similarity_matrix + popularity_weight * popularity_ratings[:, np.newaxis]



items_data = pd.read_csv('u.item',encoding='latin1', header=0, engine='python',sep='|', names=["Movie_Id","movie_title","release_date","videorelease_date","IMDb_URL","unknown","Action","Adventure","Animation","Children","Comedy","Crime","Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"])
items_df = pd.DataFrame(items_data)
items_df = items_df.drop(['release_date', 'videorelease_date',  'unknown'], axis=1)
# Define functions for finding similar items and recommending top items
def find_all_similar_items(weighted_similarity_matrix):
    num_items = weighted_similarity_matrix.shape[0]
    all_similar_items_dict = {}

    for item_index in range(num_items):
        similarity_scores = weighted_similarity_matrix[item_index, :]
        similar_item_indices = np.argsort(similarity_scores)[::-1] + 1
        all_similar_items_dict[item_index + 1] = similar_item_indices

    return all_similar_items_dict

def recommend_top_items(all_similar_items_dict, num_recommendations=5):
    top_recommendations_dict = {}

    for item_index, similar_items in all_similar_items_dict.items():
        recommended_items = [item for item in similar_items if item != item_index][:num_recommendations]
        top_recommendations_dict[item_index] = recommended_items

    return top_recommendations_dict



import re
import requests

def fetch_poster_from_tmdb_by_title(movie_title):
    api_key = 'eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJmMjEwZWEzOGJjNDhkZTQyZDQ1YzMzMzUxMGZkODI1MyIsInN1YiI6IjY1ZmQzMTFmMzc4MDYyMDE3ZTg3MWM2ZCIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.ifEhqFnSL7MLwhmIbn7mMN0fwRdD6DmWTwmvkQpZqQQ'
    url = "https://api.themoviedb.org/3/search/movie"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    # Extract title and year from the input string using regular expressions
    match = re.match(r'^(.*?)\s*\((\d{4})\)$', movie_title)
    if match:
        title, year = match.groups()
    else:
        title = movie_title
        year = None
    
    params = {
        "query": title,
        "year": year,  # Add year as a parameter
        "include_adult": False,
        "language": "en-US",
        "page": 1
    }
    print("API Request Params:", params)
    response = requests.get(url, headers=headers, params=params)
    print("API Response:", response.json())
    data = response.json()
    if 'results' in data and data['results']:
        poster_path = data['results'][0].get('poster_path')
        if poster_path:
            base_url = "https://image.tmdb.org/t/p/w500"  # Adjust the size as needed
            poster_url = f"{base_url}{poster_path}"
            return poster_url
    return None

def main():
    # Assuming you have items_df loaded
    selected_movie = st.selectbox("Select a Movie:", items_df['movie_title'].values)

    if st.button('Show Recommendations'):
        # Check if selected_movie is not '-- Select Movie --'
        if selected_movie != '-- Select Movie --':
            # Find movie_id corresponding to the selected movie title
            movie_id_candidates = items_df.loc[items_df['movie_title'] == selected_movie, 'Movie_Id'].values
            if len(movie_id_candidates) > 0:
                # If at least one movie_id is found, use the first one
                movie_id = movie_id_candidates[0]
                all_similar_items_dict = find_all_similar_items(weighted_similarity_matrix)
                recommendations = recommend_top_items(all_similar_items_dict)[movie_id]
                recommended_movie_titles = [items_df.loc[rec - 1, 'movie_title'] for rec in recommendations]
                num_recommendations = len(recommended_movie_titles)
                
                # Define number of columns for arrangement
                num_columns = 5
                # Calculate number of rows required
                num_rows = (num_recommendations + num_columns - 1) // num_columns
                
                # Create columns for arranging posters
                columns = st.columns(num_columns)
                
                for i in range(num_rows):
                    for j in range(num_columns):
                        index = i * num_columns + j
                        if index < num_recommendations:
                            title = recommended_movie_titles[index]
                            try:
                                poster_url = fetch_poster_from_tmdb_by_title(title)
                            except KeyError:
                                poster_url = None

                            if poster_url is not None:
                                with columns[j]:
                                    st.image(poster_url, caption=title, use_column_width=True)
                            else:
                                with columns[j]:
                                    st.write(f"No poster available for {title}")

            else:
                # If no movie_id is found, display a message
                st.write(f"No movie found with the title '{selected_movie}'.")
        else:
            # If '-- Select Movie --' is selected, display a message
            st.write("Please select a movie.")

if __name__ == "__main__":
    main()