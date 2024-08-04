import joblib
import pandas as pd

# Carga el modelo y el vectorizador
model = joblib.load('src/models/model.pkl')
tfidf_vectorizer = joblib.load('src/models/tfidf_vectorizer.pkl')

# Carga los datos
total_data = pd.read_csv('src/data/peliculas.csv')

def recommend(movie):
    if movie not in total_data["title"].values:
        return "Movie not found in the dataset."
    
    movie_index = total_data[total_data["title"] == movie].index[0]
    movie_tags = total_data["tags"].iloc[movie_index]
    movie_vector = tfidf_vectorizer.transform([movie_tags])
    
    distances, indices = model.kneighbors(movie_vector)
    
    similar_movies = []
    for i in range(1, len(distances[0])):
        similar_movies.append((total_data["title"].iloc[indices[0][i]], distances[0][i]))
    
    if not similar_movies:
        return "No similar movies found."
    
    return similar_movies
