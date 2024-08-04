from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Cargar el modelo y la base de datos
model = joblib.load('src/models/model.pkl')
total_data = pd.read_csv('src/data/peliculas.csv')

def recommend(movie):
    if movie not in total_data["title"].values:
        return "Movie not found in the dataset."
    
    movie_index = total_data[total_data["title"] == movie].index[0]
    distances, indices = model.kneighbors(tfidf_matrix[movie_index])
    
    similar_movies = []
    for i in range(1, len(distances[0])):
        similar_movies.append((total_data["title"].iloc[indices[0][i]], distances[0][i]))
    
    if not similar_movies:
        return "No similar movies found."
    
    return similar_movies

@app.route('/recommend', methods=['GET'])
def recommend_movies():
    movie_title = request.args.get('title')
    
    if not movie_title:
        return jsonify({"error": "No movie title provided"}), 400
    
    recommendations = recommend(movie_title)
    
    if isinstance(recommendations, str):  # Handle error messages
        return jsonify({"error": recommendations}), 404
    
    return jsonify({"recommendations": recommendations})

if __name__ == '__main__':
    app.run(debug=True)
