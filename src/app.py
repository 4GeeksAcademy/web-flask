from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Cargar el modelo y los datos
model = joblib.load('src/models/model.pkl')
total_data = pd.read_csv('src/data/peliculas.csv')
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(total_data["tags"])

@app.route('/recommend', methods=['POST'])
def recommend():
    movie = request.form.get('movie')
    if not movie:
        return jsonify({"error": "No se ha proporcionado el título de la película."}), 400

    try:
        movie_index = total_data[total_data["title"] == movie].index[0]
        distances, indices = model.kneighbors(tfidf_matrix[movie_index])
        similar_movies = [(total_data["title"][i], distances[0][j]) for j, i in enumerate(indices[0])]
        recommendations = similar_movies[1:]
        return jsonify({"recommendations": recommendations})
    except IndexError:
        return jsonify({"error": "Película no encontrada."}), 404

if __name__ == '__main__':
    app.run(debug=True)