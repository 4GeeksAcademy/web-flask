from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Cargar el modelo y los datos
model = joblib.load('src/models/model.pkl')
total_data = pd.read_csv('src/data/peliculas.csv')

# Ruta para la página principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para obtener recomendaciones
@app.route('/recommend', methods=['GET'])
def recommend():
    movie_title = request.args.get('title')
    if not movie_title:
        return jsonify({'error': 'No movie title provided'}), 400

    if movie_title not in total_data["title"].values:
        return jsonify({'error': 'Movie not found in the dataset'}), 404
    
    movie_index = total_data[total_data["title"] == movie_title].index[0]
    distances, indices = model.kneighbors(tfidf_matrix[movie_index])
    
    similar_movies = []
    for i in range(1, len(distances[0])):
        similar_movies.append((total_data["title"].iloc[indices[0][i]], distances[0][i]))
    
    if not similar_movies:
        return jsonify({'error': 'No similar movies found'}), 404
    
    return jsonify({'recommendations': similar_movies})

if __name__ == '__main__':
    app.run(debug=True)
