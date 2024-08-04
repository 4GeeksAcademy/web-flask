from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Cargar el modelo y los datos
try:
    model = joblib.load('src/models/model.pkl')
    total_data = pd.read_csv('src/data/peliculas.csv')

    # Recalcular la matriz TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(total_data["tags"])
except Exception as e:
    print(f"Error loading model or data: {e}")

# Ruta para la página principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para obtener recomendaciones
@app.route('/recommend', methods=['GET'])
def recommend():
    try:
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

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)