from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Cargar el modelo
try:
    model = joblib.load('src/models/model.pkl')
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")

# Cargar el vectorizador
try:
    tfidf_vectorizer = joblib.load('src/models/tfidf_vectorizer.pkl')
    print("Vectorizador cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el vectorizador: {e}")

# Cargar los datos
try:
    peliculas_df = pd.read_csv('src/data/peliculas.csv')
    print("Datos cargados exitosamente.")
except Exception as e:
    print(f"Error al cargar los datos: {e}")

@app.route('/')
def home():
    return '''
        <!doctype html>
        <html lang="es">
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
            <title>Recomendaciones de Películas</title>
          </head>
          <body>
            <div class="container">
              <h1>Bienvenido a la aplicación de recomendaciones de películas</h1>
              <form action="/recommend" method="get">
                <div class="form-group">
                  <label for="title">Título de la película</label>
                  <input type="text" class="form-control" id="title" name="title" required>
                </div>
                <button type="submit" class="btn btn-primary">Obtener Recomendaciones</button>
              </form>
              {% if recommendations %}
                <h2>Recomendaciones:</h2>
                <ul>
                  {% for recommendation in recommendations %}
                    <li>{{ recommendation }}</li>
                  {% endfor %}
                </ul>
              {% endif %}
            </div>
          </body>
        </html>
    '''

@app.route('/recommend', methods=['GET'])
def recommend():
    try:
        title = request.args.get('title')
        if not title:
            return jsonify({"error": "Falta el parámetro 'title'"}), 400

        title_vector = tfidf_vectorizer.transform([title])
        distances, indices = model.kneighbors(title_vector, n_neighbors=10)

        recommended_titles = [peliculas_df.iloc[idx]['title'] for idx in indices.flatten()]

        return render_template('recommend.html', recommendations=recommended_titles)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=8000)
