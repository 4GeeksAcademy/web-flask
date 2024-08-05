from flask import Flask, request, jsonify
import pickle
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Rutas para cargar el modelo y el vectorizador
MODEL_PATH = 'src/models/model.pkl'
VECTORIZER_PATH = 'src/models/tfidf_vectorizer.pkl'
DATA_PATH = 'src/data/peliculas.csv'

def load_model():
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        model = None
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        model = None
    finally:
        return model

def load_vectorizer():
    try:
        vectorizer = joblib.load(VECTORIZER_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        vectorizer = None
    except Exception as e:
        print(f"Error al cargar el vectorizador: {e}")
        vectorizer = None
    finally:
        return vectorizer

# Cargar el modelo y el vectorizador al iniciar la aplicación
model = load_model()
vectorizer = load_vectorizer()
data = None
try:
    data = pd.read_csv(DATA_PATH)
except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Error al cargar los datos: {e}")

@app.route('/')
def home():
    return "Bienvenido a la aplicación de recomendaciones de películas."

@app.route('/recommend', methods=['GET'])
def recommend():
    title = request.args.get('title')
    if not title:
        return jsonify({"error": "Se requiere el título de la película"}), 400

    if model is None or vectorizer is None:
        return jsonify({"error": "Modelo o vectorizador no disponibles"}), 500

    try:
        # Convertir el título a una representación TF-IDF
        title_vector = vectorizer.transform([title])
        # Obtener recomendaciones
        recommendations = model.predict(title_vector)
        return jsonify({"recommendations": recommendations.tolist()})
    except Exception as e:
        print(f"Error al procesar la recomendación: {e}")
        return jsonify({"error": "Error al procesar la recomendación"}), 500

if __name__ == "__main__":
    app.run(debug=True)
