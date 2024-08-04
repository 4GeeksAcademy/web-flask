from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar el modelo
model = joblib.load('src/model.pkl')  # o 'src/model.sav'

# Ruta principal
@app.route('/')
def home():
    return render_template('index.html')

# Ruta para obtener recomendaciones
@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.json['input']
    # Preprocesar el input si es necesario
    # Obtener la recomendación del modelo
    recommendation = model.predict([user_input])
    return jsonify({'recommendation': recommendation.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
