from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Cargar el modelo KNN
model = joblib.load('C:\Users\Fukushima\Documents\GitHub\web-flask\src\models\model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json['input']
    # Procesar la entrada y obtener la recomendación
    recommendation = model.predict([data])[0]
    return jsonify({'recommendation': recommendation})

if __name__ == "__main__":
    app.run(debug=True)

