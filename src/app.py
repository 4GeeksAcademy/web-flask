from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Cargar el modelo
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file:
        df = pd.read_csv(file)
        predictions = model.predict(df)
        df['Predictions'] = predictions
        data_html = df.to_html()
        return render_template('index.html', data=data_html)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)

