from flask import Flask, request, render_template
import pandas as pd
from waitress import serve

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data', methods=['POST'])
def data():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            df = pd.read_csv(file)
            data_html = df.to_html()
            return render_template('index.html', data=data_html)
    return render_template('index.html')

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)
