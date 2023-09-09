from flask import Flask, render_template, request
import numpy as np
import joblib
import os

model_path = os.path.join(os.path.dirname(__file__), 'models', 'modelo_taller_arbol.pkl')
model = joblib.load(model_path)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    Nitrogeno = int(request.form['Nitrogeno'])
    Fosforo = int(request.form['Fosforo'])
    Potasio = int(request.form['Potasio'])
    Temperatura = int(request.form['Temperatura'])
    Humedad = int(request.form['Humedad'])
    Ph = int(request.form['Ph'])
    Precipitacion = int(request.form['Precipitacion'])

    new_samples = np.array([[Nitrogeno, Fosforo, Potasio, Temperatura, Humedad, Ph, Precipitacion]])

    prediction = model.predict(new_samples)

    mensaje = "La predicción agricola es: "
    mensaje += prediction[0]

    return render_template('result.html', pred = mensaje)


if __name__ == '__main__':
    app.run()