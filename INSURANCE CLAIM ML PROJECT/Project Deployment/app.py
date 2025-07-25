from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    model = None
    scaler = None
    print("Error: Model or scaler file not found. Please check the path.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        print("Error: Model or scaler not loaded.")
        return render_template('index.html', prediction="Model not loaded.")

    try:
        # Extract input values
        claim_amount = float(request.form.get('claim_amount'))
        accident_severity = float(request.form.get('accident_severity'))
        clmage = int(request.form.get('clmage'))
        clmsex = int(request.form.get('clmsex'))
        clmins = int(request.form.get('clmins'))

        # Create input array and scale it
        input_data = np.array([[claim_amount, accident_severity, clmage, clmsex, clmins]])
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)
        result = "Attorney Involved" if prediction[0] == 1 else "Attorney Not Involved"

        print(f"Model Prediction: {result}")  # Debugging output

    except Exception as e:
        result = f"Error: {str(e)}"
        print(result)  # Print the error in terminal for debugging

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
