import os
import logging
import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template, jsonify

# --- Configuration ---
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH = os.path.join(BASE_DIR, 'pipeline.pkl')

# --- Load the Model Pipeline ---
# This model includes the preprocessors, SMOTE logic, and the final LDA model.
try:
    with open(PIPELINE_PATH, 'rb') as file:
        model_pipeline = pickle.load(file)
    print("Model pipeline loaded successfully.")
except FileNotFoundError:
    print("ERROR: pipeline.pkl not found. Please run training.py first.")
    model_pipeline = None

# --- Define Expected Columns ---
# These must match the features used in model training, in the correct order.
INPUT_COLUMNS = [
    'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
    'smoking_status'
]

# --- Flask Routes ---

@app.route('/')
def home():
    """Renders the input form for stroke prediction."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request."""
    if model_pipeline is None:
        err_msg = 'Model not loaded. Please ensure pipeline.pkl is generated.'
        if request.is_json:
            return jsonify({'error': err_msg}), 500
        return render_template('index.html', error=err_msg)

    try:
        # Get data from the form or JSON body
        if request.is_json:
            data = request.get_json() or {}
        else:
            data = request.form.to_dict()

        # --- Data Validation and Formatting ---
        input_data = {}
        input_data['gender'] = data.get('gender')

        try:
            # Handle numerical inputs (use float to support decimal inputs such as 58.5)
            input_data['age'] = float(data.get('age'))
            input_data['avg_glucose_level'] = float(data.get('avg_glucose_level'))

            # Handle BMI, which may be missing in the original data (imputer handles NaN)
            bmi_val = data.get('bmi')
            input_data['bmi'] = float(bmi_val) if (bmi_val is not None and str(bmi_val).strip() != '') else np.nan

            # Handle binary/categorical inputs
            input_data['hypertension'] = int(data.get('hypertension'))
            input_data['heart_disease'] = int(data.get('heart_disease'))
            input_data['ever_married'] = data.get('ever_married')
            input_data['work_type'] = data.get('work_type')
            input_data['Residence_type'] = data.get('Residence_type')
            input_data['smoking_status'] = data.get('smoking_status')

        except (ValueError, TypeError) as num_err:
            app.logger.warning(f"Validation error: {num_err}")
            err_msg = 'Invalid numerical input for age, glucose, or BMI.'
            if request.is_json:
                return jsonify({'error': err_msg}), 400
            return render_template('index.html', error=err_msg)

        # Create DataFrame in the exact order the model expects
        df_predict = pd.DataFrame([input_data])[INPUT_COLUMNS]

        # Use predict_proba to get the probability of stroke (class 1)
        probabilities = model_pipeline.predict_proba(df_predict)
        stroke_prob = probabilities[0][1] * 100  # Probability of class 1 (stroke)

        if stroke_prob >= 50:
            outcome = 'High Risk'
        elif stroke_prob >= 25:
            outcome = 'Medium Risk'
        else:
            outcome = 'Low Risk'

        # Generate response data
        prediction_result = {
            'probability': f"{stroke_prob:.2f}%",
            'outcome': outcome,
            'model_type': 'Linear Discriminant Analysis (LDA) with SMOTE',
            'input_features': df_predict.iloc[0].to_dict()
        }

        if request.is_json:
            return jsonify(prediction_result)

        return render_template('index.html', prediction=prediction_result)

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        err_msg = f'An unexpected error occurred: {e}'
        if request.is_json:
            return jsonify({'error': err_msg}), 500
        return render_template('index.html', error=err_msg)

# --- Run the Flask Application ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True, host='127.0.0.1', port=5000)
