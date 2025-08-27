import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model and scaler
try:
    model = joblib.load('Card_fraud_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    print("Error: Model or scaler files not found. Make sure they are in the same directory.")
    exit()

# Define the feature names in the correct order
feature_names = ['amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']

# FIX: Add a root route to handle visitors to the main URL
@app.route('/', methods=['GET'])
def home():
    """A simple welcome message to confirm the API is running."""
    return "Credit Card Fraud Detection API is live."

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives transaction data and returns a fraud prediction.
    """
    try:
        # Get the JSON data from the request
        data = request.get_json()
        
        # Convert the incoming JSON to a pandas DataFrame
        # The data should be a dictionary, so we wrap it in a list
        transaction_df = pd.DataFrame([data])
        
        # Ensure the columns are in the correct order
        transaction_df = transaction_df[feature_names]

        # Scale the transaction data using the loaded scaler
        scaled_transaction = scaler.transform(transaction_df)

        # Make a prediction
        prediction = model.predict(scaled_transaction)
        prediction_proba = model.predict_proba(scaled_transaction)

        # Return the result as JSON
        if prediction[0] == 1:
            result = 'Fraud'
            confidence = prediction_proba[0][1]
        else:
            result = 'Legitimate'
            confidence = prediction_proba[0][0]
            
        return jsonify({
            'prediction': result,
            'confidence': f'{confidence:.4f}'
        })

    except Exception as e:
        return jsonify({'error': str(e)})

# The if __name__ == '__main__': block has been removed for production deployment.
# Gunicorn will be used to run the app.
