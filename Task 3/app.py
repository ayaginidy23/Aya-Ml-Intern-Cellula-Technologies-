import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib
import os

app = Flask(__name__)

# Load the trained model pipeline
try:
    MODEL_PATH = 'models/log_reg_pipeline.pkl'
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model_pipeline = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model_pipeline = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model_pipeline is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500

    try:
        data = request.json
        
        # The backend calculates total_guests and total_nights from the user inputs
        for item in data:
            item['total_guests'] = item.get('number of adults', 0) + item.get('number of children', 0)
            item['total_nights'] = item.get('number of weekend nights', 0) + item.get('number of week nights', 0)
        
        # Define the exact features your model was trained on
        features_for_model = [
            'average price', 
            'total_guests',
            'total_nights',
            'room type'
        ]
        
        input_df = pd.DataFrame(data)
        
        # Ensure the DataFrame only contains the features the model was trained on
        input_df = input_df[features_for_model]
        
        predictions = model_pipeline.predict(input_df)
        
        results = predictions.tolist()
        return jsonify({'predictions': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)