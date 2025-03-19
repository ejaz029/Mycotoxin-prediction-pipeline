from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# ✅ Load trained model safely
try:
    model = pickle.load(open("model.pkl", "rb"))
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ Error: 'model.pkl' not found! Train the model first.")
    exit(1)

# ✅ Home route to prevent 404 errors
@app.route('/')
def home():
    return jsonify({
        "message": "Welcome to the Mycotoxin Prediction API!",
        "usage": "Send a POST request to /predict with JSON data: {'features': [value1, value2, ...]}"
    })

# ✅ Prediction route with error handling
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure JSON data is provided
        if not request.is_json:
            return jsonify({"error": "Request must be in JSON format"}), 400
        
        # Get features from JSON request
        data = request.get_json()
        if "features" not in data:
            return jsonify({"error": "Missing 'features' key in JSON"}), 400
        
        # Convert to NumPy array and reshape
        features = np.array(data["features"]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]

        return jsonify({
            "predicted_vomitoxin": round(float(prediction), 4),
            "message": "Prediction successful"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
