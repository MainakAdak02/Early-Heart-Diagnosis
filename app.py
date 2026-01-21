from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd # Added to fix the warning
import math

app = Flask(__name__)
CORS(app) # Allow the frontend to talk to this server

# --- 1. LOAD THE TRAINED MODEL ---
try:
    model = pickle.load(open('heart_model.pkl', 'rb'))
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ ERROR: heart_model.pkl not found. Please run train_model.py first!")
    exit()

# --- 2. MOCK HOSPITAL DATABASE ---
# In a real production app, this would be replaced by the Google Places API.
# Distances will be calculated from the User's location to these points.
HOSPITALS = [
    {"name": "City General Hospital", "lat": 28.6139, "lon": 77.2090, "phone": "+91-11-1234-5678"}, # Delhi
    {"name": "Green Valley Heart Center", "lat": 19.0760, "lon": 72.8777, "phone": "+91-22-8765-4321"}, # Mumbai
    {"name": "Sunrise Emergency Unit", "lat": 12.9716, "lon": 77.5946, "phone": "+91-80-5555-5555"}, # Bangalore
    {"name": "Tech City Medical", "lat": 37.7749, "lon": -122.4194, "phone": "911"}, # San Francisco
    {"name": "Metro Central Hospital", "lat": 40.7128, "lon": -74.0060, "phone": "911"} # New York
]

# Function to calculate distance (Haversine Formula)
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371 # Radius of earth in km
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat/2) * math.sin(dLat/2) + \
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
        math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c
    return d

# --- 3. THE PREDICTION API ROUTE ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from Frontend
        data = request.get_json()
        
        # Extract features list
        # We expect a list of 13 values in this exact order:
        features_list = data['features']
        
        # User Location
        user_lat = data.get('lat', 0)
        user_lon = data.get('lon', 0)

        # Define column names to match the training data exactly
        # This fixes the "UserWarning" in the terminal
        feature_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
            'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        
        # Convert list to DataFrame
        df_features = pd.DataFrame([features_list], columns=feature_names)

        # --- PREDICTION LOGIC ---
        # Get the probability (0.0 to 1.0) instead of just Yes/No
        # This allows us to be more sensitive to risk
        probabilities = model.predict_proba(df_features)
        risk_probability = probabilities[0][1] # Probability of "1" (Disease)

        # THRESHOLD SETTING:
        # 0.35 means if there is a >35% chance of heart disease, we alert the user.
        THRESHOLD = 0.35 
        
        if risk_probability > THRESHOLD:
            risk_score = 1 # High Risk
        else:
            risk_score = 0 # Low Risk

        print(f"Calculated Risk Probability: {risk_probability * 100:.2f}%")

        # Prepare Response
        response = {
            "risk": risk_score,
            "probability": f"{risk_probability * 100:.1f}%",
            "message": "High Risk Detected" if risk_score == 1 else "Low Risk",
            "nearby_hospitals": []
        }

        # If High Risk, find nearest hospitals
        if risk_score == 1:
            print(f"⚠️ High risk! Searching hospitals near {user_lat}, {user_lon}...")
            nearby = []
            for hosp in HOSPITALS:
                dist = calculate_distance(user_lat, user_lon, hosp['lat'], hosp['lon'])
                hosp_entry = hosp.copy()
                hosp_entry['distance_km'] = round(dist, 2)
                nearby.append(hosp_entry)
            
            # Sort by distance (closest first) and take top 3
            nearby.sort(key=lambda x: x['distance_km'])
            response['nearby_hospitals'] = nearby[:3]

        return jsonify(response)

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": str(e)})

# --- 4. START SERVER ---
if __name__ == '__main__':
    print("Starting Flask Server...")
    app.run(debug=True, port=5000)