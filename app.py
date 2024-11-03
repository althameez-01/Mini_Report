
import pandas as pd
from flask import Flask, request, render_template, jsonify
import joblib
import folium
import requests

app = Flask(__name__)

# Load the trained model
model_xgb = joblib.load('trained_xgb_model.pkl')  # Update to your model path

def get_area_name(latitude, longitude):
    """Get the area name from the latitude and longitude using Nominatim API."""
    try:
        # Validate latitude and longitude
        if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
            return "Invalid coordinates"

        response = requests.get(f"https://nominatim.openstreetmap.org/reverse?lat={latitude}&lon={longitude}&format=json")
        if response.status_code == 200:  # Check if the request was successful
            data = response.json()
            area_name = data.get('display_name', 'Unknown location')
            return area_name
        else:
            return "Location not found"
    except Exception as e:
        return "Location not found"

@app.route('/')
def index():
    return render_template('index.html')  # Render the form template

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])
        humidity = float(request.form['humidity'])
        temperature = float(request.form['temperature'])
        num_hospitals = int(request.form['num_hospitals'])
        num_cases = int(request.form['num_cases'])
        num_deaths = int(request.form['num_deaths'])
        
        # Extract date and split into day, month, year
        date = request.form['date']
        day, month, year = map(int, date.split('-'))

        # Validate latitude and longitude
        if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
            return jsonify({'error': 'Invalid latitude or longitude values'}), 400

        # Prepare input data for prediction
        new_data = pd.DataFrame({
            'Latitude': [latitude],
            'Humidity': [humidity],
            'Temperature': [temperature],
            'Longitude': [longitude],
            'Number of Hospitals': [num_hospitals],
            'Number of Cases': [num_cases],
            'Number of Deaths': [num_deaths],
            'Day': [day],
            'Year': [year],
            'Month': [month]
        })

        # Get predicted probabilities
        prediction_proba = model_xgb.predict_proba(new_data)
        predicted_class = prediction_proba.argmax(axis=1)[0]

        label_mapping = {0: 'low', 1: 'medium', 2: 'high', 3: 'very high'}
        predicted_label = label_mapping.get(predicted_class, 'unknown')

        confidence_score = prediction_proba[0][predicted_class]

        # Set a minimum confidence threshold
        min_confidence = 0.6
        if confidence_score < min_confidence:
            predicted_label = 'uncertain'

        # Get the area name from the coordinates
        area_name = get_area_name(latitude, longitude)

        # Create a map centered at the input location
        map_center = [latitude, longitude]
        mymap = folium.Map(location=map_center, zoom_start=12)

        # Add a marker for the location with risk level and confidence score
        folium.Marker(
            location=map_center,
            popup=f"Predicted Risk Level: {predicted_label}<br>Confidence Score: {confidence_score:.2f}<br>Area: {area_name}",
            icon=folium.Icon(color="red")
        ).add_to(mymap)

        map_html = mymap._repr_html_()

        # Return the result template with required parameters
        return render_template('result.html', prediction=predicted_label, map_html=map_html, confidence=confidence_score, area_name=area_name, latitude=latitude, longitude=longitude)
    except ValueError as ve:
        return jsonify({'error': 'Invalid input values'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
