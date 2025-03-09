from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

# Check if required files exist
model_path = "safety_model.pkl"
scaler_path = "scaler.pkl"
data_path = "processed_data.csv"

if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(data_path):
    raise FileNotFoundError("One or more required files (model, scaler, dataset) are missing!")

# Load model, scaler, and dataset
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
final_df = pd.read_csv(data_path)

# Initialize Flask app
app = Flask(__name__)

# Function to predict safety of a locality
def predict_locality_safety(locality_name):
    locality_name = locality_name.strip().lower()  # Normalize input

    # Find matching locality (case-insensitive)
    matching_locality = final_df[final_df["Locality"].str.lower() == locality_name]

    if matching_locality.empty:
        return "Locality not found ❌"

    # Extract features and scale
    locality_data = matching_locality.drop(columns=["Locality", "Safety_Label"])
    locality_data_scaled = scaler.transform(locality_data)

    # Predict safety
    prediction = model.predict(locality_data_scaled)[0]
    return "Safe ✅" if prediction == 1 else "Unsafe ❌"

# Home route
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        locality = request.form["locality"]
        prediction = predict_locality_safety(locality)
    return render_template("index.html", prediction=prediction)

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # Required for Render
