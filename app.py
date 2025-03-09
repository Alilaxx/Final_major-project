from flask import Flask, request, render_template
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('safety_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load processed dataset to get locality data
final_df = pd.read_csv('processed_data.csv')

# Initialize Flask app
app = Flask(__name__)

# Function to predict safety of a locality
def predict_locality_safety(locality_name):
    if locality_name not in final_df['Locality'].values:
        return "Locality not found ❌"

    # Extract features for the given locality
    locality_data = final_df[final_df['Locality'] == locality_name].drop(columns=['Locality', 'Safety_Label'])

    # Scale the input features
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
    app.run(debug=True)
