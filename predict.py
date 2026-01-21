import pickle
import pandas as pd
from flask import Flask, request, jsonify

# Create Flask app FIRST
app = Flask("bike-demand")
# Load model and encoder
with open("best_bike_sharing_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Categorical features (MUST MATCH TRAINING)
categorical_features = ['season', 'weathersit', 'weekday', 'mnth']
numerical_features = [
    'yr', 'holiday', 'workingday',
    'temp', 'hum', 'windspeed'
]

@app.route("/", methods=["GET"])
def health():
    return {"status": "Bike Demand Prediction API running"}




@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Convert input to DataFrame
    df = pd.DataFrame([data])

    # Encode categorical features
    X_cat = encoder.transform(df[categorical_features])
    encoded_feature_names = encoder.get_feature_names_out(categorical_features)
    X_cat_df = pd.DataFrame(X_cat, columns=encoded_feature_names)

    # Get non-categorical features
    non_categorical_features = [
        col for col in df.columns if col not in categorical_features
    ]
    X_num_df = df[non_categorical_features]

    # Combine features exactly like training
    X_processed = pd.concat([X_cat_df, X_num_df], axis=1)

    # Predict
    prediction = model.predict(X_processed)[0]

    return jsonify({
        "predicted_rentals": int(prediction)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696)



