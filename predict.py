import os
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Create FastAPI app
app = FastAPI(
    title="Bike Demand Prediction API",
    version="1.0.0"
)

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

# Input schema
class BikeInput(BaseModel):
    season: int
    weathersit: int
    weekday: int
    mnth: int
    yr: int
    holiday: int
    workingday: int
    temp: float
    hum: float
    windspeed: float

# Health check
@app.get("/")
def health():
    return {"status": "Bike Demand Prediction API running"}

# Prediction endpoint
@app.post("/predict")
def predict(data: BikeInput):
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])

    # Encode categorical features
    X_cat = encoder.transform(df[categorical_features])
    encoded_feature_names = encoder.get_feature_names_out(categorical_features)
    X_cat_df = pd.DataFrame(X_cat, columns=encoded_feature_names)

    # Numerical features
    X_num_df = df[numerical_features]

    # Combine features exactly like training
    X_processed = pd.concat([X_cat_df, X_num_df], axis=1)

    # Predict
    prediction = model.predict(X_processed)[0]

    return {
        "predicted_rentals": int(prediction)
    }


