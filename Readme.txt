---

# 🚲 Bike Rental Demand Prediction

A complete **machine learning regression project** that predicts **daily bike rental demand (`cnt`)** based on weather and calendar features.
The project covers **EDA, feature engineering, model training with hyperparameter tuning, API deployment, Docker containerization, and cloud deployment readiness**.

---

## 📌 Project Overview

Bike-sharing systems rely heavily on accurate demand forecasting to ensure:

* Optimal bike availability
* Efficient station rebalancing
* Better maintenance planning
* Improved customer experience

This project builds a **predictive regression model** that estimates the **total number of daily bike rentals** using historical weather and temporal data from the Capital Bikeshare system.

The final model is exposed as a **REST API**.

---

## 🎯 Problem Statement

**Goal:**
Predict the total daily bike rentals (`cnt`) given environmental and calendar-related features.

**Type:**
Supervised Machine Learning – **Regression**

**Target Variable:**
`cnt` → Total number of bikes rented per day

---

## 📊 Dataset

**Source:**
UCI Machine Learning Repository – Bike Sharing Dataset
🔗 [https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)

**File Used:**
`day.csv`

**Time Period:**
2011 – 2012 (731 daily records)

### Key Features

| Feature    | Description                      |
| ---------- | -------------------------------- |
| season     | Season (1 = Winter → 4 = Fall)   |
| yr         | Year (0 = 2011, 1 = 2012)        |
| mnth       | Month (1–12)                     |
| weekday    | Day of week (0–6)                |
| holiday    | Whether the day is a holiday     |
| workingday | Working day indicator            |
| weathersit | Weather condition (Clear → Rain) |
| temp       | Normalized temperature           |
| hum        | Normalized humidity              |
| windspeed  | Normalized wind speed            |
| **cnt**    | **Target – total daily rentals** |

✅ No missing values
❌ Leakage features (`casual`, `registered`) removed

---

## 📂 Repository Structure

```
bike-rental-demand/
├── data/
│   └── day.csv
├── notebook.ipynb              # EDA + experimentation
├── train.py                    # Model training & tuning
├── predict.py                  # Flask API service
├── best_bike_sharing_model.pkl # Trained regression model
├── encoder.pkl                 # OneHotEncoder
├── requirements.txt            # Dependencies
├── Dockerfile                  # Container configuration
└── README.md
```

---

## 🔍 Exploratory Data Analysis (EDA)

Performed in `bike_analysis.ipynb`.

### EDA Highlights

* Target distribution analysis (`cnt`)
* Seasonal and yearly trends
* Weather impact on demand
* Correlation analysis
* Outlier inspection
* Feature importance reasoning

### Key Insights

* Temperature is the strongest driver of demand
* Rentals increased significantly from 2011 to 2012
* Summer and Fall show highest demand
* Poor weather reduces rentals substantially

---

## 🧪 Feature Engineering & Preprocessing

* Dropped non-informative or leakage columns
* One-Hot Encoding applied to:

  * `season`
  * `weathersit`
  * `weekday`
  * `mnth`
* Encoder fitted **only on training data**
* Consistent feature order enforced
* Encoder saved as `encoder.pkl`

---

## 🧠 Model Training

### Models Trained

Multiple regression models were trained and evaluated:

* Linear Regression(Base model)
* Ridge Regression
* KNN Regressor
* Support Vector Regressor (SVR)
* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost Regressor

### Hyperparameter Tuning

* `GridSearchCV`
* 5-fold cross-validation
* Scoring: Negative Mean Squared Error

### Evaluation Metrics

* RMSE (primary metric)
* MAE
* R² Score

### Best Model

**Gradient Boosting Regressor**

Saved as:

```
best_bike_sharing_model.pkl
```

---

## 📈 Model Performance (Test Set)

| Model             | RMSE | MAE  | R²     |
| ----------------- | ---- | ---- | ------ |
| Gradient Boosting | ~679 | ~481 | ~0.885 |
| XGBoost           | ~637 | ~466 | ~0.899 |
| Random Forest     | ~736 | ~533 | ~0.865 |

---

## 🚀 Prediction API

The trained model is served via a **Flask API**.

### Run Locally

```bash
python predict.py
```

API runs at:

```
API Documentation

FastAPI automatically generates interactive docs:

Swagger UI: http://localhost:9696/docs

ReDoc: http://localhost:9696/redoc
```

### Health Check

```
GET /
```

### Prediction Endpoint

```
POST /predict
```

#### Example Request

```json
{
  "season": 2,
  "weathersit": 1,
  "weekday": 3,
  "mnth": 7,
  "yr": 1,
  "holiday": 0,
  "workingday": 1,
  "temp": 0.62,
  "hum": 0.58,
  "windspeed": 0.19
}
```

#### Example Response

```json
{
  "predicted_rentals": 5234
}
```

---

## ⚙️ Installation & Local Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-username/bike_sharing.git
cd bike_sharing
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate  # Linux / Mac
```

### 3. Install Dependencies

```requirements.txt
pandas
numpy
scikit-learn
flask
gunicorn
```

```bash
pip install -r requirements.txt
```
```Run the API 
uvicorn predict:app --reload --port 9696
```
---

## 🐳 Docker Containerization

### Build Image

```bash
docker build -t bike-demand-api .
```

### Run Container

```bash
docker run -p 9696:9696 bike-demand-api
```

API available at:

```
http://localhost:9696/docs
```

---

## 📦 Dependencies

Main libraries used:

* pandas
* numpy
* scikit-learn
* flask
* gunicorn

(Full list in `requirements.txt`)

---
☁️ Cloud Deployment (Render) – Live Service

The Bike Rental Demand Prediction API is deployed on Render as a production web service.

```Render Configuration
Environment:   Python
Build Command: pip install -r requirements.txt
Start Command: gunicorn -k uvicorn.workers.UvicornWorker predict:app
Port:          Auto ($PORT)
```

Live API URL
[https://bike-sharing-3.onrender.com/docs](https://bike-sharing-3.onrender.com/docs)

Test the Deployment
```Health Check
GET /
```

```
Prediction Request
POST /predict
```
bike_sharing/
├── Readme
├── Screenshot_21-1-2026_143623_127.0.0.1
📸 Screenshot testing. 
---
## 🌟 Project Highlights

* End-to-end ML pipeline
* Extensive EDA with insights
* Multiple models + hyperparameter tuning
* Clean separation of training and inference
* Saved preprocessing artifacts
* Flask-based prediction API
* Dockerized for portability
* Cloud deployment ready

---
**Author:** Nelly Alex
**Program:** ML Zoomcamp
**Project:** Bike Rental Demand Prediction 🚲


