# 🚲 Bike Sharing Demand Prediction (Regression Project)

This project is an end-to-end **machine learning regression application** that predicts daily bike rental demand (`cnt`) based on weather and calendar features. It was developed as **Capstone 2** for the **ML Zoomcamp**, covering the full ML lifecycle: problem formulation, EDA, model training and tuning, deployment as an API, and containerization with Docker.

---

## 📌 Problem Description

Bike-sharing systems need accurate demand forecasts to:

* Optimize bike allocation and availability
* Improve operational efficiency
* Plan maintenance and staffing

Given historical data on **weather conditions** and **calendar information**, the goal is to build a regression model that predicts the **total number of bike rentals per day**.

**Target variable:**

* `cnt` – total daily bike rentals

---

## 📊 Dataset

**Bike Sharing Dataset (day.csv)**

Source:

* [https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)

The dataset contains daily aggregated bike rental data with features such as:

* Season, month, weekday
* Weather situation
* Temperature, humidity, windspeed
* Working day / holiday indicators

---

## 🧪 Exploratory Data Analysis (EDA)

Key findings from EDA:

* Strong positive correlation between `cnt` and temperature-related variables (`temp`, `atemp`)
* `registered` users drive overall demand more than `casual` users
* Seasonality plays a major role in rental patterns
* Weather conditions (humidity and weather situation) negatively impact demand

EDA included:

* Distribution analysis
* Correlation heatmaps
* Feature-target relationship analysis

---

## 🛠 Feature Engineering & Preprocessing

### Categorical Features (One-Hot Encoded)

* `season`
* `weathersit`
* `weekday`
* `mnth`

### Numerical Features

* `yr`, `holiday`, `workingday`
* `temp`, `hum`, `windspeed`

Preprocessing steps:

* OneHotEncoding with `drop='first'`
* Encoder fitted on training data and reused for validation, testing, and deployment
* Final feature matrix constructed to exactly match training feature order

---

## 🤖 Model Training & Selection

Multiple regression models were trained and tuned using **GridSearchCV**:

* Linear Regression
* Ridge Regression
* KNN Regressor
* Support Vector Regressor (SVR)
* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost Regressor

### Evaluation Metrics

* RMSE (primary metric)
* MAE
* R²

The best-performing model was selected based on **Test RMSE** and saved for deployment.

---

## 📦 Project Structure

```
bike_sharing/
├── data/
│   └── day.csv
├── notebook.ipynb        # EDA, feature engineering, model training
├── train.py              # Train final model and save artifacts
├── predict.py            # FastAPI prediction service
├── best_bike_sharing_model.pkl
├── encoder.pkl
├── requirements.txt
├── Dockerfile
├── README.md
```

---

## 🚀 Prediction Service (FastAPI)

The trained model is served via a **FastAPI web service**.

### Endpoint

**POST /predict**

### Example Request

```json
{
  "season": 2,
  "weathersit": 1,
  "weekday": 3,
  "mnth": 7,
  "yr": 1,
  "holiday": 0,
  "workingday": 1,
  "temp": 0.32,
  "hum": 0.65,
  "windspeed": 0.19
}
```

### Example Response

```json
{
  "predicted_rentals": 5234
}
```

### API Documentation

FastAPI automatically generates interactive docs:

* Swagger UI: `http://localhost:9696/docs`
* ReDoc: `http://localhost:9696/redoc`

---

## 🧪 Running Locally (Without Docker)

### 1️⃣ Create virtual environment

**Windows (PowerShell):**

```powershell
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the API

```bash
uvicorn predict:app --reload --port 9696
```

---

## 🐳 Docker Instructions

### Build Docker image

Run this command **inside the project folder**:

```bash
docker build -t bike-demand-api .
```

### Run Docker container

```bash
docker run -p 9696:9696 bike-demand-api
```

Open in browser:

* [http://localhost:9696/docs](http://localhost:9696/docs)

---

## ☁️ Cloud Deployment (Optional)

The service can be deployed on platforms such as:

* Render
* Railway
* AWS / GCP

On Render:

* Create a new Web Service
* Connect GitHub repository
* Use Docker deployment
* Start command handled via Dockerfile

---

## 📈 Evaluation Criteria Coverage

✔ Problem description
✔ Extensive EDA
✔ Multiple models + hyperparameter tuning
✔ Train script and saved model
✔ Prediction API
✔ Dependency management
✔ Docker containerization
✔ Cloud deployment ready

---

## 🏁 Conclusion

This project demonstrates a complete machine learning workflow—from raw data to a production-ready API. The final system can reliably predict bike rental demand and is suitable for real-world deployment and scaling.

---

👩‍💻 **Author:** Nelly Alex
🎓 **Program:** ML Zoomcamp
📅 **Capstone Project 2**
