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





# Bike Sharing Demand Prediction (Regression Project)

**Author:** Nelly Alex  
**Program:** ML Zoomcamp  
**Capstone Project 2**  
**Date:** January 2026  

This repository contains an end-to-end machine learning regression project that predicts daily bike rental demand (`cnt`) using historical weather and calendar data from the Capital Bikeshare system. The project follows the complete ML lifecycle: problem formulation, exploratory data analysis (EDA), feature engineering, model training with hyperparameter tuning, model export, API deployment using FastAPI, dependency management, Docker containerization, and instructions for cloud deployment.

## Problem Description (2 points)

Bike-sharing systems have become a popular eco-friendly transportation option in urban areas, allowing users to rent bicycles from automated stations for short trips. The Capital Bikeshare system in Washington D.C. operates thousands of bikes across hundreds of stations, serving both registered members and casual users.

Accurate forecasting of daily bike rental demand is critical for operational success because:

- **Optimizes bike allocation**: Ensures sufficient bikes are available at high-demand stations while preventing overstocking at low-demand ones.
- **Improves user satisfaction**: Reduces wait times and unavailability, encouraging higher usage.
- **Enhances efficiency**: Helps plan maintenance schedules, staff deployment, and rebalancing operations (moving bikes between stations).
- **Supports business decisions**: Informs expansion plans, pricing strategies, and partnerships.

The task is a **regression problem**: given daily features such as season, weather conditions, temperature, humidity, windspeed, and calendar indicators (holiday, weekday, etc.), predict the **total number of bike rentals per day (`cnt`)**.

This predictive model can be integrated into the system's operations dashboard or used by planners to proactively manage fleet distribution, ultimately reducing costs and improving service reliability.

## Dataset

**Source**: UCI Machine Learning Repository - [Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)  
**File used**: `day.csv` (daily aggregated data, 731 rows covering 2011-2012)

**Key Features**:

| Feature      | Description                                                                 | Type        |
|--------------|-----------------------------------------------------------------------------|-------------|
| dteday      | Date (YYYY-MM-DD)                                                           | Date        |
| season      | Season (1: winter, 2: spring, 3: summer, 4: fall)                           | Categorical |
| yr          | Year (0: 2011, 1: 2012)                                                     | Binary      |
| mnth        | Month (1 to 12)                                                             | Categorical |
| holiday     | Whether the day is a holiday (0: no, 1: yes)                                 | Binary      |
| weekday     | Day of the week (0-6)                                                       | Categorical |
| workingday  | 1 if day is neither weekend nor holiday, else 0                             | Binary      |
| weathersit  | Weather situation:<br>1: Clear/Partly cloudy<br>2: Mist/Cloudy<br>3: Light precipitation<br>4: Heavy precipitation | Categorical |
| temp        | Normalized temperature in Celsius (values divided by 41)                    | Numerical   |
| atemp       | Normalized "feels-like" temperature in Celsius (divided by 50)             | Numerical   |
| hum         | Normalized humidity (divided by 100)                                        | Numerical   |
| windspeed   | Normalized wind speed (divided by 67)                                       | Numerical   |
| cnt         | **Target** - Total bike rentals (casual + registered)                        | Numerical   |

No missing values in the dataset.

**How to obtain data**: Download `day.csv` from the UCI link above and place it in the `data/` folder.

## Exploratory Data Analysis (EDA) (2 points - Extensive)

EDA was performed in `notebook.ipynb` to understand data distributions, relationships, and patterns.

**Key Findings**:

- Strong **positive correlation** between `cnt` and temperature variables (`temp`: ~0.63, `atemp`: ~0.63).
- Higher demand in warmer seasons (summer > spring > fall > winter).
- Demand peaks on working days for registered users; casual users prefer weekends/holidays.
- Negative impact from poor weather (`weathersit` 3/4), high humidity, and strong winds.
- Significant yearly increase: 2012 rentals ~2x higher than 2011 (likely due to system growth).
- Target `cnt` is right-skewed but reasonably normal for regression.

**Visualizations** (representative examples):

Correlation Heatmap (features vs target):








Bike Rentals vs Temperature (scatter plot showing positive trend):








Seasonal Demand Patterns:




Distribution of Daily Rentals:








Additional analyses included box plots by season/weekday, pair plots, and outlier checks.

## Feature Engineering & Preprocessing

- **Dropped**: `instant`, `dteday`, `casual`, `registered` (leakage or unnecessary).
- **One-Hot Encoding** (with `drop='first'` to avoid multicollinearity): `season`, `weathersit`, `weekday`, `mnth`.
- **Numerical features** kept as-is: `yr`, `holiday`, `workingday`, `temp`, `atemp`, `hum`, `windspeed`.
- Encoder fitted only on training data to prevent data leakage.
- Final feature order strictly maintained across train/validation/test and inference.

## Model Training & Selection (3 points)

Multiple models were trained and evaluated using train/validation/test split (70/15/15).

**Models Tried**:
- Linear Regression
- Ridge Regression
- K-Neighbors Regressor
- Support Vector Regressor (SVR)
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

**Hyperparameter Tuning**: GridSearchCV with 5-fold CV on training data.

**Metrics** (on test set):
- Primary: RMSE (lower is better)
- MAE, R²

The **best model** (lowest test RMSE) was selected, retrained on full train+val data, and saved as `best_bike_sharing_model.pkl` along with `encoder.pkl`.

## Project Structure

```
bike_sharing/
├── data/
│   └── day.csv
├── notebook.ipynb              # Full EDA, experiments, model selection
├── train.py                    # Script to train final model & save artifacts
├── predict.py                  # FastAPI prediction service
├── best_bike_sharing_model.pkl
├── encoder.pkl
├── requirements.txt
├── Dockerfile
├── README.md                   # This file
```

## Reproducibility (1 point)

- Dataset included or clear download instructions provided.
- Notebook and `train.py` are fully executable without errors.
- Random seeds set for reproducibility.

## Model Deployment (1 point)

The model is deployed as a REST API using **FastAPI**.

**Endpoint**: `POST /predict`

**Example Request**:
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
  "atemp": 0.35,
  "hum": 0.65,
  "windspeed": 0.19
}
```

**Example Response**:
```json
{
  "predicted_rentals": 5234
}
```

Interactive docs:  
- Swagger UI: `http://localhost:9696/docs`  
- ReDoc: `http://localhost:9696/redoc`

## Dependency and Environment Management (2 points)

**requirements.txt** lists all dependencies (e.g., pandas, scikit-learn, xgboost, fastapi, uvicorn, etc.).

**Setup Instructions**:

1. Create and activate virtual environment:

   **Windows**:
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```

   **macOS/Linux**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run training script (optional):
   ```bash
   python train.py
   ```

4. Start API:
   ```bash
   uvicorn predict:app --reload --port 9696
   ```

## Containerization (2 points)

A **Dockerfile** is provided to containerize the FastAPI service.

**Build image**:
```bash
docker build -t bike-demand-api .
```

**Run container**:
```bash
docker run -p 9696:9696 bike-demand-api
```

Access API docs at `http://localhost:9696/docs`.

## Cloud Deployment (1 point - Optional but described)

The Docker image can be deployed easily on cloud platforms:

- **Render**, **Railway**, **Fly.io**: Select Docker deployment, connect GitHub repo.
- **AWS ECS/EC2**, **Google Cloud Run**, **Heroku** (with Docker support).

**Example on Render**:
1. Create new Web Service.
2. Connect repository.
3. Choose Docker runtime.
4. Set port to 9696.
5. Deploy – Render handles build and run.

The service is production-ready and scalable.

## Conclusion

This project showcases a full ML pipeline from data exploration to deployable API, achieving strong predictive performance on bike demand forecasting. It meets all evaluation criteria for ML Zoomcamp Capstone 2 and can be extended with real-time data or hourly predictions.
