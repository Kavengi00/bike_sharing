import requests

url = "http://127.0.0.1:9696/predict"
data = {
    "season": 2,
    "yr": 1,
    "mnth": 6,
    "holiday": 0,
    "weekday": 3,
    "workingday": 1,
    "weathersit": 1,
    "temp": 0.3,
    "hum": 0.7,
    "windspeed": 0.2
}

response = requests.post(url, json=data)
print(response.json())

