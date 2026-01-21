FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py .
COPY best_bike_sharing_model.pkl .
COPY encoder.pkl .

EXPOSE 9696

# Start FastAPI app
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]