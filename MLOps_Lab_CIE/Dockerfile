FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/predict_cli.py ./src/predict_cli.py
COPY models/champion.joblib ./models/champion.joblib

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "src/predict_cli.py"]
