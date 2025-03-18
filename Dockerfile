# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install torch torchvision torchaudio

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure the model directory is created and contains the model files
RUN mkdir -p model
COPY model /app/model

RUN mkdir -p data/raw data/processed data/external

ENV PYTHONPATH=/app

ENV FLASK_APP=predict_api.py

EXPOSE 9000

CMD ["python", "predict_api.py", "--config", "configs/train_config.yaml", "--combined_data_path", "data/processed/combined_data.csv", "--end_year", "2025"]