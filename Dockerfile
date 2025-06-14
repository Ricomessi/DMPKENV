# Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED True
ENV APP_HOME /app

# Create and set the working directory
WORKDIR $APP_HOME

# Install system dependencies that might be needed by Prophet
# pystan, a dependency of Prophet, often needs a C++ compiler
RUN apt-get update && apt-get install -y build-essential

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the models directory into the container
COPY . .

# Tell the container to listen on port 8080 (standard for Cloud Run)
EXPOSE 8080

# Define the command to run the application using Gunicorn
# This command starts 4 worker processes.
# It binds to 0.0.0.0:8080.
# 'app:app' means "in the file named app.py, run the Flask object named app".
CMD exec gunicorn --bind :$PORT --workers 4 --threads 1 --timeout 0 app:app