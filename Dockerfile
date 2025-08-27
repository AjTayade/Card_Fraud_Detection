# Use an official lightweight Python image as the base
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application files into the container
COPY app.py .
COPY Card_fraud_model.pkl .
COPY scaler.pkl .

# Expose port 5000 to allow communication with the app
EXPOSE 5000

# FIX: Command to run the Gunicorn server when the container starts
CMD ["gunicorn", "app:app"]
