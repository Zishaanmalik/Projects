# Base Python image
FROM python:3.13.5

# Set working directory inside the container
WORKDIR /app

# Copy all files from your PC's img/ folder into the container
COPY . /app

# Install required libraries
RUN pip install --no-cache-dir -r requirements.txt

# Streamlit uses port 5000 (we expose it)
EXPOSE 5000

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=5000", "--server.address=0.0.0.0", "--server.enableCORS=false"]
