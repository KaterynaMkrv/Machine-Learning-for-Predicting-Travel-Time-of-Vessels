# Base image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything needed explicitly
COPY main.py .  
COPY data/ ./data/
COPY scripts/ ./scripts/
COPY cache/ ./cache/
COPY docs/ ./docs/


# Expose Streamlit port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]

#docker build -t ship-duration-app .
#docker run -p 8501:8501 ship-duration-app
#http://localhost:8501