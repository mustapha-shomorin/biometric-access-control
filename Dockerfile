# Use the official Python base image
FROM python:3.10-slim

# # Set environment variables
# ENV PYTHONDONTWRITEBYTECODE 1
# ENV PYTHONUNBUFFERED 1

ENV PYTHONUNBUFFERED=1
ENV PATH="/app/venv310/Scripts:$PATH"

# Set work directory
WORKDIR /app

COPY requirements.txt .

RUN python -m venv venv310
RUN ./venv310/bin/pip install --upgrade pip
RUN ./venv310/bin/pip install -r requirements.txt

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     ffmpeg \
#     libsm6 \
#     libxext6 \
#     && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the Streamlit port
EXPOSE 8501

# Streamlit configuration to avoid asking for email, etc.
ENV STREAMLIT_HOME="/app"
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false

# Run the Streamlit app
# CMD ["streamlit", "run", "final_biometric_app.py"]

# Run the Streamlit app
CMD ["./venv310/bin/python", "final_biometric_app.py"]
