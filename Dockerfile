FROM python:3.10-slim

# Install system-level dependencies (PortAudio and ffmpeg for sounddevice)
RUN apt-get update && apt-get install -y portaudio19-dev ffmpeg && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files into the container
COPY . .

# Expose the port your app uses
EXPOSE 8080

# Run the application
CMD ["python3", "app.py"]
