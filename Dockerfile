# Use official slim Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy project files into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Run your app
CMD ["python", "app.py"]
