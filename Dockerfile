# Use the official lightweight Python image as base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the entire content of the current directory to the working directory
COPY . .

# Expose the port your app runs on
EXPOSE 5000

# Command to run the Flask application
CMD ["python", "api_scope_of_science_model.py"]
