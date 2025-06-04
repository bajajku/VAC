#!/bin/bash

# Exit on any error
set -e

# Create a virtual environment, if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Navigate to the backend directory
cd ../backend

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Start the server
echo "Starting server..."
uvicorn api:app_api --host 0.0.0.0 --port 8000