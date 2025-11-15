#!/bin/bash

# Step 1: Start Docker containers in detached mode
docker compose up --build -d

# Check if Docker Compose started correctly
if [ $? -ne 0 ]; then
    echo "Docker Compose failed to start."
    exit 1
fi

# Step 2: Navigate to the backend directory
cd backend || { echo "Failed to change to backend directory"; exit 1; }

# Step 3: Activate the virtual environment (adjust if your venv is in a different path)
source ./.venv/bin/activate

# Check if virtual environment activation succeeded
if [ $? -ne 0 ]; then
    echo "Failed to activate the virtual environment."
    exit 1
fi

# Step 4: Run the Python script with the --skip-prompt flag
python ./telemetry/replayer/replayer.py --skip-prompt

# Check if the Python script executed successfully
if [ $? -ne 0 ]; then
    echo "Python script failed to run."
    exit 1
fi

echo "Setup completed successfully."
