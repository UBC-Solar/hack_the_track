# Start the docker stack in detached mode
docker compose up --build -d

# Start the replayer.py Python script
cd backend
.\.venv\Scripts\activate
uv run .\telemetry\replayer\replayer.py --skip-prompt
