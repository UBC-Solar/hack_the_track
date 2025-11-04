# UBC Solar x Hack the Track
UBC Solar's submission for Hack the Track Presented by Toyota GR

# Development

Docker can be used to build and run the application for production,
but for development, it is easier to manually start the frontend and backend.

When running the application this way, Vite will update the frontend in real time each time a file modification is saved.

## Frontend

To run the React / Vite frontend:

1. Create a new terminal and enter the `frontend` directory: `cd frontend`
2. Start the development server: `npm run dev`
3. Access the web UI at http://localhost:5173/

## Backend

To run the FastAPI backend:

1. Create a new terminal and enter the `backend` directory: `cd backend`
2. Start the backend: `uv run uvicorn main:app --reload --port 8000`
3. You can access the backend api directly at  http://localhost:8000/
4. Auto-generated FastAPI docs are available at http://localhost:8000/docs/

The page will automatically update when changes to the React frontend (e.g., `App.jsx`) are made.
