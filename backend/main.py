# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from random import randint

app = FastAPI()

# Allow frontend to call backend (important for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Hello World!"}

@app.get("/randommember")
def read_root():
    members: list = [
        "Josh",
        "Jonah",
        "Rifana",
        "Marwan"
    ]

    return {"member": members[randint(0, len(members) - 1)]}
