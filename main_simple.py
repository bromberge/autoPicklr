# Simple version to test startup
from fastapi import FastAPI
from sqlmodel import SQLModel, create_engine

# Basic setup without lifespan
engine = create_engine("sqlite:///picklr.db", echo=False, connect_args={"check_same_thread": False})
app = FastAPI(title="autoPicklr Trading Simulator")

@app.get("/ping")
def ping():
    return {"message": "pong"}

@app.get("/")
def root():
    return {"status": "running", "message": "Trading simulator"}