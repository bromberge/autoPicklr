#!/usr/bin/env python3
"""
Test server to isolate the issue
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="Test Server")

@app.get("/")
def read_root():
    return {"message": "Test server is working"}

@app.get("/test")
def test():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")