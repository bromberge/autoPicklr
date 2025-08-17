#!/usr/bin/env python3
"""
Simple server runner for autoPicklr Trading Simulator
Usage: python run_server.py
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=5000, 
        reload=False,
        log_level="info"
    )