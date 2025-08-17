#!/bin/bash
echo "Starting autoPicklr Trading Simulator..."
exec python -m uvicorn main:app --host 0.0.0.0 --port 5000 --log-level info