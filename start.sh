#!/usr/bin/env bash
set -e

echo "[start.sh] Installing deps…"
python -m pip install -q -r requirements.txt

echo "[start.sh] Launching Uvicorn…"
exec python -m uvicorn main:app --host 0.0.0.0 --port 8080 --log-level debug
