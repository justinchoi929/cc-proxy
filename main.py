# main.py
import json
import time
import uuid
from typing import Any

import httpx
import yaml
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# --- Config ---

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CONFIG = load_config()
SERVER = CONFIG["server"]
UPSTREAM = CONFIG["upstream"]

app = FastAPI(title="cc-proxy")

if __name__ == "__main__":
    uvicorn.run(app, host=SERVER["host"], port=SERVER["port"])
