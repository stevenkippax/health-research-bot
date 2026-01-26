#!/usr/bin/env python3
"""
Railway entry point - runs the viral bot server.
Railpack detects FastAPI from the imports below.
"""
import os
import sys

# FastAPI import at top level for Railpack detection
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Add the viral_bot source to the path
_project_root = os.path.dirname(os.path.abspath(__file__))
_src_path = os.path.join(_project_root, "viral_bot", "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

# Enable scheduler by default for Railway deployment
os.environ.setdefault("ENABLE_SCHEDULER", "true")

# Import and re-export the actual app from viral_bot
from viral_bot.server import app  # noqa: E402

__all__ = ["app"]

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
