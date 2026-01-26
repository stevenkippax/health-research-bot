#!/usr/bin/env python3
"""
Railway entry point - runs the viral bot server.
This file exists at the root so Railway can auto-detect and run it.
"""
import os
import sys

# Add the viral_bot source to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "viral_bot", "src"))

# Enable scheduler by default for Railway deployment
os.environ.setdefault("ENABLE_SCHEDULER", "true")

from viral_bot.server import app, run_server

if __name__ == "__main__":
    # Get port from environment (Railway sets PORT)
    port = int(os.environ.get("PORT", 8000))

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
