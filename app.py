#!/usr/bin/env python3
"""
Railway entry point - runs the viral bot server.
This file exists at the root so Railway/Railpack can auto-detect and run it.

The FastAPI `app` is exposed at module level for uvicorn to import:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""
import os
import sys

# Add the viral_bot source to the path BEFORE any imports
_project_root = os.path.dirname(os.path.abspath(__file__))
_src_path = os.path.join(_project_root, "viral_bot", "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

# Enable scheduler by default for Railway deployment
os.environ.setdefault("ENABLE_SCHEDULER", "true")

# Import the FastAPI app - this must be at module level for uvicorn app:app to work
from viral_bot.server import app  # noqa: E402

# Re-export app explicitly for uvicorn
__all__ = ["app"]

if __name__ == "__main__":
    # Get port from environment (Railway sets PORT)
    port = int(os.environ.get("PORT", 8000))

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
