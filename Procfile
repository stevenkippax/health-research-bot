# Web server (FastAPI with uvicorn) - Railway default
web: uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}

# One-off run (for Railway cron jobs)
# Schedule this with: railway run -- python -c "from app import app; from viral_bot.main import run_bot_sync; run_bot_sync()"
