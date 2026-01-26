# Main worker process (runs the bot with built-in scheduler)
worker: cd viral_bot && python -m viral_bot serve --with-scheduler

# Alternative: Just the health server (use Railway cron for scheduling)
web: cd viral_bot && python -m viral_bot serve

# One-off run (for Railway cron jobs)
# Schedule this with: railway run -- cd viral_bot && python -m viral_bot run
