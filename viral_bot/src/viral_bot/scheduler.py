"""
Scheduler module for automatic bot runs.

Uses APScheduler to run the bot at configured times.
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from .config import get_settings
from .logging_conf import get_logger, setup_logging
from .main import run_v3_pipeline

logger = get_logger(__name__)


class BotScheduler:
    """
    Scheduler for automatic bot runs.
    """
    
    def __init__(self):
        """Initialize scheduler."""
        self.settings = get_settings()
        self.scheduler = AsyncIOScheduler()
        self._running = False
        
        logger.info("scheduler_initialized")
    
    def _create_job(self) -> None:
        """Create the scheduled job."""
        hours = self.settings.schedule_hours_list
        
        # Create cron trigger for specified hours
        # Runs at minute 0 of each specified hour
        hour_spec = ",".join(str(h) for h in hours)
        
        self.scheduler.add_job(
            self._run_job,
            CronTrigger(hour=hour_spec, minute=0),
            id="viral_bot_run",
            name="Viral Bot Run",
            replace_existing=True,
        )
        
        logger.info("job_scheduled", hours=hours)
    
    async def _run_job(self) -> None:
        """Execute a single bot run."""
        logger.info("scheduled_run_starting")
        
        try:
            stats = await run_v3_pipeline()
            logger.info(
                "scheduled_run_completed",
                status=stats.get("status"),
                outputs=stats.get("items_generated", 0),
            )
        except Exception as e:
            logger.error("scheduled_run_failed", error=str(e))
    
    def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            logger.warning("scheduler_already_running")
            return
        
        self._create_job()
        self.scheduler.start()
        self._running = True
        
        logger.info("scheduler_started")
    
    def stop(self) -> None:
        """Stop the scheduler."""
        if not self._running:
            return
        
        self.scheduler.shutdown()
        self._running = False
        
        logger.info("scheduler_stopped")
    
    def get_next_run(self) -> Optional[datetime]:
        """Get the next scheduled run time."""
        job = self.scheduler.get_job("viral_bot_run")
        if job:
            return job.next_run_time
        return None
    
    def run_now(self) -> None:
        """Trigger an immediate run (non-blocking)."""
        self.scheduler.add_job(
            self._run_job,
            id="viral_bot_manual",
            name="Manual Run",
            replace_existing=True,
        )
        logger.info("manual_run_triggered")


async def run_scheduler():
    """
    Run the scheduler indefinitely.
    
    This is typically called from the main entry point
    when running as a worker process.
    """
    settings = get_settings()
    
    # Setup logging
    setup_logging(
        level=settings.log_level,
        json_output=settings.log_json,
    )
    
    scheduler = BotScheduler()
    scheduler.start()
    
    try:
        # Keep running
        while True:
            next_run = scheduler.get_next_run()
            if next_run:
                logger.info(
                    "scheduler_waiting",
                    next_run=next_run.isoformat(),
                )
            await asyncio.sleep(3600)  # Check every hour
            
    except (KeyboardInterrupt, SystemExit):
        logger.info("scheduler_shutting_down")
        scheduler.stop()


def run_scheduler_sync():
    """Synchronous wrapper for run_scheduler."""
    asyncio.run(run_scheduler())
