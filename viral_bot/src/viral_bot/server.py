"""
FastAPI server for health endpoints and monitoring.

Provides:
- Health check endpoint
- Latest outputs endpoint
- Database stats endpoint
- Feedback submission endpoint
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config import get_settings
from .logging_conf import get_logger, setup_logging
from .db import get_database, Output
from .main import run_bot
from .scheduler import BotScheduler

logger = get_logger(__name__)


# Pydantic models for API
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str = "1.0.0"


class StatsResponse(BaseModel):
    total_items: int
    total_evaluations: int
    total_outputs: int
    total_feedback: int
    total_runs: int
    successful_runs: int


class FeedbackRequest(BaseModel):
    output_id: int
    likes: Optional[int] = None
    shares: Optional[int] = None
    saves: Optional[int] = None
    notes: Optional[str] = None


class FeedbackResponse(BaseModel):
    success: bool
    message: str


class RunResponse(BaseModel):
    run_id: str
    status: str
    message: str


# Global scheduler instance
scheduler: Optional[BotScheduler] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global scheduler
    
    settings = get_settings()
    
    # Setup logging
    setup_logging(
        level=settings.log_level,
        json_output=settings.log_json,
    )
    
    logger.info("server_starting")
    
    # Initialize database
    db = get_database()
    
    # Start scheduler if enabled
    if settings.enable_scheduler:
        scheduler = BotScheduler()
        scheduler.start()
        logger.info("scheduler_enabled")
    
    yield
    
    # Shutdown
    if scheduler:
        scheduler.stop()
    
    logger.info("server_stopped")


# Create FastAPI app
app = FastAPI(
    title="Viral Health Post Bot",
    description="API for monitoring and controlling the Viral Health Post Bot",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get database statistics."""
    db = get_database()
    session = db.get_session()
    
    try:
        stats = db.get_stats(session)
        return StatsResponse(**stats)
    finally:
        session.close()


@app.get("/latest")
async def get_latest_outputs(limit: int = 20):
    """Get the most recent outputs."""
    db = get_database()
    session = db.get_session()
    
    try:
        outputs = db.get_recent_outputs(session, days=7)[:limit]
        
        result = []
        for output in outputs:
            # Get associated content item
            item = output.content_item
            
            result.append({
                "id": output.id,
                "run_id": output.run_id,
                "created_at": output.created_at.isoformat() if output.created_at else None,
                "headline": output.headline,
                "archetype": output.archetype,
                "image_suggestion": output.image_suggestion,
                "virality_score": output.virality_score,
                "confidence": output.confidence,
                "extracted_claim": output.extracted_claim,
                "why_it_will_work": output.why_it_will_work,
                "status": output.status,
                "source": {
                    "name": item.source if item else None,
                    "url": item.url if item else None,
                    "title": item.title if item else None,
                    "published_at": item.published_at.isoformat() if item and item.published_at else None,
                },
                "feedback": [
                    {
                        "likes": fb.actual_likes,
                        "shares": fb.actual_shares,
                        "saves": fb.actual_saves,
                        "notes": fb.notes,
                        "recorded_at": fb.recorded_at.isoformat() if fb.recorded_at else None,
                    }
                    for fb in output.feedback
                ] if output.feedback else [],
            })
        
        return {"outputs": result, "count": len(result)}
        
    finally:
        session.close()


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """Submit performance feedback for an output."""
    db = get_database()
    session = db.get_session()
    
    try:
        # Verify output exists
        output = session.query(Output).filter(Output.id == request.output_id).first()
        
        if not output:
            raise HTTPException(status_code=404, detail="Output not found")
        
        # Save feedback
        db.save_feedback(
            session,
            output_id=request.output_id,
            likes=request.likes,
            shares=request.shares,
            saves=request.saves,
            notes=request.notes,
        )
        
        session.commit()
        
        return FeedbackResponse(
            success=True,
            message=f"Feedback saved for output {request.output_id}",
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("feedback_save_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


@app.post("/run", response_model=RunResponse)
async def trigger_run(background_tasks: BackgroundTasks):
    """Trigger a manual bot run."""
    
    async def run_in_background():
        try:
            stats = await run_bot()
            logger.info("manual_run_completed", stats=stats)
        except Exception as e:
            logger.error("manual_run_failed", error=str(e))
    
    background_tasks.add_task(run_in_background)
    
    return RunResponse(
        run_id="manual_run_triggered",
        status="started",
        message="Bot run started in background",
    )


@app.get("/scheduler")
async def scheduler_status():
    """Get scheduler status."""
    if scheduler is None:
        return {"enabled": False, "next_run": None}
    
    next_run = scheduler.get_next_run()
    
    return {
        "enabled": True,
        "next_run": next_run.isoformat() if next_run else None,
    }


def run_server(
    host: str = "0.0.0.0",
    port: Optional[int] = None,
    with_scheduler: bool = False,
):
    """
    Run the FastAPI server.

    Args:
        host: Host to bind to
        port: Port to bind to (defaults to settings.port)
        with_scheduler: Whether to enable the scheduler
    """
    import os
    import uvicorn

    settings = get_settings()

    if with_scheduler:
        # Set environment variable instead of mutating cached settings
        os.environ["ENABLE_SCHEDULER"] = "true"
        # Clear the settings cache so it picks up the new value
        get_settings.cache_clear()

    port = port or settings.port

    logger.info("starting_server", host=host, port=port)

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )
