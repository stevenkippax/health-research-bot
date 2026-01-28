"""
Database models and operations using SQLAlchemy.

Supports both SQLite and PostgreSQL backends.
"""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional, Generator
import hashlib
import json

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Text,
    DateTime,
    Boolean,
    ForeignKey,
    Index,
    JSON,
    func,
)
from sqlalchemy.orm import (
    declarative_base,
    sessionmaker,
    relationship,
    Session,
)

from .config import get_settings
from .logging_conf import get_logger

logger = get_logger(__name__)
Base = declarative_base()


class ContentItem(Base):
    """
    Ingested content items from various sources.
    """
    __tablename__ = "content_items"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(100), nullable=False, index=True)
    url = Column(String(2048), nullable=False, unique=True)
    title = Column(String(1024), nullable=False)
    published_at = Column(DateTime(timezone=True), nullable=True, index=True)
    summary = Column(Text, nullable=True)
    content_hash = Column(String(64), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Embedding for semantic deduplication (stored as JSON array)
    embedding = Column(JSON, nullable=True)
    
    # Relationships
    evaluations = relationship("Evaluation", back_populates="content_item", cascade="all, delete-orphan")
    outputs = relationship("Output", back_populates="content_item", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("ix_content_items_source_published", "source", "published_at"),
    )
    
    @staticmethod
    def compute_hash(url: str, title: str) -> str:
        """Compute content hash for deduplication."""
        content = f"{url.lower().strip()}|{title.lower().strip()}"
        return hashlib.sha256(content.encode()).hexdigest()


class Evaluation(Base):
    """
    AI evaluations of content items for virality potential.
    """
    __tablename__ = "evaluations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    item_id = Column(Integer, ForeignKey("content_items.id"), nullable=False, index=True)
    
    relevant = Column(Boolean, nullable=False)
    relevance_reason = Column(Text, nullable=True)
    virality_score = Column(Integer, nullable=True)  # 0-100
    confidence = Column(Float, nullable=True)  # 0.0-1.0
    
    suggested_archetype = Column(String(50), nullable=True)
    extracted_claim = Column(Text, nullable=True)
    why_it_will_work = Column(JSON, nullable=True)  # List of strings
    must_include_numbers = Column(JSON, nullable=True)  # List of strings
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Relationship
    content_item = relationship("ContentItem", back_populates="evaluations")
    
    __table_args__ = (
        Index("ix_evaluations_virality", "virality_score"),
    )


class Output(Base):
    """
    Generated post ideas ready for use.
    """
    __tablename__ = "outputs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(50), nullable=False, index=True)
    item_id = Column(Integer, ForeignKey("content_items.id"), nullable=False, index=True)
    
    headline = Column(Text, nullable=False)
    archetype = Column(String(50), nullable=False)
    image_suggestion = Column(Text, nullable=True)
    layout_notes = Column(JSON, nullable=True)  # List of strings
    highlight_words = Column(JSON, nullable=True)  # List of strings
    
    # Denormalized for easy export
    extracted_claim = Column(Text, nullable=True)
    virality_score = Column(Integer, nullable=True)
    confidence = Column(Float, nullable=True)
    why_it_will_work = Column(JSON, nullable=True)
    sources_json = Column(JSON, nullable=True)  # Source metadata
    
    status = Column(String(20), default="NEW", index=True)  # NEW, USED, REJECTED
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Relationship
    content_item = relationship("ContentItem", back_populates="outputs")
    feedback = relationship("Feedback", back_populates="output", cascade="all, delete-orphan")


class Feedback(Base):
    """
    Performance feedback for outputs (future training loop).
    """
    __tablename__ = "feedback"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    output_id = Column(Integer, ForeignKey("outputs.id"), nullable=False, index=True)
    
    actual_likes = Column(Integer, nullable=True)
    actual_shares = Column(Integer, nullable=True)
    actual_saves = Column(Integer, nullable=True)
    notes = Column(Text, nullable=True)
    
    recorded_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Relationship
    output = relationship("Output", back_populates="feedback")


class RunLog(Base):
    """
    Log of bot runs for monitoring.
    """
    __tablename__ = "run_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(50), nullable=False, unique=True, index=True)

    started_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(20), default="RUNNING")  # RUNNING, SUCCESS, FAILED

    items_fetched = Column(Integer, default=0)
    items_evaluated = Column(Integer, default=0)
    items_output = Column(Integer, default=0)

    error_message = Column(Text, nullable=True)


class WinnerHeadline(Base):
    """
    Cached winner headlines for viral likeness scoring.
    """
    __tablename__ = "winner_headlines"

    id = Column(Integer, primary_key=True, autoincrement=True)
    headline = Column(Text, nullable=False)
    headline_hash = Column(String(64), nullable=False, unique=True, index=True)
    embedding = Column(JSON, nullable=True)  # Cached embedding vector
    cluster_id = Column(Integer, nullable=True)  # Cluster assignment
    cached_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class WinnerCluster(Base):
    """
    Cluster centroids for winner headlines.
    """
    __tablename__ = "winner_clusters"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cluster_id = Column(Integer, nullable=False, unique=True, index=True)
    centroid = Column(JSON, nullable=False)  # Centroid embedding vector
    headline_count = Column(Integer, default=0)
    computed_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    

# Database manager
class Database:
    """Database connection and operation manager."""
    
    def __init__(self, url: Optional[str] = None):
        """
        Initialize database connection.
        
        Args:
            url: Database URL (defaults to settings)
        """
        settings = get_settings()
        self.url = url or settings.effective_database_url
        
        # Create engine with appropriate settings
        connect_args = {}
        if "sqlite" in self.url:
            connect_args["check_same_thread"] = False
        
        self.engine = create_engine(
            self.url,
            connect_args=connect_args,
            echo=False,
        )
        
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
        )
        
        logger.info("database_initialized", url=self.url[:50] + "...")
    
    def create_tables(self) -> None:
        """Create all tables if they don't exist."""
        Base.metadata.create_all(self.engine)
        logger.info("database_tables_created")
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
    
    # Content Items
    def get_or_create_item(
        self,
        session: Session,
        source: str,
        url: str,
        title: str,
        published_at: Optional[datetime],
        summary: Optional[str] = None,
    ) -> tuple[ContentItem, bool]:
        """
        Get existing item or create new one.
        
        Returns:
            Tuple of (item, created) where created is True if new
        """
        content_hash = ContentItem.compute_hash(url, title)
        
        # Check for existing
        existing = session.query(ContentItem).filter(
            ContentItem.url == url
        ).first()
        
        if existing:
            return existing, False
        
        # Create new
        item = ContentItem(
            source=source,
            url=url,
            title=title,
            published_at=published_at,
            summary=summary,
            content_hash=content_hash,
        )
        session.add(item)
        session.flush()  # Get ID without committing
        
        return item, True
    
    def get_recent_items(
        self,
        session: Session,
        hours: int = 48,
        source: Optional[str] = None,
    ) -> list[ContentItem]:
        """Get items from the last N hours."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        query = session.query(ContentItem).filter(
            ContentItem.published_at >= cutoff
        )
        
        if source:
            query = query.filter(ContentItem.source == source)
        
        return query.order_by(ContentItem.published_at.desc()).all()
    
    def get_items_without_evaluation(
        self,
        session: Session,
        limit: int = 100,
    ) -> list[ContentItem]:
        """Get items that haven't been evaluated yet."""
        return session.query(ContentItem).outerjoin(
            Evaluation
        ).filter(
            Evaluation.id.is_(None)
        ).limit(limit).all()
    
    def update_item_embedding(
        self,
        session: Session,
        item_id: int,
        embedding: list[float],
    ) -> None:
        """Update item's embedding vector."""
        session.query(ContentItem).filter(
            ContentItem.id == item_id
        ).update({"embedding": embedding})
    
    # Evaluations
    def save_evaluation(
        self,
        session: Session,
        item_id: int,
        relevant: bool,
        relevance_reason: Optional[str] = None,
        virality_score: Optional[int] = None,
        confidence: Optional[float] = None,
        suggested_archetype: Optional[str] = None,
        extracted_claim: Optional[str] = None,
        why_it_will_work: Optional[list[str]] = None,
        must_include_numbers: Optional[list[str]] = None,
    ) -> Evaluation:
        """Save an evaluation for a content item."""
        eval = Evaluation(
            item_id=item_id,
            relevant=relevant,
            relevance_reason=relevance_reason,
            virality_score=virality_score,
            confidence=confidence,
            suggested_archetype=suggested_archetype,
            extracted_claim=extracted_claim,
            why_it_will_work=why_it_will_work,
            must_include_numbers=must_include_numbers,
        )
        session.add(eval)
        session.flush()
        return eval
    
    def get_top_evaluated_items(
        self,
        session: Session,
        min_score: int = 40,
        limit: int = 10,
        hours: int = 48,
    ) -> list[tuple[ContentItem, Evaluation]]:
        """Get top-scoring evaluated items from recent period."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        results = session.query(ContentItem, Evaluation).join(
            Evaluation
        ).filter(
            Evaluation.relevant == True,
            Evaluation.virality_score >= min_score,
            ContentItem.published_at >= cutoff,
        ).order_by(
            Evaluation.virality_score.desc()
        ).limit(limit).all()
        
        return results
    
    # Outputs
    def save_output(
        self,
        session: Session,
        run_id: str,
        item_id: int,
        headline: str,
        archetype: str,
        image_suggestion: Optional[str] = None,
        layout_notes: Optional[list[str]] = None,
        highlight_words: Optional[list[str]] = None,
        extracted_claim: Optional[str] = None,
        virality_score: Optional[int] = None,
        confidence: Optional[float] = None,
        why_it_will_work: Optional[list[str]] = None,
        sources_json: Optional[dict] = None,
    ) -> Output:
        """Save a generated output."""
        output = Output(
            run_id=run_id,
            item_id=item_id,
            headline=headline,
            archetype=archetype,
            image_suggestion=image_suggestion,
            layout_notes=layout_notes,
            highlight_words=highlight_words,
            extracted_claim=extracted_claim,
            virality_score=virality_score,
            confidence=confidence,
            why_it_will_work=why_it_will_work,
            sources_json=sources_json,
        )
        session.add(output)
        session.flush()
        return output
    
    def get_outputs_by_run(
        self,
        session: Session,
        run_id: str,
    ) -> list[Output]:
        """Get all outputs for a specific run."""
        return session.query(Output).filter(
            Output.run_id == run_id
        ).order_by(Output.virality_score.desc()).all()
    
    def get_recent_outputs(
        self,
        session: Session,
        days: int = 7,
        status: Optional[str] = None,
    ) -> list[Output]:
        """Get recent outputs."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        
        query = session.query(Output).filter(
            Output.created_at >= cutoff
        )
        
        if status:
            query = query.filter(Output.status == status)
        
        return query.order_by(Output.created_at.desc()).all()
    
    def get_recently_used_headlines(
        self,
        session: Session,
        days: int = 30,
    ) -> list[str]:
        """Get headlines used in the last N days (for dedup)."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        
        outputs = session.query(Output.headline).filter(
            Output.created_at >= cutoff
        ).all()
        
        return [o.headline for o in outputs]
    
    def get_recently_used_urls(
        self,
        session: Session,
        days: int = 30,
    ) -> set[str]:
        """Get URLs used in the last N days (for dedup)."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        
        items = session.query(ContentItem.url).join(
            Output
        ).filter(
            Output.created_at >= cutoff
        ).all()
        
        return {i.url for i in items}
    
    # Feedback
    def save_feedback(
        self,
        session: Session,
        output_id: int,
        likes: Optional[int] = None,
        shares: Optional[int] = None,
        saves: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> Feedback:
        """Save performance feedback for an output."""
        fb = Feedback(
            output_id=output_id,
            actual_likes=likes,
            actual_shares=shares,
            actual_saves=saves,
            notes=notes,
        )
        session.add(fb)
        session.flush()
        return fb
    
    # Run logs
    def start_run(self, session: Session, run_id: str) -> RunLog:
        """Record the start of a bot run."""
        run = RunLog(run_id=run_id)
        session.add(run)
        session.commit()
        return run
    
    def complete_run(
        self,
        session: Session,
        run_id: str,
        status: str = "SUCCESS",
        items_fetched: int = 0,
        items_evaluated: int = 0,
        items_output: int = 0,
        error_message: Optional[str] = None,
    ) -> None:
        """Record completion of a bot run."""
        session.query(RunLog).filter(
            RunLog.run_id == run_id
        ).update({
            "completed_at": datetime.now(timezone.utc),
            "status": status,
            "items_fetched": items_fetched,
            "items_evaluated": items_evaluated,
            "items_output": items_output,
            "error_message": error_message,
        })
        session.commit()
    
    def get_stats(self, session: Session) -> dict:
        """Get database statistics."""
        return {
            "total_items": session.query(func.count(ContentItem.id)).scalar(),
            "total_evaluations": session.query(func.count(Evaluation.id)).scalar(),
            "total_outputs": session.query(func.count(Output.id)).scalar(),
            "total_feedback": session.query(func.count(Feedback.id)).scalar(),
            "total_runs": session.query(func.count(RunLog.id)).scalar(),
            "successful_runs": session.query(func.count(RunLog.id)).filter(
                RunLog.status == "SUCCESS"
            ).scalar(),
        }


# Singleton instance
_db_instance: Optional[Database] = None


def get_database() -> Database:
    """Get or create database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
        _db_instance.create_tables()
    return _db_instance


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Context manager for database sessions."""
    db = get_database()
    session = db.get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
