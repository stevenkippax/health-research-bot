"""
Configuration management using Pydantic Settings.

All configuration is loaded from environment variables with sensible defaults.
"""

import json
import base64
from pathlib import Path
from typing import Optional
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field("gpt-4o", description="OpenAI model for evaluation/generation")
    openai_embedding_model: str = Field("text-embedding-3-small", description="OpenAI embedding model")
    openai_call_delay: float = Field(0.5, description="Delay between OpenAI calls (seconds)")

    # Google Sheets
    google_sheet_id: str = Field(..., description="Google Sheet ID")
    google_service_account_json: str = Field(..., description="Service account JSON (raw or base64)")
    google_service_account_base64: bool = Field(False, description="Is the JSON base64 encoded?")
    google_sheet_tab_name: str = Field("PostIdeas", description="Tab name in Google Sheet")

    # Database
    database_url: Optional[str] = Field(None, description="PostgreSQL URL (leave empty for SQLite)")
    database_path: str = Field("./data/viral_bot.db", description="SQLite file path")

    # Bot configuration
    freshness_hours: int = Field(48, description="Only fetch items from last N hours")
    max_outputs_per_run: int = Field(5, description="Max post ideas per run")
    min_virality_score: int = Field(40, description="Minimum virality score to output")
    dedup_similarity_threshold: float = Field(0.85, description="Semantic similarity threshold for dedup")
    dedup_retention_days: int = Field(30, description="Days to keep items in dedup store")

    # Anti-generic settings (NEW)
    min_body_chars_paper: int = Field(600, description="Minimum body text chars for papers/abstracts")
    min_body_chars_news: int = Field(1200, description="Minimum body text chars for news articles")
    min_differentiators: int = Field(2, description="Minimum differentiators required (number, population, time)")
    enable_pre_ai_filter: bool = Field(True, description="Enable pre-AI generic content filter")
    enable_post_ai_filter: bool = Field(True, description="Enable post-AI differentiator check")

    # Novelty settings (NEW)
    novelty_threshold: float = Field(0.86, description="Max similarity to recent findings before rejection")
    novelty_retention_days: int = Field(60, description="Days of history to check for novelty")
    novelty_use_embeddings: bool = Field(True, description="Use embeddings for novelty check (vs lexical)")
    novelty_penalty_factor: float = Field(0.3, description="Score penalty factor for near-duplicates")

    # Mixing/diversity settings (NEW)
    max_study_stat_per_run: int = Field(2, description="Max STUDY_STAT archetype per run")
    min_non_study_stat_per_run: int = Field(1, description="Min non-STUDY_STAT when available")
    max_per_archetype: int = Field(2, description="Max of any single archetype per run")

    # Content fetching settings (NEW)
    content_fetch_timeout: float = Field(30.0, description="Timeout for content fetching (seconds)")
    max_concurrent_content_fetches: int = Field(5, description="Max concurrent content fetch requests")

    # Story compression settings (NEW - V2 pipeline)
    min_clarity_score: int = Field(7, description="Minimum standalone clarity score (1-10)")
    require_emotional_hook: bool = Field(True, description="Require emotional_hook != 'none'")
    tier_a_target_ratio: float = Field(0.65, description="Target ratio of Tier A sources (0-1)")
    tier_b_target_ratio: float = Field(0.25, description="Target ratio of Tier B sources (0-1)")
    tier_c_target_ratio: float = Field(0.10, description="Target ratio of Tier C sources (0-1)")

    # Server
    enable_health_server: bool = Field(True, description="Enable FastAPI health server")
    port: int = Field(8000, description="Health server port")

    # Scheduler
    enable_scheduler: bool = Field(False, description="Enable built-in scheduler")
    schedule_hours: str = Field("9,18", description="Hours to run at (UTC), comma-separated")

    # Logging
    log_level: str = Field("INFO", description="Log level")
    log_json: bool = Field(False, description="Output logs as JSON")

    # Rate limiting
    max_concurrent_fetches: int = Field(5, description="Max concurrent HTTP requests for RSS")

    @field_validator("schedule_hours")
    @classmethod
    def parse_schedule_hours(cls, v: str) -> str:
        """Validate schedule hours format."""
        try:
            hours = [int(h.strip()) for h in v.split(",")]
            for h in hours:
                if not 0 <= h <= 23:
                    raise ValueError(f"Hour {h} not in range 0-23")
        except Exception as e:
            raise ValueError(f"Invalid schedule_hours format: {e}")
        return v

    @property
    def schedule_hours_list(self) -> list[int]:
        """Get schedule hours as a list of integers."""
        return [int(h.strip()) for h in self.schedule_hours.split(",")]

    @property
    def google_credentials_dict(self) -> dict:
        """Get Google service account credentials as a dictionary."""
        json_str = self.google_service_account_json

        if self.google_service_account_base64:
            json_str = base64.b64decode(json_str).decode("utf-8")

        return json.loads(json_str)

    @property
    def effective_database_url(self) -> str:
        """Get the effective database URL (PostgreSQL or SQLite)."""
        if self.database_url:
            return self.database_url

        # For SQLite, try to ensure directory exists
        # On Railway/cloud platforms with read-only filesystem, this may fail
        # In that case, use /tmp or require DATABASE_URL to be set
        db_path = Path(self.database_path)
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            # Fall back to /tmp for cloud environments with read-only filesystems
            db_path = Path("/tmp") / db_path.name
            db_path.parent.mkdir(parents=True, exist_ok=True)

        return f"sqlite:///{db_path.absolute()}"

    @property
    def is_postgres(self) -> bool:
        """Check if using PostgreSQL."""
        return bool(self.database_url and "postgres" in self.database_url.lower())


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Clear cache to allow re-reading settings (useful for tests)
def clear_settings_cache():
    """Clear the settings cache."""
    get_settings.cache_clear()


# Convenience function for direct access
settings = get_settings()
