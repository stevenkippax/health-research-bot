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
    dedup_similarity_threshold: float = Field(0.85, description="Semantic similarity threshold")
    dedup_retention_days: int = Field(30, description="Days to keep items in dedup store")
    
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
    max_concurrent_fetches: int = Field(5, description="Max concurrent HTTP requests")
    
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
        
        # Ensure directory exists for SQLite
        db_path = Path(self.database_path)
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


# Convenience function for direct access
settings = get_settings()
