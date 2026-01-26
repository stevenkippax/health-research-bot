"""
Base classes for content sources.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, AsyncIterator
import hashlib


@dataclass
class FetchedItem:
    """
    A content item fetched from a source.
    
    This is the normalized format used across all source types.
    """
    source_name: str
    url: str
    title: str
    published_at: Optional[datetime]
    summary: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    raw_data: Optional[dict] = None  # Original data for debugging
    
    @property
    def content_hash(self) -> str:
        """Compute a hash for deduplication."""
        content = f"{self.url.lower().strip()}|{self.title.lower().strip()}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def __str__(self) -> str:
        return f"[{self.source_name}] {self.title[:60]}..."
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "source_name": self.source_name,
            "url": self.url,
            "title": self.title,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "summary": self.summary,
            "authors": self.authors,
            "categories": self.categories,
        }


class ContentSource(ABC):
    """
    Abstract base class for content sources.
    
    Each source (RSS, API, HTML) implements this interface.
    """
    
    def __init__(
        self,
        name: str,
        url: str,
        enabled: bool = True,
        priority: int = 1,
        **kwargs
    ):
        """
        Initialize content source.
        
        Args:
            name: Human-readable source name
            url: Base URL or feed URL
            enabled: Whether this source is active
            priority: Higher = fetched first (for rate limiting)
        """
        self.name = name
        self.url = url
        self.enabled = enabled
        self.priority = priority
        self.extra_config = kwargs
    
    @abstractmethod
    async def fetch(
        self,
        freshness_hours: int = 48,
    ) -> list[FetchedItem]:
        """
        Fetch items from this source.
        
        Args:
            freshness_hours: Only return items from last N hours
        
        Returns:
            List of fetched items
        """
        pass
    
    def is_relevant(self, item: FetchedItem) -> bool:
        """
        Check if an item is relevant to health/longevity topics.
        
        Override in subclasses for source-specific filtering.
        Default implementation accepts all items.
        """
        return True
    
    def __str__(self) -> str:
        status = "✓" if self.enabled else "✗"
        return f"[{status}] {self.name} ({self.url[:50]}...)"
    
    def __repr__(self) -> str:
        return f"ContentSource(name={self.name!r}, enabled={self.enabled})"
