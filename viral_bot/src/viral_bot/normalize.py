"""
Normalized item schema for unified content processing.

Provides a consistent format for all content types (papers, news, policy)
with full body text and metadata extraction.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Literal
from enum import Enum


class ContentType(str, Enum):
    """Type of content source."""
    PAPER = "paper"
    NEWS = "news"
    POLICY = "policy"


@dataclass
class NormalizedItem:
    """
    A unified content item with full text and metadata.

    This extends FetchedItem with:
    - Full body text (abstract or article text)
    - Content type classification
    - Structured metadata for papers (sample size, study type, etc.)
    """
    # Core fields
    id: Optional[str] = None
    url: str = ""
    source_name: str = ""
    published_at: Optional[datetime] = None

    # Text content
    title: str = ""
    body_text: str = ""  # Full abstract or article text
    snippet: Optional[str] = None  # Short preview (optional)

    # Classification
    content_type: ContentType = ContentType.NEWS

    # Metadata (primarily for papers)
    metadata: dict = field(default_factory=dict)
    # Possible metadata fields:
    # - journal: str
    # - sample_size: str
    # - followup: str (e.g., "10 years")
    # - population: str (e.g., "adults 50-70")
    # - study_type: str (e.g., "RCT", "cohort", "meta-analysis")
    # - doi: str
    # - authors: list[str]

    # Original data for debugging
    raw_data: Optional[dict] = None

    @property
    def body_length(self) -> int:
        """Get length of body text."""
        return len(self.body_text) if self.body_text else 0

    @property
    def has_sufficient_content(self) -> bool:
        """
        Check if item has enough content for quality evaluation.

        Papers need >= 600 chars (abstract)
        News/policy need >= 1200 chars (article body)
        """
        if self.content_type == ContentType.PAPER:
            return self.body_length >= 600
        return self.body_length >= 1200

    @property
    def min_body_length(self) -> int:
        """Get minimum required body length for this content type."""
        if self.content_type == ContentType.PAPER:
            return 600
        return 1200

    def get_full_text_for_evaluation(self) -> str:
        """
        Get the full text to pass to the AI evaluator.

        Combines title, body, and relevant metadata.
        """
        parts = [f"TITLE: {self.title}"]

        if self.body_text:
            label = "ABSTRACT" if self.content_type == ContentType.PAPER else "ARTICLE TEXT"
            parts.append(f"{label}:\n{self.body_text}")

        # Add structured metadata if available
        meta_parts = []
        if self.metadata.get("journal"):
            meta_parts.append(f"Journal: {self.metadata['journal']}")
        if self.metadata.get("study_type"):
            meta_parts.append(f"Study type: {self.metadata['study_type']}")
        if self.metadata.get("sample_size"):
            meta_parts.append(f"Sample size: {self.metadata['sample_size']}")
        if self.metadata.get("population"):
            meta_parts.append(f"Population: {self.metadata['population']}")
        if self.metadata.get("followup"):
            meta_parts.append(f"Follow-up: {self.metadata['followup']}")
        if self.metadata.get("doi"):
            meta_parts.append(f"DOI: {self.metadata['doi']}")

        if meta_parts:
            parts.append("METADATA:\n" + "\n".join(meta_parts))

        return "\n\n".join(parts)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "url": self.url,
            "source_name": self.source_name,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "title": self.title,
            "body_text": self.body_text,
            "snippet": self.snippet,
            "content_type": self.content_type.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NormalizedItem":
        """Create from dictionary."""
        content_type = data.get("content_type", "news")
        if isinstance(content_type, str):
            content_type = ContentType(content_type)

        published_at = data.get("published_at")
        if isinstance(published_at, str):
            from dateutil.parser import parse as parse_date
            published_at = parse_date(published_at)

        return cls(
            id=data.get("id"),
            url=data.get("url", ""),
            source_name=data.get("source_name", ""),
            published_at=published_at,
            title=data.get("title", ""),
            body_text=data.get("body_text", ""),
            snippet=data.get("snippet"),
            content_type=content_type,
            metadata=data.get("metadata", {}),
            raw_data=data.get("raw_data"),
        )

    def __str__(self) -> str:
        return f"[{self.content_type.value}] {self.source_name}: {self.title[:60]}..."


def from_fetched_item(item, body_text: str = "", content_type: ContentType = ContentType.NEWS, metadata: dict = None) -> NormalizedItem:
    """
    Convert a FetchedItem to a NormalizedItem.

    Args:
        item: FetchedItem from source
        body_text: Full body text (abstract or article)
        content_type: Type of content
        metadata: Additional metadata dict
    """
    from .sources.base import FetchedItem

    return NormalizedItem(
        id=item.content_hash if hasattr(item, 'content_hash') else None,
        url=item.url,
        source_name=item.source_name,
        published_at=item.published_at,
        title=item.title,
        body_text=body_text or item.summary or "",
        snippet=item.summary,
        content_type=content_type,
        metadata=metadata or {},
        raw_data=item.raw_data,
    )
