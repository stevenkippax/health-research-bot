"""
Normalized item schema for unified content processing.

Provides a consistent format for all content types with full body text,
metadata extraction, and credibility tier tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal, TYPE_CHECKING
from enum import Enum


class ContentType(str, Enum):
    """Type of content source - expanded for story compression."""
    PAPER = "paper"           # Academic paper/preprint
    NEWS = "news"             # News article
    POLICY = "policy"         # Government/org policy
    PRESS_RELEASE = "press_release"  # University/institution press release
    SCIENCE_NEWS = "science_news"    # Science journalism (ScienceDaily, etc.)


class CredibilityTier(str, Enum):
    """Source credibility tiers for content selection."""
    A = "A"  # Press-release style, IG-compatible
    B = "B"  # Quality press, needs extraction
    C = "C"  # Raw research, rarely lead


@dataclass
class NormalizedItem:
    """
    A unified content item with full text, metadata, and credibility tier.

    This extends FetchedItem with:
    - Full body text (abstract or article text)
    - Content type classification
    - Credibility tier from source
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
    credibility_tier: CredibilityTier = CredibilityTier.B  # Default to middle tier

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
        Press releases / science news need >= 800 chars
        News/policy need >= 1200 chars (article body)
        """
        return self.body_length >= self.min_body_length

    @property
    def min_body_length(self) -> int:
        """Get minimum required body length for this content type."""
        if self.content_type == ContentType.PAPER:
            return 600
        if self.content_type in (ContentType.PRESS_RELEASE, ContentType.SCIENCE_NEWS):
            return 800
        return 1200

    @property
    def is_tier_a_source(self) -> bool:
        """Check if this item is from a Tier A (IG-compatible) source."""
        return self.credibility_tier == CredibilityTier.A

    def get_full_text_for_evaluation(self) -> str:
        """
        Get the full text to pass to the AI evaluator.

        Combines title, body, and relevant metadata.
        """
        parts = [f"TITLE: {self.title}"]

        if self.body_text:
            if self.content_type == ContentType.PAPER:
                label = "ABSTRACT"
            elif self.content_type in (ContentType.PRESS_RELEASE, ContentType.SCIENCE_NEWS):
                label = "PRESS RELEASE"
            else:
                label = "ARTICLE TEXT"
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
            "credibility_tier": self.credibility_tier.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NormalizedItem":
        """Create from dictionary."""
        content_type = data.get("content_type", "news")
        if isinstance(content_type, str):
            content_type = ContentType(content_type)

        credibility_tier = data.get("credibility_tier", "B")
        if isinstance(credibility_tier, str):
            credibility_tier = CredibilityTier(credibility_tier)

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
            credibility_tier=credibility_tier,
            metadata=data.get("metadata", {}),
            raw_data=data.get("raw_data"),
        )

    def __str__(self) -> str:
        return f"[{self.credibility_tier.value}:{self.content_type.value}] {self.source_name}: {self.title[:60]}..."


def from_fetched_item(
    item,
    body_text: str = "",
    content_type: ContentType = ContentType.NEWS,
    credibility_tier: Optional[CredibilityTier] = None,
    metadata: dict = None,
) -> NormalizedItem:
    """
    Convert a FetchedItem to a NormalizedItem.

    Args:
        item: FetchedItem from source
        body_text: Full body text (abstract or article)
        content_type: Type of content
        credibility_tier: Source credibility tier (auto-detected from raw_data if not provided)
        metadata: Additional metadata dict
    """
    # Auto-detect tier from raw_data if available
    if credibility_tier is None:
        raw_tier = None
        if hasattr(item, 'raw_data') and item.raw_data:
            raw_tier = item.raw_data.get("_credibility_tier")
        if raw_tier:
            credibility_tier = CredibilityTier(raw_tier)
        else:
            credibility_tier = CredibilityTier.B  # Default

    # Auto-detect content type based on source patterns
    if content_type == ContentType.NEWS:
        source_lower = item.source_name.lower()
        if any(kw in source_lower for kw in ["sciencedaily", "eurekalert", "futurity", "medical xpress"]):
            content_type = ContentType.PRESS_RELEASE
        elif any(kw in source_lower for kw in ["scitechdaily", "new atlas", "science news"]):
            content_type = ContentType.SCIENCE_NEWS
        elif any(kw in source_lower for kw in ["pubmed", "biorxiv", "medrxiv"]):
            content_type = ContentType.PAPER
        elif any(kw in source_lower for kw in ["who", "cdc", "fda", "nih", "nia", "gov"]):
            content_type = ContentType.POLICY

    return NormalizedItem(
        id=item.content_hash if hasattr(item, 'content_hash') else None,
        url=item.url,
        source_name=item.source_name,
        published_at=item.published_at,
        title=item.title,
        body_text=body_text or item.summary or "",
        snippet=item.summary,
        content_type=content_type,
        credibility_tier=credibility_tier,
        metadata=metadata or {},
        raw_data=item.raw_data,
    )
