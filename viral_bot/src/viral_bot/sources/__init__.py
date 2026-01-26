"""
Content sources module.

Provides parsers for various health/longevity content sources:
- RSS feeds (bioRxiv, medRxiv, news sites)
- APIs (PubMed)
- HTML scraping (fallback)
"""

from .registry import SourceRegistry, get_source_registry
from .base import ContentSource, FetchedItem

__all__ = [
    "SourceRegistry",
    "get_source_registry",
    "ContentSource",
    "FetchedItem",
]
