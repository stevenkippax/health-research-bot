"""
RSS feed parser for content sources.

Handles standard RSS/Atom feeds with ETag/Last-Modified caching.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional
import time

import httpx
import feedparser
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ContentSource, FetchedItem
from ..logging_conf import get_logger

logger = get_logger(__name__)


class RSSSource(ContentSource):
    """
    Parser for RSS/Atom feeds.
    
    Supports:
    - Standard RSS 2.0 and Atom feeds
    - ETag/Last-Modified caching
    - Custom date parsing
    - Category filtering
    """
    
    def __init__(
        self,
        name: str,
        url: str,
        enabled: bool = True,
        priority: int = 1,
        category_filter: Optional[list[str]] = None,
        title_keywords: Optional[list[str]] = None,
        **kwargs
    ):
        """
        Initialize RSS source.
        
        Args:
            category_filter: Only include items with these categories
            title_keywords: Only include items with these keywords in title
        """
        super().__init__(name, url, enabled, priority, **kwargs)
        self.category_filter = category_filter
        self.title_keywords = title_keywords
        
        # Caching headers
        self._etag: Optional[str] = None
        self._last_modified: Optional[str] = None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def fetch(self, freshness_hours: int = 48) -> list[FetchedItem]:
        """Fetch items from RSS feed."""
        logger.debug("fetching_rss", source=self.name, url=self.url)
        
        items = []
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=freshness_hours)
        
        try:
            # Fetch with conditional headers
            headers = {}
            if self._etag:
                headers["If-None-Match"] = self._etag
            if self._last_modified:
                headers["If-Modified-Since"] = self._last_modified
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(self.url, headers=headers)
            
            # Handle 304 Not Modified
            if response.status_code == 304:
                logger.debug("rss_not_modified", source=self.name)
                return []
            
            response.raise_for_status()
            
            # Update cache headers
            self._etag = response.headers.get("ETag")
            self._last_modified = response.headers.get("Last-Modified")
            
            # Parse feed
            feed = feedparser.parse(response.text)
            
            if feed.bozo and feed.bozo_exception:
                logger.warning(
                    "rss_parse_warning",
                    source=self.name,
                    error=str(feed.bozo_exception)
                )
            
            # Process entries
            for entry in feed.entries:
                item = self._parse_entry(entry)
                
                if item is None:
                    continue
                
                # Apply freshness filter
                if item.published_at and item.published_at < cutoff_time:
                    continue
                
                # Apply relevance filters
                if not self._passes_filters(item):
                    continue
                
                items.append(item)
            
            logger.info(
                "rss_fetched",
                source=self.name,
                total_entries=len(feed.entries),
                filtered_items=len(items),
            )
            
        except httpx.HTTPError as e:
            logger.error("rss_fetch_error", source=self.name, error=str(e))
            raise
        
        return items
    
    def _parse_entry(self, entry: dict) -> Optional[FetchedItem]:
        """Parse a single feed entry."""
        try:
            # Get URL
            url = entry.get("link", "")
            if not url:
                return None
            
            # Get title
            title = entry.get("title", "").strip()
            if not title:
                return None
            
            # Parse date
            published_at = self._parse_date(entry)
            
            # Get summary
            summary = None
            if "summary" in entry:
                summary = entry.summary
            elif "description" in entry:
                summary = entry.description
            
            # Clean HTML from summary
            if summary:
                summary = self._clean_html(summary)
                summary = summary[:2000]  # Truncate
            
            # Get authors
            authors = []
            if "authors" in entry:
                authors = [a.get("name", "") for a in entry.authors if a.get("name")]
            elif "author" in entry:
                authors = [entry.author]
            
            # Get categories
            categories = []
            if "tags" in entry:
                categories = [t.term for t in entry.tags if hasattr(t, "term")]
            
            return FetchedItem(
                source_name=self.name,
                url=url,
                title=title,
                published_at=published_at,
                summary=summary,
                authors=authors,
                categories=categories,
                raw_data=dict(entry),
            )
            
        except Exception as e:
            logger.warning("entry_parse_error", source=self.name, error=str(e))
            return None
    
    def _parse_date(self, entry: dict) -> Optional[datetime]:
        """Parse date from various feed formats."""
        # Try common date fields
        for field in ["published_parsed", "updated_parsed", "created_parsed"]:
            parsed = entry.get(field)
            if parsed:
                try:
                    # feedparser returns a time.struct_time
                    timestamp = time.mktime(parsed)
                    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                    return dt
                except (ValueError, OverflowError):
                    continue
        
        # Try string date fields
        for field in ["published", "updated", "created"]:
            date_str = entry.get(field)
            if date_str:
                try:
                    from dateutil.parser import parse as parse_date
                    dt = parse_date(date_str)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt
                except Exception:
                    continue
        
        return None
    
    def _clean_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(text, "lxml")
        return soup.get_text(separator=" ", strip=True)
    
    def _passes_filters(self, item: FetchedItem) -> bool:
        """Check if item passes configured filters."""
        # Category filter
        if self.category_filter:
            item_categories = [c.lower() for c in item.categories]
            if not any(
                f.lower() in cat for f in self.category_filter for cat in item_categories
            ):
                return False
        
        # Title keyword filter
        if self.title_keywords:
            title_lower = item.title.lower()
            if not any(kw.lower() in title_lower for kw in self.title_keywords):
                return False
        
        return True
