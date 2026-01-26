"""
HTML scraper for sources without RSS feeds.

Provides lightweight, respectful scraping with caching.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, Callable
from dataclasses import dataclass
import re

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ContentSource, FetchedItem
from ..logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class HTMLParsingRules:
    """
    Rules for parsing HTML content from a specific site.
    """
    # CSS selectors for finding article links
    article_selector: str
    
    # How to extract URL from article element
    url_attribute: str = "href"
    url_prefix: str = ""  # Prefix to add to relative URLs
    
    # How to extract title
    title_selector: Optional[str] = None  # If None, use text of article_selector
    
    # How to extract date (optional)
    date_selector: Optional[str] = None
    date_format: Optional[str] = None  # strptime format
    
    # How to extract summary (optional)
    summary_selector: Optional[str] = None
    
    # Custom filter function
    url_filter: Optional[Callable[[str], bool]] = None


class HTMLSource(ContentSource):
    """
    HTML scraper for news sites without RSS.
    
    Uses BeautifulSoup for parsing with configurable rules.
    """
    
    def __init__(
        self,
        name: str,
        url: str,
        parsing_rules: HTMLParsingRules,
        enabled: bool = True,
        priority: int = 3,
        **kwargs
    ):
        """
        Initialize HTML source.
        
        Args:
            parsing_rules: Rules for extracting content from HTML
        """
        super().__init__(name, url, enabled, priority, **kwargs)
        self.rules = parsing_rules
        
        # Cache
        self._last_fetch: Optional[datetime] = None
        self._cached_items: list[FetchedItem] = []
        self._cache_ttl = timedelta(minutes=30)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def fetch(self, freshness_hours: int = 48) -> list[FetchedItem]:
        """Fetch items from HTML page."""
        # Check cache
        if self._last_fetch:
            if datetime.now(timezone.utc) - self._last_fetch < self._cache_ttl:
                logger.debug("html_cache_hit", source=self.name)
                return self._filter_by_freshness(self._cached_items, freshness_hours)
        
        logger.debug("fetching_html", source=self.name, url=self.url)
        
        items = []
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    self.url,
                    headers={
                        "User-Agent": "ViralBot/1.0 (health content aggregator)",
                        "Accept": "text/html,application/xhtml+xml",
                    }
                )
                response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "lxml")
            
            # Find article elements
            articles = soup.select(self.rules.article_selector)
            
            for article in articles:
                item = self._parse_article(article)
                if item:
                    items.append(item)
            
            # Update cache
            self._cached_items = items
            self._last_fetch = datetime.now(timezone.utc)
            
            logger.info(
                "html_fetched",
                source=self.name,
                total_articles=len(articles),
                parsed_items=len(items),
            )
            
        except httpx.HTTPError as e:
            logger.error("html_fetch_error", source=self.name, error=str(e))
            raise
        
        return self._filter_by_freshness(items, freshness_hours)
    
    def _parse_article(self, element) -> Optional[FetchedItem]:
        """Parse a single article element."""
        try:
            # Extract URL
            if element.name == "a":
                url = element.get(self.rules.url_attribute, "")
            else:
                link = element.select_one("a")
                url = link.get(self.rules.url_attribute, "") if link else ""
            
            if not url:
                return None
            
            # Handle relative URLs
            if url.startswith("/"):
                url = self.rules.url_prefix + url
            
            # Apply URL filter
            if self.rules.url_filter and not self.rules.url_filter(url):
                return None
            
            # Extract title
            if self.rules.title_selector:
                title_el = element.select_one(self.rules.title_selector)
                title = title_el.get_text(strip=True) if title_el else ""
            else:
                title = element.get_text(strip=True)
            
            if not title:
                return None
            
            # Extract date (optional)
            pub_date = None
            if self.rules.date_selector:
                date_el = element.select_one(self.rules.date_selector)
                if date_el:
                    pub_date = self._parse_date(date_el)
            
            # Extract summary (optional)
            summary = None
            if self.rules.summary_selector:
                summary_el = element.select_one(self.rules.summary_selector)
                if summary_el:
                    summary = summary_el.get_text(strip=True)[:500]
            
            return FetchedItem(
                source_name=self.name,
                url=url,
                title=title,
                published_at=pub_date,
                summary=summary,
            )
            
        except Exception as e:
            logger.warning("html_parse_error", source=self.name, error=str(e))
            return None
    
    def _parse_date(self, element) -> Optional[datetime]:
        """Parse date from an element."""
        # Try datetime attribute first
        date_str = element.get("datetime")
        if not date_str:
            date_str = element.get_text(strip=True)
        
        if not date_str:
            return None
        
        try:
            if self.rules.date_format:
                dt = datetime.strptime(date_str, self.rules.date_format)
            else:
                from dateutil.parser import parse as parse_date
                dt = parse_date(date_str)
            
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            
            return dt
            
        except Exception:
            return None
    
    def _filter_by_freshness(
        self,
        items: list[FetchedItem],
        freshness_hours: int
    ) -> list[FetchedItem]:
        """Filter items by publication date."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=freshness_hours)
        
        result = []
        for item in items:
            # If no date, include it (we can't filter)
            if item.published_at is None:
                result.append(item)
            elif item.published_at >= cutoff:
                result.append(item)
        
        return result


# Pre-configured HTML sources for common sites

def create_bbc_health_source() -> HTMLSource:
    """Create BBC Health HTML source."""
    return HTMLSource(
        name="BBC Health",
        url="https://www.bbc.com/news/health",
        parsing_rules=HTMLParsingRules(
            article_selector="a[data-testid='internal-link']",
            url_prefix="https://www.bbc.com",
            url_filter=lambda u: "/news/" in u and "/health" not in u.split("/")[-1],
        ),
    )


def create_guardian_health_source() -> HTMLSource:
    """Create Guardian Health HTML source."""
    return HTMLSource(
        name="Guardian Health",
        url="https://www.theguardian.com/society/health",
        parsing_rules=HTMLParsingRules(
            article_selector="a.dcr-lv2v9o",  # May need updating
            url_prefix="https://www.theguardian.com",
            url_filter=lambda u: "/society/" in u or "/science/" in u,
        ),
    )
