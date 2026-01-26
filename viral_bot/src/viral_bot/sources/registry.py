"""
Source registry - central configuration of all content sources.

This module defines all the sources the bot fetches from and
provides methods to run fetches across all sources.
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional

from .base import ContentSource, FetchedItem
from .rss import RSSSource
from .pubmed import PubMedSource
from .html import HTMLSource, HTMLParsingRules
from ..logging_conf import get_logger

logger = get_logger(__name__)


class SourceRegistry:
    """
    Central registry of all content sources.
    
    Manages source configuration and coordinated fetching.
    """
    
    def __init__(self):
        """Initialize registry with default sources."""
        self.sources: list[ContentSource] = []
        self._setup_default_sources()
    
    def _setup_default_sources(self) -> None:
        """Configure all default sources."""
        
        # ==================
        # RESEARCH SOURCES
        # ==================
        
        # bioRxiv - Neuroscience, Aging, Physiology
        self.sources.append(RSSSource(
            name="bioRxiv Neuroscience",
            url="https://connect.biorxiv.org/biorxiv_xml.php?subject=neuroscience",
            priority=2,
            title_keywords=["aging", "age", "cognitive", "brain", "memory", "neuro"],
        ))
        
        self.sources.append(RSSSource(
            name="bioRxiv Physiology",
            url="https://connect.biorxiv.org/biorxiv_xml.php?subject=physiology",
            priority=2,
            title_keywords=["aging", "longevity", "lifespan", "metabol", "exercise"],
        ))
        
        # medRxiv - Public health, Epidemiology
        self.sources.append(RSSSource(
            name="medRxiv",
            url="https://connect.medrxiv.org/medrxiv_xml.php?subject=public_global_health",
            priority=2,
            title_keywords=["aging", "mortality", "longevity", "health", "chronic"],
        ))
        
        # PubMed API
        self.sources.append(PubMedSource(
            name="PubMed",
            priority=3,
            max_results_per_query=15,
        ))
        
        # ==================
        # PUBLIC HEALTH SOURCES
        # ==================
        
        # WHO News
        self.sources.append(RSSSource(
            name="WHO News",
            url="https://www.who.int/rss-feeds/news-english.xml",
            priority=1,
            title_keywords=["health", "disease", "chronic", "aging", "prevention"],
        ))
        
        # CDC Newsroom
        self.sources.append(RSSSource(
            name="CDC Newsroom",
            url="https://tools.cdc.gov/podcasts/feed.asp?feedid=183",
            priority=1,
        ))
        
        # NIH News (Research Matters)
        self.sources.append(RSSSource(
            name="NIH News",
            url="https://www.nih.gov/news-events/nih-research-matters/rss.xml",
            priority=1,
            title_keywords=["aging", "health", "study", "research", "disease"],
        ))

        # NIA (National Institute on Aging) - Blog feed
        self.sources.append(RSSSource(
            name="NIA News",
            url="https://www.nia.nih.gov/news/rss.xml",
            priority=1,
        ))
        
        # ==================
        # NEWS SOURCES
        # ==================
        
        # BBC Health (RSS)
        self.sources.append(RSSSource(
            name="BBC Health",
            url="https://feeds.bbci.co.uk/news/health/rss.xml",
            priority=1,
        ))
        
        # Guardian Health
        self.sources.append(RSSSource(
            name="Guardian Health",
            url="https://www.theguardian.com/society/health/rss",
            priority=1,
        ))
        
        # Reuters Health (via Science section) - fixed URL without www
        self.sources.append(RSSSource(
            name="Reuters Science",
            url="https://reutersagency.com/feed/?best-topics=science&post_type=best",
            priority=1,
            title_keywords=["health", "study", "disease", "aging", "diet", "exercise"],
        ))
        
        # STAT News (if RSS available)
        self.sources.append(RSSSource(
            name="STAT News",
            url="https://www.statnews.com/feed/",
            priority=1,
            title_keywords=["aging", "longevity", "health", "study", "disease"],
        ))
        
        # Medical News Today - disabled, blocks automated requests
        # self.sources.append(RSSSource(
        #     name="Medical News Today",
        #     url="https://www.medicalnewstoday.com/rss/health-news.xml",
        #     priority=2,
        # ))
        
        # Science Daily - Health
        self.sources.append(RSSSource(
            name="ScienceDaily Health",
            url="https://www.sciencedaily.com/rss/health_medicine.xml",
            priority=2,
            title_keywords=["aging", "longevity", "brain", "heart", "exercise", "diet"],
        ))

        # ==================
        # LONGEVITY SOURCES
        # ==================

        # Lifespan.io - Longevity news and research
        self.sources.append(RSSSource(
            name="Lifespan.io",
            url="https://www.lifespan.io/feed/",
            priority=3,
        ))

        # Fight Aging! - Longevity research news
        self.sources.append(RSSSource(
            name="Fight Aging!",
            url="https://www.fightaging.org/feed/",
            priority=3,
        ))

        # Longevity Technology - Industry news
        self.sources.append(RSSSource(
            name="Longevity Technology",
            url="https://longevity.technology/feed/",
            priority=2,
        ))

        # Harvard Health Blog
        self.sources.append(RSSSource(
            name="Harvard Health",
            url="https://www.health.harvard.edu/blog/feed",
            priority=2,
            title_keywords=["aging", "longevity", "brain", "heart", "exercise", "diet", "study"],
        ))

        # EurekAlert Health/Medicine
        self.sources.append(RSSSource(
            name="EurekAlert Health",
            url="https://www.eurekalert.org/rss/medicine_health.xml",
            priority=2,
            title_keywords=["aging", "longevity", "lifespan", "cognitive", "disease"],
        ))

        logger.info("source_registry_initialized", total_sources=len(self.sources))
    
    def add_source(self, source: ContentSource) -> None:
        """Add a custom source to the registry."""
        self.sources.append(source)
        logger.info("source_added", name=source.name)
    
    def remove_source(self, name: str) -> bool:
        """Remove a source by name."""
        for i, source in enumerate(self.sources):
            if source.name == name:
                del self.sources[i]
                logger.info("source_removed", name=name)
                return True
        return False
    
    def get_source(self, name: str) -> Optional[ContentSource]:
        """Get a source by name."""
        for source in self.sources:
            if source.name == name:
                return source
        return None
    
    def list_sources(self, enabled_only: bool = True) -> list[ContentSource]:
        """List all sources."""
        if enabled_only:
            return [s for s in self.sources if s.enabled]
        return self.sources
    
    async def fetch_all(
        self,
        freshness_hours: int = 48,
        max_concurrent: int = 5,
    ) -> list[FetchedItem]:
        """
        Fetch from all enabled sources concurrently.
        
        Args:
            freshness_hours: Only return items from last N hours
            max_concurrent: Max concurrent fetch operations
        
        Returns:
            List of all fetched items
        """
        enabled_sources = [s for s in self.sources if s.enabled]
        
        # Sort by priority (higher first)
        enabled_sources.sort(key=lambda s: s.priority, reverse=True)
        
        logger.info(
            "starting_fetch_all",
            total_sources=len(enabled_sources),
            freshness_hours=freshness_hours,
        )
        
        all_items: list[FetchedItem] = []
        errors: list[tuple[str, str]] = []
        
        # Use semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_with_semaphore(source: ContentSource) -> list[FetchedItem]:
            async with semaphore:
                try:
                    items = await source.fetch(freshness_hours)
                    logger.debug(
                        "source_fetch_complete",
                        source=source.name,
                        items=len(items),
                    )
                    return items
                except Exception as e:
                    logger.error(
                        "source_fetch_failed",
                        source=source.name,
                        error=str(e),
                    )
                    errors.append((source.name, str(e)))
                    return []
        
        # Fetch from all sources concurrently
        tasks = [fetch_with_semaphore(s) for s in enabled_sources]
        results = await asyncio.gather(*tasks)
        
        # Combine results
        for items in results:
            all_items.extend(items)
        
        logger.info(
            "fetch_all_complete",
            total_items=len(all_items),
            sources_succeeded=len(enabled_sources) - len(errors),
            sources_failed=len(errors),
        )
        
        return all_items
    
    def get_stats(self) -> dict:
        """Get registry statistics."""
        enabled = [s for s in self.sources if s.enabled]
        
        by_type = {}
        for source in self.sources:
            type_name = type(source).__name__
            by_type[type_name] = by_type.get(type_name, 0) + 1
        
        return {
            "total_sources": len(self.sources),
            "enabled_sources": len(enabled),
            "by_type": by_type,
        }


# Singleton instance
_registry_instance: Optional[SourceRegistry] = None


def get_source_registry() -> SourceRegistry:
    """Get or create the source registry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = SourceRegistry()
    return _registry_instance
