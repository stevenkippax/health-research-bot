"""
Source registry with credibility tiers for story-compressed headlines.

Tier A: Press-release style sources (60-80% of output should come from here)
        - Pre-written for general audiences, IG-compatible
        - ScienceDaily, EurekAlert, Medical Xpress, SciTechDaily, New Atlas

Tier B: Quality press/embargoed (needs extraction but high credibility)
        - NIH Research Matters, Nature News, The Conversation, Harvard Health

Tier C: Raw research (cross-reference for RCTs, rarely lead)
        - PubMed, bioRxiv, medRxiv, longevity niche
"""

import asyncio
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from .base import ContentSource, FetchedItem
from .rss import RSSSource
from .pubmed import PubMedSource
from .html import HTMLSource, HTMLParsingRules
from ..logging_conf import get_logger

logger = get_logger(__name__)


class CredibilityTier(str, Enum):
    """Source credibility tiers for content selection."""
    A = "A"  # Press-release style, IG-compatible
    B = "B"  # Quality press, needs extraction
    C = "C"  # Raw research, rarely lead


@dataclass
class SourceConfig:
    """Configuration for a source including tier and weight."""
    source: ContentSource
    tier: CredibilityTier
    weight: float = 1.0  # Selection weight within tier


class SourceRegistry:
    """
    Central registry with credibility-tiered sources.

    Sources are organized by tier:
    - Tier A (60-80%): Press releases, science news sites
    - Tier B (15-30%): Quality journalism, academic news
    - Tier C (5-15%): Raw research, niche sites
    """

    def __init__(self):
        """Initialize registry with default sources."""
        self.source_configs: list[SourceConfig] = []
        self._setup_default_sources()

    @property
    def sources(self) -> list[ContentSource]:
        """Backward-compatible sources list."""
        return [c.source for c in self.source_configs]

    def _add_source(
        self,
        source: ContentSource,
        tier: CredibilityTier,
        weight: float = 1.0,
    ) -> None:
        """Add a source with tier and weight."""
        self.source_configs.append(SourceConfig(
            source=source,
            tier=tier,
            weight=weight,
        ))

    def _setup_default_sources(self) -> None:
        """Configure all default sources by credibility tier."""

        # ==============================================================
        # TIER A: Press-release style / IG-compatible (60-80% of output)
        # ==============================================================

        # ScienceDaily - Health & Medicine (excellent press releases)
        self._add_source(
            RSSSource(
                name="ScienceDaily Health",
                url="https://www.sciencedaily.com/rss/health_medicine.xml",
                priority=3,
            ),
            tier=CredibilityTier.A,
            weight=2.0,  # High weight - excellent source
        )

        # ScienceDaily - Longevity subset
        self._add_source(
            RSSSource(
                name="ScienceDaily Longevity",
                url="https://www.sciencedaily.com/rss/health_medicine/healthy_aging.xml",
                priority=3,
            ),
            tier=CredibilityTier.A,
            weight=1.5,
        )

        # EurekAlert - Medicine & Health (press releases from research institutions)
        self._add_source(
            RSSSource(
                name="EurekAlert Health",
                url="https://www.eurekalert.org/rss/medicine_health.xml",
                priority=3,
            ),
            tier=CredibilityTier.A,
            weight=2.0,
        )

        # Medical Xpress (science news)
        self._add_source(
            RSSSource(
                name="Medical Xpress",
                url="https://medicalxpress.com/rss-feed/",
                priority=3,
            ),
            tier=CredibilityTier.A,
            weight=1.8,
        )

        # Medical Xpress - Health subset
        self._add_source(
            RSSSource(
                name="Medical Xpress Health",
                url="https://medicalxpress.com/rss-feed/health-news/",
                priority=3,
            ),
            tier=CredibilityTier.A,
            weight=1.5,
        )

        # SciTechDaily - Health section
        self._add_source(
            RSSSource(
                name="SciTechDaily Health",
                url="https://scitechdaily.com/category/health-medicine/feed/",
                priority=3,
            ),
            tier=CredibilityTier.A,
            weight=1.5,
        )

        # New Atlas - Science/Health
        self._add_source(
            RSSSource(
                name="New Atlas Health",
                url="https://newatlas.com/health-wellbeing/rss/",
                priority=3,
            ),
            tier=CredibilityTier.A,
            weight=1.5,
        )

        # Futurity - University research news
        self._add_source(
            RSSSource(
                name="Futurity Health",
                url="https://www.futurity.org/category/health-medicine/feed/",
                priority=3,
            ),
            tier=CredibilityTier.A,
            weight=1.5,
        )

        # Science News - Health
        self._add_source(
            RSSSource(
                name="Science News Health",
                url="https://www.sciencenews.org/topic/health-medicine/feed",
                priority=3,
            ),
            tier=CredibilityTier.A,
            weight=1.2,
        )

        # ==============================================================
        # TIER B: Quality press/embargoed (15-30% of output)
        # ==============================================================

        # NIH Research Matters (excellent, pre-written for public)
        self._add_source(
            RSSSource(
                name="NIH Research Matters",
                url="https://www.nih.gov/news-events/nih-research-matters/rss.xml",
                priority=2,
            ),
            tier=CredibilityTier.B,
            weight=2.0,
        )

        # NIA (National Institute on Aging)
        self._add_source(
            RSSSource(
                name="NIA News",
                url="https://www.nia.nih.gov/news/rss.xml",
                priority=2,
            ),
            tier=CredibilityTier.B,
            weight=1.8,
        )

        # The Conversation - Health
        self._add_source(
            RSSSource(
                name="The Conversation Health",
                url="https://theconversation.com/us/health/articles.atom",
                priority=2,
            ),
            tier=CredibilityTier.B,
            weight=1.5,
        )

        # Harvard Health Blog
        self._add_source(
            RSSSource(
                name="Harvard Health",
                url="https://www.health.harvard.edu/blog/feed",
                priority=2,
            ),
            tier=CredibilityTier.B,
            weight=1.5,
        )

        # BBC Health
        self._add_source(
            RSSSource(
                name="BBC Health",
                url="https://feeds.bbci.co.uk/news/health/rss.xml",
                priority=2,
            ),
            tier=CredibilityTier.B,
            weight=1.2,
        )

        # Guardian Health
        self._add_source(
            RSSSource(
                name="Guardian Health",
                url="https://www.theguardian.com/society/health/rss",
                priority=2,
            ),
            tier=CredibilityTier.B,
            weight=1.0,
        )

        # NPR Health
        self._add_source(
            RSSSource(
                name="NPR Health",
                url="https://feeds.npr.org/1027/rss.xml",
                priority=2,
            ),
            tier=CredibilityTier.B,
            weight=1.0,
        )

        # NYT Health
        self._add_source(
            RSSSource(
                name="NYT Health",
                url="https://rss.nytimes.com/services/xml/rss/nyt/Health.xml",
                priority=2,
            ),
            tier=CredibilityTier.B,
            weight=1.0,
        )

        # STAT News
        self._add_source(
            RSSSource(
                name="STAT News",
                url="https://www.statnews.com/feed/",
                priority=2,
                title_keywords=["study", "research", "trial", "health"],
            ),
            tier=CredibilityTier.B,
            weight=1.2,
        )

        # WHO News
        self._add_source(
            RSSSource(
                name="WHO News",
                url="https://www.who.int/rss-feeds/news-english.xml",
                priority=2,
            ),
            tier=CredibilityTier.B,
            weight=1.0,
        )

        # CDC Newsroom
        self._add_source(
            RSSSource(
                name="CDC Newsroom",
                url="https://tools.cdc.gov/podcasts/feed.asp?feedid=183",
                priority=2,
            ),
            tier=CredibilityTier.B,
            weight=1.0,
        )

        # FDA News
        self._add_source(
            RSSSource(
                name="FDA News",
                url="https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/fda-news-releases/rss.xml",
                priority=2,
            ),
            tier=CredibilityTier.B,
            weight=0.8,
        )

        # ==============================================================
        # TIER C: Raw research (5-15% - RCT cross-ref, rarely lead)
        # ==============================================================

        # PubMed API
        self._add_source(
            PubMedSource(
                name="PubMed",
                priority=1,
                max_results_per_query=15,
            ),
            tier=CredibilityTier.C,
            weight=1.0,
        )

        # bioRxiv - Neuroscience
        self._add_source(
            RSSSource(
                name="bioRxiv Neuroscience",
                url="https://connect.biorxiv.org/biorxiv_xml.php?subject=neuroscience",
                priority=1,
                title_keywords=["aging", "cognitive", "brain", "memory"],
            ),
            tier=CredibilityTier.C,
            weight=0.8,
        )

        # bioRxiv - Physiology
        self._add_source(
            RSSSource(
                name="bioRxiv Physiology",
                url="https://connect.biorxiv.org/biorxiv_xml.php?subject=physiology",
                priority=1,
                title_keywords=["aging", "longevity", "lifespan", "exercise"],
            ),
            tier=CredibilityTier.C,
            weight=0.8,
        )

        # medRxiv
        self._add_source(
            RSSSource(
                name="medRxiv",
                url="https://connect.medrxiv.org/medrxiv_xml.php?subject=public_global_health",
                priority=1,
                title_keywords=["aging", "mortality", "longevity"],
            ),
            tier=CredibilityTier.C,
            weight=0.8,
        )

        # Lifespan.io - Longevity niche
        self._add_source(
            RSSSource(
                name="Lifespan.io",
                url="https://www.lifespan.io/feed/",
                priority=1,
            ),
            tier=CredibilityTier.C,
            weight=0.5,
        )

        # Fight Aging!
        self._add_source(
            RSSSource(
                name="Fight Aging!",
                url="https://www.fightaging.org/feed/",
                priority=1,
            ),
            tier=CredibilityTier.C,
            weight=0.5,
        )

        # Longevity Technology
        self._add_source(
            RSSSource(
                name="Longevity Technology",
                url="https://longevity.technology/feed/",
                priority=1,
            ),
            tier=CredibilityTier.C,
            weight=0.6,
        )

        logger.info(
            "source_registry_initialized",
            total_sources=len(self.source_configs),
            tier_a=len([c for c in self.source_configs if c.tier == CredibilityTier.A]),
            tier_b=len([c for c in self.source_configs if c.tier == CredibilityTier.B]),
            tier_c=len([c for c in self.source_configs if c.tier == CredibilityTier.C]),
        )

    def get_tier(self, source_name: str) -> CredibilityTier:
        """Get the credibility tier for a source."""
        for config in self.source_configs:
            if config.source.name == source_name:
                return config.tier
        return CredibilityTier.C  # Default to lowest tier

    def get_weight(self, source_name: str) -> float:
        """Get the selection weight for a source."""
        for config in self.source_configs:
            if config.source.name == source_name:
                return config.weight
        return 1.0

    def add_source(
        self,
        source: ContentSource,
        tier: CredibilityTier = CredibilityTier.B,
        weight: float = 1.0,
    ) -> None:
        """Add a custom source to the registry."""
        self._add_source(source, tier, weight)
        logger.info("source_added", name=source.name, tier=tier.value)

    def remove_source(self, name: str) -> bool:
        """Remove a source by name."""
        for i, config in enumerate(self.source_configs):
            if config.source.name == name:
                del self.source_configs[i]
                logger.info("source_removed", name=name)
                return True
        return False

    def get_source(self, name: str) -> Optional[ContentSource]:
        """Get a source by name."""
        for config in self.source_configs:
            if config.source.name == name:
                return config.source
        return None

    def list_sources(
        self,
        enabled_only: bool = True,
        tier: Optional[CredibilityTier] = None,
    ) -> list[ContentSource]:
        """List sources, optionally filtered by tier."""
        configs = self.source_configs

        if tier:
            configs = [c for c in configs if c.tier == tier]

        if enabled_only:
            return [c.source for c in configs if c.source.enabled]

        return [c.source for c in configs]

    def list_sources_by_tier(self) -> dict[CredibilityTier, list[ContentSource]]:
        """Get sources organized by tier."""
        result = {tier: [] for tier in CredibilityTier}
        for config in self.source_configs:
            if config.source.enabled:
                result[config.tier].append(config.source)
        return result

    async def fetch_all(
        self,
        freshness_hours: int = 48,
        max_concurrent: int = 5,
    ) -> list[FetchedItem]:
        """
        Fetch from all enabled sources concurrently.

        Items are tagged with their source tier for later filtering.
        """
        enabled_configs = [c for c in self.source_configs if c.source.enabled]

        # Sort by priority (higher first)
        enabled_configs.sort(key=lambda c: c.source.priority, reverse=True)

        logger.info(
            "starting_fetch_all",
            total_sources=len(enabled_configs),
            freshness_hours=freshness_hours,
        )

        all_items: list[FetchedItem] = []
        errors: list[tuple[str, str]] = []

        # Use semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_with_semaphore(config: SourceConfig) -> list[FetchedItem]:
            async with semaphore:
                try:
                    items = await config.source.fetch(freshness_hours)
                    # Tag each item with tier
                    for item in items:
                        if item.raw_data is None:
                            item.raw_data = {}
                        item.raw_data["_credibility_tier"] = config.tier.value
                        item.raw_data["_source_weight"] = config.weight

                    logger.debug(
                        "source_fetch_complete",
                        source=config.source.name,
                        tier=config.tier.value,
                        items=len(items),
                    )
                    return items
                except Exception as e:
                    logger.error(
                        "source_fetch_failed",
                        source=config.source.name,
                        error=str(e),
                    )
                    errors.append((config.source.name, str(e)))
                    return []

        # Fetch from all sources concurrently
        tasks = [fetch_with_semaphore(c) for c in enabled_configs]
        results = await asyncio.gather(*tasks)

        # Combine results
        for items in results:
            all_items.extend(items)

        logger.info(
            "fetch_all_complete",
            total_items=len(all_items),
            sources_succeeded=len(enabled_configs) - len(errors),
            sources_failed=len(errors),
        )

        return all_items

    def weighted_sample(
        self,
        items: list,
        max_items: int,
        tier_distribution: Optional[dict[CredibilityTier, float]] = None,
    ) -> list:
        """
        Sample items with weighted selection by tier.

        Args:
            items: List of items (must have source_name attribute)
            max_items: Maximum items to return
            tier_distribution: Target distribution by tier (default: A=0.65, B=0.25, C=0.10)

        Returns:
            Weighted sample of items
        """
        if not tier_distribution:
            tier_distribution = {
                CredibilityTier.A: 0.65,
                CredibilityTier.B: 0.25,
                CredibilityTier.C: 0.10,
            }

        # Group items by tier
        by_tier: dict[CredibilityTier, list] = {tier: [] for tier in CredibilityTier}
        for item in items:
            source_name = getattr(item, 'source_name', None)
            if source_name:
                tier = self.get_tier(source_name)
                by_tier[tier].append(item)

        # Calculate target counts per tier
        selected = []
        for tier, target_ratio in tier_distribution.items():
            tier_items = by_tier[tier]
            if not tier_items:
                continue

            target_count = int(max_items * target_ratio)

            # Weight items within tier
            weights = [self.get_weight(getattr(item, 'source_name', '')) for item in tier_items]
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            else:
                weights = [1.0 / len(tier_items)] * len(tier_items)

            # Sample with replacement if needed
            sample_count = min(target_count, len(tier_items))
            if sample_count > 0:
                if sample_count < len(tier_items):
                    # Weighted random sample
                    sampled_indices = random.choices(
                        range(len(tier_items)),
                        weights=weights,
                        k=sample_count,
                    )
                    # Remove duplicates while preserving order
                    seen = set()
                    unique_indices = []
                    for idx in sampled_indices:
                        if idx not in seen:
                            seen.add(idx)
                            unique_indices.append(idx)
                    selected.extend([tier_items[i] for i in unique_indices])
                else:
                    selected.extend(tier_items)

        # If we don't have enough, fill from remaining items
        if len(selected) < max_items:
            remaining = [item for item in items if item not in selected]
            additional = min(max_items - len(selected), len(remaining))
            if additional > 0:
                selected.extend(random.sample(remaining, additional))

        return selected[:max_items]

    def get_stats(self) -> dict:
        """Get registry statistics."""
        enabled = [c for c in self.source_configs if c.source.enabled]

        by_type = {}
        for config in self.source_configs:
            type_name = type(config.source).__name__
            by_type[type_name] = by_type.get(type_name, 0) + 1

        by_tier = {
            "A": len([c for c in enabled if c.tier == CredibilityTier.A]),
            "B": len([c for c in enabled if c.tier == CredibilityTier.B]),
            "C": len([c for c in enabled if c.tier == CredibilityTier.C]),
        }

        return {
            "total_sources": len(self.source_configs),
            "enabled_sources": len(enabled),
            "by_type": by_type,
            "by_tier": by_tier,
        }


# Singleton instance
_registry_instance: Optional[SourceRegistry] = None


def get_source_registry() -> SourceRegistry:
    """Get or create the source registry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = SourceRegistry()
    return _registry_instance
