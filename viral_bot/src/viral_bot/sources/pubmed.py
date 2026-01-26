"""
PubMed API parser for fetching recent research articles.

Uses the NCBI E-utilities API to search for health/longevity research.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional
import asyncio

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ContentSource, FetchedItem
from ..logging_conf import get_logger

logger = get_logger(__name__)

# NCBI E-utilities base URLs
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


class PubMedSource(ContentSource):
    """
    Parser for PubMed via NCBI E-utilities API.
    
    Searches for recent articles matching health/longevity queries.
    """
    
    # Default search queries for health/longevity content
    DEFAULT_QUERIES = [
        "(aging[Title/Abstract] OR longevity[Title/Abstract]) AND humans[MeSH]",
        "(lifespan[Title/Abstract] OR healthspan[Title/Abstract]) AND humans[MeSH]",
        "cognitive decline[Title/Abstract] AND prevention[Title/Abstract]",
        "(exercise[Title/Abstract] OR physical activity[Title/Abstract]) AND mortality[Title/Abstract]",
        "(diet[Title/Abstract] OR nutrition[Title/Abstract]) AND (aging[Title/Abstract] OR longevity[Title/Abstract])",
        "sleep[Title/Abstract] AND health outcomes[Title/Abstract]",
        "(telomere[Title/Abstract] OR senescence[Title/Abstract]) AND humans[MeSH]",
    ]
    
    def __init__(
        self,
        name: str = "PubMed",
        url: str = "https://pubmed.ncbi.nlm.nih.gov",
        enabled: bool = True,
        priority: int = 2,
        queries: Optional[list[str]] = None,
        max_results_per_query: int = 20,
        **kwargs
    ):
        """
        Initialize PubMed source.
        
        Args:
            queries: Custom search queries (uses defaults if None)
            max_results_per_query: Max results to fetch per query
        """
        super().__init__(name, url, enabled, priority, **kwargs)
        self.queries = queries or self.DEFAULT_QUERIES
        self.max_results_per_query = max_results_per_query
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def fetch(self, freshness_hours: int = 48) -> list[FetchedItem]:
        """Fetch recent articles from PubMed."""
        logger.debug("fetching_pubmed", queries=len(self.queries))
        
        all_items = []
        seen_pmids = set()
        
        # Calculate date range
        # PubMed uses YYYY/MM/DD format
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(hours=freshness_hours)
        date_range = f"{start_date.strftime('%Y/%m/%d')}:{end_date.strftime('%Y/%m/%d')}[PDAT]"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for query in self.queries:
                try:
                    # Add date range to query
                    full_query = f"({query}) AND {date_range}"
                    
                    # Search for PMIDs
                    pmids = await self._search(client, full_query)
                    
                    # Filter out already seen
                    new_pmids = [p for p in pmids if p not in seen_pmids]
                    seen_pmids.update(new_pmids)
                    
                    if not new_pmids:
                        continue
                    
                    # Fetch article details
                    items = await self._fetch_details(client, new_pmids)
                    all_items.extend(items)
                    
                    # Rate limiting - NCBI requests 0.33 req/sec without API key
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.warning("pubmed_query_error", query=query[:50], error=str(e))
                    continue
        
        logger.info("pubmed_fetched", total_items=len(all_items))
        return all_items
    
    async def _search(self, client: httpx.AsyncClient, query: str) -> list[str]:
        """Search PubMed and return PMIDs."""
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": self.max_results_per_query,
            "retmode": "json",
            "sort": "date",
        }
        
        response = await client.get(ESEARCH_URL, params=params)
        response.raise_for_status()
        
        data = response.json()
        result = data.get("esearchresult", {})
        pmids = result.get("idlist", [])
        
        return pmids
    
    async def _fetch_details(
        self,
        client: httpx.AsyncClient,
        pmids: list[str]
    ) -> list[FetchedItem]:
        """Fetch article details for PMIDs."""
        if not pmids:
            return []
        
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json",
        }
        
        response = await client.get(ESUMMARY_URL, params=params)
        response.raise_for_status()
        
        data = response.json()
        result = data.get("result", {})
        
        items = []
        for pmid in pmids:
            article = result.get(pmid)
            if not article or not isinstance(article, dict):
                continue
            
            item = self._parse_article(pmid, article)
            if item:
                items.append(item)
        
        return items
    
    def _parse_article(self, pmid: str, article: dict) -> Optional[FetchedItem]:
        """Parse a single article from summary data."""
        try:
            title = article.get("title", "").strip()
            if not title:
                return None
            
            # Remove trailing period from title if present
            if title.endswith("."):
                title = title[:-1]
            
            # Build URL
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            
            # Parse publication date
            pub_date = None
            if "pubdate" in article:
                try:
                    from dateutil.parser import parse as parse_date
                    pub_date = parse_date(article["pubdate"])
                    if pub_date.tzinfo is None:
                        pub_date = pub_date.replace(tzinfo=timezone.utc)
                except Exception:
                    pass
            
            # Get authors
            authors = []
            if "authors" in article:
                authors = [a.get("name", "") for a in article["authors"] if a.get("name")]
            
            # Build summary from available fields
            summary_parts = []
            
            # Source/journal
            if "source" in article:
                summary_parts.append(f"Published in: {article['source']}")
            
            # Abstract might not be in summary, but we can note the source
            if "elocationid" in article:
                summary_parts.append(f"DOI: {article['elocationid']}")
            
            summary = " | ".join(summary_parts) if summary_parts else None
            
            return FetchedItem(
                source_name=self.name,
                url=url,
                title=title,
                published_at=pub_date,
                summary=summary,
                authors=authors,
                categories=["research", "pubmed"],
                raw_data=article,
            )
            
        except Exception as e:
            logger.warning("pubmed_parse_error", pmid=pmid, error=str(e))
            return None
