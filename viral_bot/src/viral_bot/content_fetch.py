"""
Content fetching module for extracting full article text.

Handles:
- PubMed/Europe PMC abstract fetching
- News article text extraction
- Robust HTML parsing with fallbacks
"""

import asyncio
import re
from typing import Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from .logging_conf import get_logger
from .normalize import NormalizedItem, ContentType, from_fetched_item
from .sources.base import FetchedItem

logger = get_logger(__name__)

# PubMed/PMC API endpoints
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
EUROPEPMC_API_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

# User agent for web fetching
USER_AGENT = "Mozilla/5.0 (compatible; HealthResearchBot/1.0; +https://github.com/health-research-bot)"

# Domains where we should not attempt to fetch full content (paywalled, blocked)
SKIP_FETCH_DOMAINS = {
    "nature.com",
    "sciencedirect.com",
    "springer.com",
    "wiley.com",
    "cell.com",
    "thelancet.com",
    "nejm.org",
    "jamanetwork.com",
}


class ContentFetcher:
    """
    Fetches full content for items from various sources.

    For papers: fetches abstracts from PubMed/Europe PMC
    For news: extracts article text from web pages
    """

    def __init__(self, timeout: float = 30.0, max_concurrent: int = 5):
        """
        Initialize content fetcher.

        Args:
            timeout: HTTP request timeout in seconds
            max_concurrent: Maximum concurrent fetches
        """
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_content(self, item: FetchedItem) -> NormalizedItem:
        """
        Fetch full content for an item.

        Determines the appropriate fetching strategy based on source.

        Args:
            item: FetchedItem from source

        Returns:
            NormalizedItem with full body text
        """
        async with self._semaphore:
            # Determine content type and fetch strategy
            if self._is_pubmed_item(item):
                return await self._fetch_pubmed_content(item)
            elif self._is_research_preprint(item):
                return await self._fetch_preprint_content(item)
            else:
                return await self._fetch_news_content(item)

    async def fetch_batch(
        self,
        items: list[FetchedItem],
        min_body_chars_paper: int = 600,
        min_body_chars_news: int = 1200,
    ) -> list[NormalizedItem]:
        """
        Fetch content for multiple items concurrently.

        Args:
            items: List of FetchedItems
            min_body_chars_paper: Minimum chars for papers
            min_body_chars_news: Minimum chars for news

        Returns:
            List of NormalizedItems with sufficient content
        """
        logger.info("starting_content_fetch", items=len(items))

        tasks = [self.fetch_content(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        normalized = []
        fetch_stats = {"success": 0, "insufficient": 0, "failed": 0}

        for item, result in zip(items, results):
            if isinstance(result, Exception):
                logger.debug("content_fetch_failed", url=item.url[:80], error=str(result))
                fetch_stats["failed"] += 1
                # Still include with available content
                result = from_fetched_item(
                    item,
                    body_text=item.summary or "",
                    content_type=self._guess_content_type(item),
                )

            # Check minimum content requirements
            min_chars = (
                min_body_chars_paper
                if result.content_type == ContentType.PAPER
                else min_body_chars_news
            )

            if result.body_length < min_chars:
                logger.debug(
                    "content_insufficient",
                    url=item.url[:80],
                    chars=result.body_length,
                    required=min_chars,
                )
                fetch_stats["insufficient"] += 1
                continue

            normalized.append(result)
            fetch_stats["success"] += 1

        logger.info(
            "content_fetch_complete",
            total=len(items),
            success=fetch_stats["success"],
            insufficient=fetch_stats["insufficient"],
            failed=fetch_stats["failed"],
        )

        return normalized

    def _is_pubmed_item(self, item: FetchedItem) -> bool:
        """Check if item is from PubMed."""
        return (
            "pubmed" in item.url.lower()
            or item.source_name.lower() == "pubmed"
            or "ncbi.nlm.nih.gov" in item.url
        )

    def _is_research_preprint(self, item: FetchedItem) -> bool:
        """Check if item is from a preprint server."""
        url_lower = item.url.lower()
        return any(
            server in url_lower
            for server in ["biorxiv", "medrxiv", "arxiv", "preprints"]
        )

    def _guess_content_type(self, item: FetchedItem) -> ContentType:
        """Guess content type from item source."""
        source_lower = item.source_name.lower()
        url_lower = item.url.lower()

        # Research sources
        if any(
            kw in source_lower or kw in url_lower
            for kw in ["pubmed", "biorxiv", "medrxiv", "arxiv", "journal", "research"]
        ):
            return ContentType.PAPER

        # Policy sources
        if any(
            kw in source_lower or kw in url_lower
            for kw in ["who", "cdc", "nih", "gov", "policy", "government"]
        ):
            return ContentType.POLICY

        return ContentType.NEWS

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
    )
    async def _fetch_pubmed_content(self, item: FetchedItem) -> NormalizedItem:
        """Fetch abstract and metadata from PubMed."""
        # Extract PMID from URL
        pmid = self._extract_pmid(item.url)

        if not pmid:
            # Fall back to basic normalization
            return from_fetched_item(
                item,
                body_text=item.summary or "",
                content_type=ContentType.PAPER,
            )

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Fetch abstract via efetch
            params = {
                "db": "pubmed",
                "id": pmid,
                "rettype": "abstract",
                "retmode": "xml",
            }

            response = await client.get(PUBMED_EFETCH_URL, params=params)
            response.raise_for_status()

            # Parse XML response
            soup = BeautifulSoup(response.text, "xml")

            # Extract abstract
            abstract_elem = soup.find("AbstractText")
            abstract = ""
            if abstract_elem:
                abstract = abstract_elem.get_text(strip=True)

            # If no abstract, try Europe PMC
            if not abstract or len(abstract) < 100:
                abstract = await self._fetch_europepmc_abstract(client, pmid)

            # Extract metadata
            metadata = self._extract_pubmed_metadata(soup, item)

            return NormalizedItem(
                id=pmid,
                url=item.url,
                source_name=item.source_name,
                published_at=item.published_at,
                title=item.title,
                body_text=abstract,
                snippet=item.summary,
                content_type=ContentType.PAPER,
                metadata=metadata,
                raw_data=item.raw_data,
            )

    async def _fetch_europepmc_abstract(
        self, client: httpx.AsyncClient, pmid: str
    ) -> str:
        """Fetch abstract from Europe PMC as fallback."""
        try:
            params = {
                "query": f"EXT_ID:{pmid}",
                "format": "json",
                "resultType": "core",
            }

            response = await client.get(EUROPEPMC_API_URL, params=params)
            response.raise_for_status()

            data = response.json()
            results = data.get("resultList", {}).get("result", [])

            if results:
                return results[0].get("abstractText", "")

        except Exception as e:
            logger.debug("europepmc_fetch_failed", pmid=pmid, error=str(e))

        return ""

    def _extract_pmid(self, url: str) -> Optional[str]:
        """Extract PMID from PubMed URL."""
        # Pattern: pubmed.ncbi.nlm.nih.gov/12345678/
        match = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", url)
        if match:
            return match.group(1)

        # Pattern: ?term=12345678[uid]
        match = re.search(r"\?term=(\d+)\[uid\]", url)
        if match:
            return match.group(1)

        return None

    def _extract_pubmed_metadata(self, soup: BeautifulSoup, item: FetchedItem) -> dict:
        """Extract metadata from PubMed XML."""
        metadata = {}

        # Journal
        journal = soup.find("Journal")
        if journal:
            title = journal.find("Title")
            if title:
                metadata["journal"] = title.get_text(strip=True)

        # Authors
        authors = soup.find_all("Author")
        if authors:
            author_names = []
            for author in authors[:5]:  # Limit to first 5
                last = author.find("LastName")
                first = author.find("ForeName")
                if last:
                    name = last.get_text(strip=True)
                    if first:
                        name = f"{first.get_text(strip=True)} {name}"
                    author_names.append(name)
            if author_names:
                metadata["authors"] = author_names

        # DOI
        article_ids = soup.find_all("ArticleId")
        for aid in article_ids:
            if aid.get("IdType") == "doi":
                metadata["doi"] = aid.get_text(strip=True)
                break

        # Try to extract study type from abstract keywords
        abstract_text = soup.find("AbstractText")
        if abstract_text:
            text = abstract_text.get_text(strip=True).lower()
            if "randomized" in text or "randomised" in text:
                metadata["study_type"] = "RCT"
            elif "meta-analysis" in text or "systematic review" in text:
                metadata["study_type"] = "meta-analysis"
            elif "cohort" in text:
                metadata["study_type"] = "cohort"
            elif "cross-sectional" in text:
                metadata["study_type"] = "cross-sectional"
            elif "case-control" in text:
                metadata["study_type"] = "case-control"

            # Try to extract sample size
            sample_match = re.search(
                r"(\d{1,3}(?:,\d{3})*|\d+)\s*(?:participants?|subjects?|patients?|adults?|individuals?|people|men|women)",
                text,
                re.IGNORECASE,
            )
            if sample_match:
                metadata["sample_size"] = sample_match.group(1)

        return metadata

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
    )
    async def _fetch_preprint_content(self, item: FetchedItem) -> NormalizedItem:
        """Fetch content from preprint servers (bioRxiv, medRxiv)."""
        # For preprints, the RSS often includes the abstract
        # If not, we can fetch the page and extract it

        body_text = item.summary or ""

        # If we have a decent summary, use it
        if len(body_text) >= 400:
            return from_fetched_item(
                item,
                body_text=body_text,
                content_type=ContentType.PAPER,
                metadata={"study_type": "preprint"},
            )

        # Otherwise try to fetch the page
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                headers={"User-Agent": USER_AGENT},
                follow_redirects=True,
            ) as client:
                response = await client.get(item.url)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "lxml")

                # Try common abstract selectors for preprint servers
                abstract = None
                for selector in [
                    "div.abstract",
                    "section#abstract",
                    "div[class*='abstract']",
                    "p[class*='abstract']",
                    "meta[name='description']",
                ]:
                    elem = soup.select_one(selector)
                    if elem:
                        if elem.name == "meta":
                            abstract = elem.get("content", "")
                        else:
                            abstract = elem.get_text(separator=" ", strip=True)
                        if len(abstract) >= 200:
                            break

                if abstract:
                    body_text = abstract

        except Exception as e:
            logger.debug("preprint_fetch_failed", url=item.url[:80], error=str(e))

        return from_fetched_item(
            item,
            body_text=body_text,
            content_type=ContentType.PAPER,
            metadata={"study_type": "preprint"},
        )

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
    )
    async def _fetch_news_content(self, item: FetchedItem) -> NormalizedItem:
        """Fetch article text from news pages."""
        # Check if domain should be skipped
        domain = urlparse(item.url).netloc.lower()
        for skip_domain in SKIP_FETCH_DOMAINS:
            if skip_domain in domain:
                return from_fetched_item(
                    item,
                    body_text=item.summary or "",
                    content_type=self._guess_content_type(item),
                )

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                headers={"User-Agent": USER_AGENT},
                follow_redirects=True,
            ) as client:
                response = await client.get(item.url)
                response.raise_for_status()

                article_text = self._extract_article_text(response.text, item.url)

                # Determine content type
                content_type = self._guess_content_type(item)

                return NormalizedItem(
                    url=item.url,
                    source_name=item.source_name,
                    published_at=item.published_at,
                    title=item.title,
                    body_text=article_text,
                    snippet=item.summary,
                    content_type=content_type,
                    metadata={},
                    raw_data=item.raw_data,
                )

        except Exception as e:
            logger.debug("news_fetch_failed", url=item.url[:80], error=str(e))

            # Fall back to summary
            return from_fetched_item(
                item,
                body_text=item.summary or "",
                content_type=self._guess_content_type(item),
            )

    def _extract_article_text(self, html: str, url: str) -> str:
        """
        Extract main article text from HTML.

        Uses multiple strategies for robustness.
        """
        soup = BeautifulSoup(html, "lxml")

        # Remove unwanted elements
        for tag in soup.find_all(
            ["script", "style", "nav", "header", "footer", "aside", "form", "iframe"]
        ):
            tag.decompose()

        # Strategy 1: Look for article tag
        article = soup.find("article")
        if article:
            text = self._clean_extracted_text(article)
            if len(text) >= 500:
                return text[:5000]  # Limit to 5000 chars

        # Strategy 2: Look for common content containers
        for selector in [
            "div[class*='article-body']",
            "div[class*='article-content']",
            "div[class*='story-body']",
            "div[class*='post-content']",
            "div[class*='entry-content']",
            "div[itemprop='articleBody']",
            "div.content",
            "main",
        ]:
            elem = soup.select_one(selector)
            if elem:
                text = self._clean_extracted_text(elem)
                if len(text) >= 500:
                    return text[:5000]

        # Strategy 3: Find the largest text block
        paragraphs = soup.find_all("p")
        if paragraphs:
            # Filter to substantial paragraphs
            good_paragraphs = [
                p.get_text(strip=True)
                for p in paragraphs
                if len(p.get_text(strip=True)) >= 50
            ]
            if good_paragraphs:
                text = " ".join(good_paragraphs)
                return text[:5000]

        # Fallback: use meta description
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            return meta_desc["content"]

        return ""

    def _clean_extracted_text(self, element) -> str:
        """Clean and format extracted text."""
        # Get text with space separator
        text = element.get_text(separator=" ", strip=True)

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove common boilerplate phrases
        boilerplate = [
            r"Subscribe to our newsletter.*?(?:\.|$)",
            r"Follow us on.*?(?:\.|$)",
            r"Share this article.*?(?:\.|$)",
            r"Related articles?:.*?(?:\.|$)",
            r"Read more:.*?(?:\.|$)",
            r"Advertisement\.?",
            r"Sign up for.*?(?:\.|$)",
        ]
        for pattern in boilerplate:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        return text.strip()


# Convenience function
async def fetch_full_content(
    items: list[FetchedItem],
    min_body_chars_paper: int = 600,
    min_body_chars_news: int = 1200,
) -> list[NormalizedItem]:
    """
    Fetch full content for a list of items.

    Args:
        items: List of FetchedItems
        min_body_chars_paper: Minimum chars for papers
        min_body_chars_news: Minimum chars for news

    Returns:
        List of NormalizedItems with sufficient content
    """
    fetcher = ContentFetcher()
    return await fetcher.fetch_batch(items, min_body_chars_paper, min_body_chars_news)
