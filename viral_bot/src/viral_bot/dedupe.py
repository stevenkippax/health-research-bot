"""
Deduplication module using URL matching and semantic similarity.

Prevents the bot from repeating the same topics across runs.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional
import asyncio
import numpy as np

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import get_settings
from .logging_conf import get_logger
from .sources.base import FetchedItem

logger = get_logger(__name__)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class Deduplicator:
    """
    Deduplicates content items using:
    1. Exact URL matching
    2. Semantic similarity via embeddings
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        recent_urls: Optional[set[str]] = None,
        recent_embeddings: Optional[list[tuple[str, list[float]]]] = None,
    ):
        """
        Initialize deduplicator.
        
        Args:
            similarity_threshold: Cosine similarity threshold (0.0-1.0)
            recent_urls: Set of URLs recently used (for URL dedup)
            recent_embeddings: List of (title, embedding) tuples for semantic dedup
        """
        settings = get_settings()
        
        self.threshold = similarity_threshold
        self.recent_urls = recent_urls or set()
        self.recent_embeddings = recent_embeddings or []
        
        # OpenAI client for embeddings
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.embedding_model = settings.openai_embedding_model
        
        logger.info(
            "deduplicator_initialized",
            threshold=self.threshold,
            recent_urls_count=len(self.recent_urls),
            recent_embeddings_count=len(self.recent_embeddings),
        )
    
    def deduplicate_by_url(
        self,
        items: list[FetchedItem],
    ) -> list[FetchedItem]:
        """
        Remove items with duplicate URLs.
        
        Args:
            items: List of items to deduplicate
        
        Returns:
            List of unique items (by URL)
        """
        seen_urls = set(self.recent_urls)
        unique_items = []
        
        for item in items:
            # Normalize URL
            url = item.url.lower().strip().rstrip("/")
            
            if url in seen_urls:
                logger.debug("url_duplicate_skipped", url=url[:80])
                continue
            
            seen_urls.add(url)
            unique_items.append(item)
        
        removed = len(items) - len(unique_items)
        if removed > 0:
            logger.info("url_dedup_complete", original=len(items), unique=len(unique_items))
        
        return unique_items
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def get_embedding(self, text: str) -> list[float]:
        """Get embedding for text using OpenAI."""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text[:8000],  # Truncate to model limit
        )
        return response.data[0].embedding
    
    def get_embeddings_batch(
        self,
        texts: list[str],
        batch_size: int = 20,
    ) -> list[list[float]]:
        """Get embeddings for multiple texts in batches."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Truncate each text
            batch = [t[:8000] for t in batch]
            
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=batch,
                )
                embeddings = [d.embedding for d in response.data]
                all_embeddings.extend(embeddings)
            except Exception as e:
                logger.error("embedding_batch_error", error=str(e))
                # Return empty embeddings for failed batch
                all_embeddings.extend([[] for _ in batch])
        
        return all_embeddings
    
    def is_semantically_similar(
        self,
        new_embedding: list[float],
        title: str,
    ) -> tuple[bool, Optional[str], float]:
        """
        Check if new item is semantically similar to recent items.
        
        Args:
            new_embedding: Embedding of new item
            title: Title of new item (for logging)
        
        Returns:
            Tuple of (is_similar, similar_to_title, similarity_score)
        """
        if not self.recent_embeddings or not new_embedding:
            return False, None, 0.0
        
        max_similarity = 0.0
        most_similar_title = None
        
        for recent_title, recent_embedding in self.recent_embeddings:
            if not recent_embedding:
                continue
            
            similarity = cosine_similarity(new_embedding, recent_embedding)
            
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_title = recent_title
        
        is_similar = max_similarity >= self.threshold
        
        if is_similar:
            logger.debug(
                "semantic_similarity_found",
                new_title=title[:50],
                similar_to=most_similar_title[:50] if most_similar_title else None,
                similarity=round(max_similarity, 3),
            )
        
        return is_similar, most_similar_title, max_similarity
    
    def deduplicate_semantic(
        self,
        items: list[FetchedItem],
    ) -> list[FetchedItem]:
        """
        Remove semantically similar items.
        
        This is more expensive (requires embeddings) but catches
        the same story reported by different sources.
        
        Args:
            items: List of items to deduplicate
        
        Returns:
            List of semantically unique items
        """
        if not items:
            return []
        
        logger.info("starting_semantic_dedup", items=len(items))
        
        # Get embeddings for all items
        texts = [
            f"{item.title}. {item.summary or ''}"[:1000]
            for item in items
        ]
        embeddings = self.get_embeddings_batch(texts)
        
        # Build list of recent + new embeddings for comparison
        comparison_embeddings = list(self.recent_embeddings)
        
        unique_items = []
        
        for item, embedding in zip(items, embeddings):
            if not embedding:
                # Keep items where embedding failed
                unique_items.append(item)
                continue
            
            # Check against recent and already-accepted items
            is_similar = False
            for title, emb in comparison_embeddings:
                if not emb:
                    continue
                
                similarity = cosine_similarity(embedding, emb)
                if similarity >= self.threshold:
                    logger.debug(
                        "semantic_duplicate_skipped",
                        title=item.title[:50],
                        similar_to=title[:50],
                        similarity=round(similarity, 3),
                    )
                    is_similar = True
                    break
            
            if not is_similar:
                unique_items.append(item)
                # Add to comparison set for subsequent items
                comparison_embeddings.append((item.title, embedding))
        
        removed = len(items) - len(unique_items)
        logger.info(
            "semantic_dedup_complete",
            original=len(items),
            unique=len(unique_items),
            removed=removed,
        )
        
        return unique_items
    
    def deduplicate(
        self,
        items: list[FetchedItem],
        semantic: bool = True,
    ) -> list[FetchedItem]:
        """
        Full deduplication pipeline.
        
        Args:
            items: Items to deduplicate
            semantic: Whether to use semantic deduplication
        
        Returns:
            Deduplicated items
        """
        # First pass: URL deduplication (fast)
        unique = self.deduplicate_by_url(items)
        
        # Second pass: Semantic deduplication (slower, uses API)
        if semantic and unique:
            unique = self.deduplicate_semantic(unique)
        
        return unique
    
    def add_to_recent(
        self,
        item: FetchedItem,
        embedding: Optional[list[float]] = None,
    ) -> None:
        """
        Add an item to the recent items set (for future dedup).
        
        Args:
            item: Item to add
            embedding: Pre-computed embedding (computed if not provided)
        """
        # Add URL
        url = item.url.lower().strip().rstrip("/")
        self.recent_urls.add(url)
        
        # Add embedding
        if embedding is None:
            text = f"{item.title}. {item.summary or ''}"[:1000]
            embedding = self.get_embedding(text)
        
        self.recent_embeddings.append((item.title, embedding))


def load_recent_data_from_db(db, session, days: int = 30) -> tuple[set[str], list[tuple[str, list[float]]]]:
    """
    Load recent URLs and embeddings from database.
    
    Args:
        db: Database instance
        session: Database session
        days: Number of days to look back
    
    Returns:
        Tuple of (recent_urls, recent_embeddings)
    """
    from .db import ContentItem, Output
    from datetime import datetime, timezone, timedelta
    
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    
    # Get recently used URLs
    recent_urls = db.get_recently_used_urls(session, days=days)
    
    # Get recent items with embeddings
    items = session.query(ContentItem).join(
        Output
    ).filter(
        Output.created_at >= cutoff,
        ContentItem.embedding.isnot(None),
    ).all()
    
    recent_embeddings = [
        (item.title, item.embedding)
        for item in items
        if item.embedding
    ]
    
    logger.info(
        "loaded_recent_data",
        recent_urls=len(recent_urls),
        recent_embeddings=len(recent_embeddings),
    )
    
    return recent_urls, recent_embeddings
