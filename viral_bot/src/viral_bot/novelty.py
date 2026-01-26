"""
Novelty checking module.

Prevents repetition of similar topics by comparing new candidates against
historical accepted findings using embeddings or lexical similarity.
"""

import re
from dataclasses import dataclass
from typing import Optional
from collections import Counter
import math

import numpy as np
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import get_settings
from .logging_conf import get_logger

logger = get_logger(__name__)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    norm_product = np.linalg.norm(a) * np.linalg.norm(b)
    if norm_product == 0:
        return 0.0
    return float(np.dot(a, b) / norm_product)


def jaccard_similarity(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity between two texts.

    Uses word-level tokens for comparison.
    """
    # Tokenize
    words1 = set(re.findall(r'\b\w+\b', text1.lower()))
    words2 = set(re.findall(r'\b\w+\b', text2.lower()))

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


def tfidf_similarity(text1: str, text2: str, idf_weights: Optional[dict] = None) -> float:
    """
    Calculate TF-IDF cosine similarity between two texts.

    Args:
        text1: First text
        text2: Second text
        idf_weights: Optional pre-computed IDF weights

    Returns:
        Similarity score 0.0-1.0
    """
    # Tokenize
    words1 = re.findall(r'\b\w+\b', text1.lower())
    words2 = re.findall(r'\b\w+\b', text2.lower())

    if not words1 or not words2:
        return 0.0

    # Build term frequency
    tf1 = Counter(words1)
    tf2 = Counter(words2)

    # Get all unique words
    all_words = set(tf1.keys()) | set(tf2.keys())

    if not all_words:
        return 0.0

    # Build TF-IDF vectors
    vec1 = []
    vec2 = []

    for word in all_words:
        # TF (normalized)
        tf1_val = tf1.get(word, 0) / len(words1)
        tf2_val = tf2.get(word, 0) / len(words2)

        # IDF (use provided weights or assume 1.0)
        idf = idf_weights.get(word, 1.0) if idf_weights else 1.0

        vec1.append(tf1_val * idf)
        vec2.append(tf2_val * idf)

    # Cosine similarity
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if norm_product == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / norm_product)


@dataclass
class NoveltyResult:
    """Result of novelty check."""
    is_novel: bool
    similarity_score: float
    most_similar_to: Optional[str] = None
    method: str = "embedding"  # embedding, jaccard, tfidf
    penalty_applied: float = 0.0


class NoveltyChecker:
    """
    Checks if new findings are novel compared to recent history.

    Uses OpenAI embeddings for semantic similarity with fallback
    to lexical similarity (Jaccard/TF-IDF) if embeddings fail.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.86,
        use_embeddings: bool = True,
        penalty_factor: float = 0.3,
    ):
        """
        Initialize novelty checker.

        Args:
            similarity_threshold: Max similarity before rejecting (default 0.86)
            use_embeddings: Whether to use OpenAI embeddings (fallback to lexical)
            penalty_factor: How much to penalize score (0-1) for near-duplicates
        """
        settings = get_settings()
        self.threshold = similarity_threshold
        self.use_embeddings = use_embeddings
        self.penalty_factor = penalty_factor

        # OpenAI client
        if use_embeddings:
            self.client = OpenAI(api_key=settings.openai_api_key)
            self.embedding_model = settings.openai_embedding_model
        else:
            self.client = None
            self.embedding_model = None

        # Recent findings storage
        self.recent_findings: list[tuple[str, Optional[list[float]]]] = []

        logger.info(
            "novelty_checker_initialized",
            threshold=similarity_threshold,
            use_embeddings=use_embeddings,
        )

    def load_recent_findings(
        self,
        findings: list[tuple[str, Optional[list[float]]]],
    ) -> None:
        """
        Load recent findings for comparison.

        Args:
            findings: List of (finding_text, embedding) tuples
        """
        self.recent_findings = findings
        logger.info("loaded_recent_findings", count=len(findings))

    def add_finding(
        self,
        finding: str,
        embedding: Optional[list[float]] = None,
    ) -> None:
        """
        Add a finding to the recent history.

        Args:
            finding: The finding text
            embedding: Pre-computed embedding (optional)
        """
        if embedding is None and self.use_embeddings:
            try:
                embedding = self._get_embedding(finding)
            except Exception as e:
                logger.debug("embedding_failed_for_finding", error=str(e))
                embedding = None

        self.recent_findings.append((finding, embedding))

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
    )
    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text using OpenAI."""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text[:8000],  # Truncate to model limit
        )
        return response.data[0].embedding

    def check_novelty(
        self,
        finding: str,
        virality_score: Optional[int] = None,
    ) -> tuple[NoveltyResult, Optional[int]]:
        """
        Check if a finding is novel compared to recent history.

        Args:
            finding: The finding text to check
            virality_score: Optional virality score to potentially penalize

        Returns:
            Tuple of (NoveltyResult, adjusted_virality_score)
        """
        if not self.recent_findings:
            return NoveltyResult(
                is_novel=True,
                similarity_score=0.0,
                method="none",
            ), virality_score

        max_similarity = 0.0
        most_similar_finding = None
        method_used = "embedding" if self.use_embeddings else "lexical"

        # Try embedding-based similarity first
        if self.use_embeddings:
            try:
                new_embedding = self._get_embedding(finding)

                for recent_finding, recent_embedding in self.recent_findings:
                    if recent_embedding is None:
                        continue

                    similarity = cosine_similarity(new_embedding, recent_embedding)

                    if similarity > max_similarity:
                        max_similarity = similarity
                        most_similar_finding = recent_finding

            except Exception as e:
                logger.debug("embedding_novelty_check_failed", error=str(e))
                # Fall back to lexical similarity
                method_used = "lexical_fallback"
                max_similarity, most_similar_finding = self._lexical_similarity_check(finding)

        else:
            # Use lexical similarity
            method_used = "lexical"
            max_similarity, most_similar_finding = self._lexical_similarity_check(finding)

        # Determine if novel
        is_novel = max_similarity < self.threshold

        # Calculate penalty if close to threshold
        penalty = 0.0
        adjusted_score = virality_score

        if virality_score is not None and max_similarity >= (self.threshold * 0.8):
            # Apply graduated penalty for near-duplicates
            # Full penalty at threshold, 0 penalty at 80% of threshold
            penalty_range = self.threshold - (self.threshold * 0.8)
            excess = max_similarity - (self.threshold * 0.8)
            penalty_ratio = min(1.0, excess / penalty_range)
            penalty = self.penalty_factor * penalty_ratio
            adjusted_score = max(0, int(virality_score * (1 - penalty)))

            logger.debug(
                "novelty_penalty_applied",
                original_score=virality_score,
                adjusted_score=adjusted_score,
                similarity=round(max_similarity, 3),
                penalty=round(penalty, 3),
            )

        result = NoveltyResult(
            is_novel=is_novel,
            similarity_score=max_similarity,
            most_similar_to=most_similar_finding[:100] if most_similar_finding else None,
            method=method_used,
            penalty_applied=penalty,
        )

        if not is_novel:
            logger.debug(
                "novelty_check_failed",
                finding=finding[:60],
                similarity=round(max_similarity, 3),
                similar_to=most_similar_finding[:60] if most_similar_finding else None,
            )

        return result, adjusted_score

    def _lexical_similarity_check(
        self,
        finding: str,
    ) -> tuple[float, Optional[str]]:
        """
        Check similarity using lexical methods.

        Uses average of Jaccard and TF-IDF similarity.
        """
        max_similarity = 0.0
        most_similar = None

        for recent_finding, _ in self.recent_findings:
            # Combine Jaccard and TF-IDF
            jaccard = jaccard_similarity(finding, recent_finding)
            tfidf = tfidf_similarity(finding, recent_finding)
            combined = (jaccard + tfidf) / 2

            if combined > max_similarity:
                max_similarity = combined
                most_similar = recent_finding

        return max_similarity, most_similar

    def get_stats(self) -> dict:
        """Get novelty checker statistics."""
        return {
            "recent_findings_count": len(self.recent_findings),
            "threshold": self.threshold,
            "use_embeddings": self.use_embeddings,
        }


def load_novelty_data_from_db(
    db,
    session,
    days: int = 60,
) -> list[tuple[str, Optional[list[float]]]]:
    """
    Load recent findings and embeddings from database.

    Args:
        db: Database instance
        session: Database session
        days: Number of days to look back

    Returns:
        List of (finding_text, embedding) tuples
    """
    from datetime import datetime, timezone, timedelta
    from .db import Output, ContentItem

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    # Get recent outputs with their content items
    outputs = session.query(Output).filter(
        Output.created_at >= cutoff,
    ).all()

    findings = []
    for output in outputs:
        # Use most_surprising_finding if available, otherwise extracted_claim
        finding_text = output.extracted_claim or ""

        # Get embedding from content item if available
        embedding = None
        if output.content_item and output.content_item.embedding:
            embedding = output.content_item.embedding

        if finding_text:
            findings.append((finding_text, embedding))

    logger.info(
        "loaded_novelty_data",
        total_findings=len(findings),
        with_embeddings=sum(1 for _, e in findings if e is not None),
    )

    return findings
