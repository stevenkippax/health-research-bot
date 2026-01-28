"""
Viral likeness scoring using winner corpus embeddings.

Computes similarity between candidate headlines and top-performing
winner headlines to predict viral potential.
"""

import re
import numpy as np
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass, field

from openai import OpenAI
from sklearn.cluster import KMeans

from .config import get_settings
from .logging_conf import get_logger
from .db import get_session, WinnerHeadline, WinnerCluster
from .winners import get_winner_corpus, WinnerCorpus

logger = get_logger(__name__)


@dataclass
class ViralScore:
    """Result of viral likeness scoring."""
    score: int  # 0-100 overall viral likeness score
    max_similarity: float  # Highest similarity to any winner
    cluster_similarity: float  # Similarity to nearest cluster centroid
    nearest_winner: Optional[str] = None  # Most similar winner headline
    nearest_cluster_id: Optional[int] = None
    newsword_penalty: int = 0  # Penalty for newspaper-like language
    raw_embedding_score: float = 0.0

    @property
    def passes_threshold(self) -> bool:
        """Check if score passes minimum threshold."""
        settings = get_settings()
        return self.max_similarity >= settings.viral_sim_threshold or self.score >= 60


@dataclass
class ViralScorerState:
    """State for the viral scorer including embeddings and clusters."""
    winner_embeddings: list[list[float]] = field(default_factory=list)
    winner_headlines: list[str] = field(default_factory=list)
    cluster_centroids: list[list[float]] = field(default_factory=list)
    cluster_ids: list[int] = field(default_factory=list)
    initialized: bool = False
    last_refresh: Optional[datetime] = None


# Newspaper-like words that indicate admin/policy sludge
NEWSWORD_PATTERNS = [
    r'\bobjectives?\b',
    r'\bstakeholders?\b',
    r'\blocal\s+authorit(?:y|ies)\b',
    r'\bconsultation\b',
    r'\bstrategy\b',
    r'\bframework\b',
    r'\binitiative\b',
    r'\bprogramme\b',
    r'\bmust\s+meet\b',
    r'\bcompliance\b',
    r'\bprocurement\b',
    r'\bgovernance\b',
    r'\bcommissioning\b',
    r'\bpolicy\s+document\b',
    r'\baction\s+plan\b',
    r'\bkey\s+performance\b',
    r'\bworkstream\b',
    r'\bdeliverables?\b',
]

_NEWSWORD_PATTERNS_COMPILED = [re.compile(p, re.IGNORECASE) for p in NEWSWORD_PATTERNS]


def count_newswords(text: str) -> int:
    """Count newspaper-like words in text."""
    count = 0
    for pattern in _NEWSWORD_PATTERNS_COMPILED:
        count += len(pattern.findall(text))
    return count


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


class ViralScorer:
    """
    Scores candidate headlines by similarity to winner corpus.

    Uses OpenAI embeddings and k-means clustering to identify
    viral potential based on style similarity.
    """

    def __init__(self, n_clusters: int = 8):
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.embedding_model = self.settings.openai_embedding_model
        self.n_clusters = n_clusters
        self._state = ViralScorerState()

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for a single text."""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def _get_embeddings_batch(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """Get embeddings for multiple texts in batches."""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=batch,
            )
            batch_embeddings = [d.embedding for d in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _load_cached_embeddings(self) -> bool:
        """Load winner embeddings from database cache."""
        try:
            with get_session() as session:
                winners = session.query(WinnerHeadline).filter(
                    WinnerHeadline.embedding.isnot(None)
                ).all()

                if not winners:
                    return False

                self._state.winner_headlines = [w.headline for w in winners]
                self._state.winner_embeddings = [w.embedding for w in winners]
                self._state.cluster_ids = [w.cluster_id or 0 for w in winners]

                # Load cluster centroids
                clusters = session.query(WinnerCluster).order_by(
                    WinnerCluster.cluster_id
                ).all()

                if clusters:
                    self._state.cluster_centroids = [c.centroid for c in clusters]

                logger.info(
                    "viral_scorer_loaded_cache",
                    winner_count=len(winners),
                    cluster_count=len(clusters),
                )
                return True

        except Exception as e:
            logger.warning("viral_scorer_cache_load_failed", error=str(e))
            return False

    def _save_embeddings_to_cache(self) -> None:
        """Save computed embeddings to database cache."""
        try:
            with get_session() as session:
                # Update winner embeddings
                for i, headline in enumerate(self._state.winner_headlines):
                    winner = session.query(WinnerHeadline).filter(
                        WinnerHeadline.headline == headline
                    ).first()

                    if winner:
                        winner.embedding = self._state.winner_embeddings[i]
                        if i < len(self._state.cluster_ids):
                            winner.cluster_id = int(self._state.cluster_ids[i])

                # Save cluster centroids
                session.query(WinnerCluster).delete()

                for i, centroid in enumerate(self._state.cluster_centroids):
                    cluster_count = sum(1 for cid in self._state.cluster_ids if cid == i)
                    session.add(WinnerCluster(
                        cluster_id=i,
                        centroid=centroid,
                        headline_count=cluster_count,
                    ))

                session.commit()
                logger.info("viral_scorer_saved_cache")

        except Exception as e:
            logger.error("viral_scorer_cache_save_failed", error=str(e))

    def initialize(self, force_refresh: bool = False) -> None:
        """
        Initialize scorer with winner embeddings and clusters.

        Args:
            force_refresh: Force recomputation of embeddings
        """
        if self._state.initialized and not force_refresh:
            return

        # Try loading from cache first
        if not force_refresh and self._load_cached_embeddings():
            if self._state.winner_embeddings and self._state.cluster_centroids:
                self._state.initialized = True
                self._state.last_refresh = datetime.now(timezone.utc)
                return

        # Load winner corpus
        corpus = get_winner_corpus(force_refresh=force_refresh)

        if corpus.is_empty:
            logger.warning("viral_scorer_no_winners")
            self._state.initialized = True
            return

        logger.info("viral_scorer_computing_embeddings", count=corpus.count)

        # Compute embeddings for all winners
        self._state.winner_headlines = corpus.headlines
        self._state.winner_embeddings = self._get_embeddings_batch(corpus.headlines)

        # Cluster winners
        if len(self._state.winner_embeddings) >= self.n_clusters:
            self._cluster_winners()
        else:
            # Not enough winners for clustering, use all as single cluster
            self._state.cluster_centroids = [
                np.mean(self._state.winner_embeddings, axis=0).tolist()
            ]
            self._state.cluster_ids = [0] * len(self._state.winner_embeddings)

        # Save to cache
        self._save_embeddings_to_cache()

        self._state.initialized = True
        self._state.last_refresh = datetime.now(timezone.utc)

        logger.info(
            "viral_scorer_initialized",
            winner_count=len(self._state.winner_headlines),
            cluster_count=len(self._state.cluster_centroids),
        )

    def _cluster_winners(self) -> None:
        """Cluster winner embeddings using k-means."""
        embeddings_array = np.array(self._state.winner_embeddings)

        # Adjust n_clusters if we have fewer samples
        n_clusters = min(self.n_clusters, len(embeddings_array))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_array)

        self._state.cluster_ids = cluster_labels.tolist()
        self._state.cluster_centroids = kmeans.cluster_centers_.tolist()

        logger.info(
            "viral_scorer_clustered",
            n_clusters=n_clusters,
            cluster_sizes=[sum(1 for c in cluster_labels if c == i) for i in range(n_clusters)],
        )

    def score(self, text: str) -> ViralScore:
        """
        Score a candidate headline for viral likeness.

        Args:
            text: Candidate headline text

        Returns:
            ViralScore with similarity metrics and final score
        """
        if not self._state.initialized:
            self.initialize()

        # Handle empty corpus
        if not self._state.winner_embeddings:
            return ViralScore(
                score=50,  # Neutral score
                max_similarity=0.0,
                cluster_similarity=0.0,
            )

        # Get embedding for candidate
        candidate_embedding = self._get_embedding(text)

        # Compute similarity to all winners
        similarities = [
            cosine_similarity(candidate_embedding, winner_emb)
            for winner_emb in self._state.winner_embeddings
        ]

        max_sim_idx = np.argmax(similarities)
        max_similarity = similarities[max_sim_idx]
        nearest_winner = self._state.winner_headlines[max_sim_idx]

        # Compute similarity to cluster centroids
        cluster_similarities = [
            cosine_similarity(candidate_embedding, centroid)
            for centroid in self._state.cluster_centroids
        ]

        if cluster_similarities:
            max_cluster_idx = np.argmax(cluster_similarities)
            cluster_similarity = cluster_similarities[max_cluster_idx]
            nearest_cluster_id = max_cluster_idx
        else:
            cluster_similarity = max_similarity
            nearest_cluster_id = 0

        # Compute newsword penalty
        newsword_count = count_newswords(text)
        newsword_penalty = min(newsword_count * 10, 30)  # Max 30 point penalty

        # Compute raw embedding score (0-100)
        # Map similarity from ~0.6-1.0 to 0-100
        raw_embedding_score = max(0, min(100, (max_similarity - 0.6) * 250))

        # Final score combines similarity with penalty
        final_score = int(max(0, min(100, raw_embedding_score - newsword_penalty)))

        return ViralScore(
            score=final_score,
            max_similarity=max_similarity,
            cluster_similarity=cluster_similarity,
            nearest_winner=nearest_winner,
            nearest_cluster_id=nearest_cluster_id,
            newsword_penalty=newsword_penalty,
            raw_embedding_score=raw_embedding_score,
        )

    def score_batch(self, texts: list[str]) -> list[ViralScore]:
        """Score multiple candidate headlines."""
        return [self.score(text) for text in texts]

    def get_cluster_summary(self) -> dict:
        """Get summary of winner clusters."""
        if not self._state.initialized:
            self.initialize()

        summary = {}
        for i, centroid in enumerate(self._state.cluster_centroids):
            cluster_headlines = [
                h for j, h in enumerate(self._state.winner_headlines)
                if self._state.cluster_ids[j] == i
            ]
            summary[f"cluster_{i}"] = {
                "count": len(cluster_headlines),
                "examples": cluster_headlines[:3],
            }

        return summary


# Singleton scorer
_scorer: Optional[ViralScorer] = None


def get_viral_scorer() -> ViralScorer:
    """Get singleton viral scorer instance."""
    global _scorer
    if _scorer is None:
        _scorer = ViralScorer()
    return _scorer


def score_viral_likeness(text: str) -> ViralScore:
    """Convenience function to score a single text."""
    scorer = get_viral_scorer()
    return scorer.score(text)
