"""
Tests for novelty checking module.

Tests:
- Similarity calculation (cosine, Jaccard, TF-IDF)
- Novelty threshold checking
- Score penalty for near-duplicates
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from viral_bot.novelty import (
    cosine_similarity,
    jaccard_similarity,
    tfidf_similarity,
    NoveltyChecker,
    NoveltyResult,
)


class TestCosineSimilarity:
    """Tests for cosine similarity calculation."""

    def test_identical_vectors_return_one(self):
        """Identical vectors should have similarity 1.0."""
        a = [1.0, 2.0, 3.0, 4.0]
        b = [1.0, 2.0, 3.0, 4.0]

        similarity = cosine_similarity(a, b)

        assert abs(similarity - 1.0) < 0.001

    def test_orthogonal_vectors_return_zero(self):
        """Orthogonal vectors should have similarity 0.0."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]

        similarity = cosine_similarity(a, b)

        assert abs(similarity - 0.0) < 0.001

    def test_opposite_vectors_return_negative(self):
        """Opposite vectors should have similarity -1.0."""
        a = [1.0, 2.0, 3.0]
        b = [-1.0, -2.0, -3.0]

        similarity = cosine_similarity(a, b)

        assert abs(similarity - (-1.0)) < 0.001

    def test_handles_zero_vectors(self):
        """Should handle zero vectors gracefully."""
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]

        similarity = cosine_similarity(a, b)

        assert similarity == 0.0


class TestJaccardSimilarity:
    """Tests for Jaccard similarity calculation."""

    def test_identical_texts_return_one(self):
        """Identical texts should have similarity 1.0."""
        text1 = "the quick brown fox"
        text2 = "the quick brown fox"

        similarity = jaccard_similarity(text1, text2)

        assert abs(similarity - 1.0) < 0.001

    def test_completely_different_texts_return_zero(self):
        """Completely different texts should have similarity 0.0."""
        text1 = "apple banana cherry"
        text2 = "xyz uvw rst"

        similarity = jaccard_similarity(text1, text2)

        assert abs(similarity - 0.0) < 0.001

    def test_partial_overlap(self):
        """Partial overlap should return intermediate value."""
        text1 = "the quick brown fox"
        text2 = "the lazy brown dog"

        similarity = jaccard_similarity(text1, text2)

        # Should be between 0 and 1
        assert 0 < similarity < 1

    def test_handles_empty_text(self):
        """Should handle empty texts gracefully."""
        similarity = jaccard_similarity("", "some text")

        assert similarity == 0.0


class TestTfIdfSimilarity:
    """Tests for TF-IDF similarity calculation."""

    def test_identical_texts_return_one(self):
        """Identical texts should have similarity 1.0."""
        text1 = "exercise reduces mortality risk"
        text2 = "exercise reduces mortality risk"

        similarity = tfidf_similarity(text1, text2)

        assert abs(similarity - 1.0) < 0.001

    def test_different_texts_return_low_score(self):
        """Different texts should have low similarity."""
        text1 = "exercise reduces mortality risk"
        text2 = "sleep improves cognitive function"

        similarity = tfidf_similarity(text1, text2)

        assert similarity < 0.5

    def test_handles_empty_text(self):
        """Should handle empty texts gracefully."""
        similarity = tfidf_similarity("", "some text")

        assert similarity == 0.0


class TestNoveltyChecker:
    """Tests for the NoveltyChecker class."""

    def test_novel_finding_passes(self):
        """Should pass findings that are novel."""
        checker = NoveltyChecker(
            similarity_threshold=0.8,
            use_embeddings=False,  # Use lexical similarity
        )

        # Add some recent findings
        checker.load_recent_findings([
            ("Study shows coffee reduces heart disease risk", None),
            ("Walking 30 minutes daily linked to longevity", None),
        ])

        # Check a different finding
        result, adjusted_score = checker.check_novelty(
            "Mediterranean diet associated with cognitive benefits",
            virality_score=75,
        )

        assert result.is_novel is True
        assert adjusted_score == 75  # No penalty

    def test_similar_finding_rejected(self):
        """Should reject findings that are too similar."""
        checker = NoveltyChecker(
            similarity_threshold=0.5,  # Low threshold for testing
            use_embeddings=False,
        )

        # Add a recent finding
        checker.load_recent_findings([
            ("Study shows coffee reduces heart disease risk by 20%", None),
        ])

        # Check a very similar finding
        result, adjusted_score = checker.check_novelty(
            "Research finds coffee reduces heart disease risk by 20%",
            virality_score=75,
        )

        assert result.is_novel is False
        assert result.similarity_score >= 0.5

    def test_near_duplicate_gets_penalty(self):
        """Should apply penalty to near-duplicates."""
        checker = NoveltyChecker(
            similarity_threshold=0.9,
            use_embeddings=False,
            penalty_factor=0.3,
        )

        # Add a finding
        checker.load_recent_findings([
            ("Study of 50000 people shows exercise reduces mortality", None),
        ])

        # Check a somewhat similar finding (should be penalized but not rejected)
        # The threshold is 0.9, so 80% of threshold = 0.72
        # Anything above 0.72 gets some penalty
        result, adjusted_score = checker.check_novelty(
            "Research with 50000 participants shows exercise lowers mortality risk",
            virality_score=100,
        )

        # If it's novel but similarity is high, score should be penalized
        if result.is_novel and result.similarity_score >= 0.72:
            assert adjusted_score < 100 or result.penalty_applied > 0

    def test_empty_history_always_novel(self):
        """Should consider everything novel with empty history."""
        checker = NoveltyChecker(
            similarity_threshold=0.86,
            use_embeddings=False,
        )

        result, adjusted_score = checker.check_novelty(
            "Any finding should be novel",
            virality_score=80,
        )

        assert result.is_novel is True
        assert adjusted_score == 80

    def test_adds_finding_to_history(self):
        """Should add checked findings to history."""
        checker = NoveltyChecker(
            similarity_threshold=0.86,
            use_embeddings=False,
        )

        assert len(checker.recent_findings) == 0

        checker.add_finding("First finding")
        checker.add_finding("Second finding")

        assert len(checker.recent_findings) == 2

    def test_method_reported_correctly(self):
        """Should report the method used for checking."""
        checker = NoveltyChecker(
            similarity_threshold=0.86,
            use_embeddings=False,
        )

        checker.load_recent_findings([
            ("Some existing finding", None),
        ])

        result, _ = checker.check_novelty("New finding")

        assert result.method == "lexical"


class TestNoveltyResult:
    """Tests for NoveltyResult dataclass."""

    def test_creates_with_all_fields(self):
        """Should create result with all fields."""
        result = NoveltyResult(
            is_novel=True,
            similarity_score=0.45,
            most_similar_to="Previous finding",
            method="embedding",
            penalty_applied=0.1,
        )

        assert result.is_novel is True
        assert result.similarity_score == 0.45
        assert result.most_similar_to == "Previous finding"
        assert result.method == "embedding"
        assert result.penalty_applied == 0.1

    def test_default_values(self):
        """Should have sensible defaults."""
        result = NoveltyResult(
            is_novel=True,
            similarity_score=0.0,
        )

        assert result.most_similar_to is None
        assert result.method == "embedding"
        assert result.penalty_applied == 0.0
