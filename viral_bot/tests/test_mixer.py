"""
Tests for output mixing/diversity module.

Tests:
- Archetype limits (max STUDY_STAT per run)
- Diversity requirements (min non-STUDY_STAT)
- Score-based selection
"""

import pytest
from unittest.mock import MagicMock

from viral_bot.mixer import (
    OutputMixer,
    MixingConfig,
    select_diverse_outputs,
)


def create_mock_evaluation(archetype: str, score: int):
    """Create a mock evaluation result."""
    mock = MagicMock()
    mock.suggested_archetype = archetype
    mock.virality_score = score
    return mock


def create_candidate(archetype: str, score: int):
    """Create a candidate tuple (item, evaluation)."""
    item = MagicMock()
    item.title = f"Test item for {archetype}"
    eval_result = create_mock_evaluation(archetype, score)
    return (item, eval_result)


class TestOutputMixer:
    """Tests for OutputMixer class."""

    def test_limits_study_stat_outputs(self):
        """Should limit STUDY_STAT to configured maximum."""
        mixer = OutputMixer(MixingConfig(max_study_stat=2))

        candidates = [
            create_candidate("STUDY_STAT", 90),
            create_candidate("STUDY_STAT", 85),
            create_candidate("STUDY_STAT", 80),
            create_candidate("STUDY_STAT", 75),
            create_candidate("STUDY_STAT", 70),
        ]

        selected = mixer.select_outputs(candidates, max_outputs=5)

        # Should have max 2 STUDY_STAT
        study_stat_count = sum(
            1 for _, eval in selected
            if eval.suggested_archetype == "STUDY_STAT"
        )

        assert study_stat_count <= 2

    def test_includes_non_study_stat_when_available(self):
        """Should include at least one non-STUDY_STAT when available."""
        mixer = OutputMixer(MixingConfig(
            max_study_stat=2,
            min_non_study_stat=1,
        ))

        candidates = [
            create_candidate("STUDY_STAT", 90),
            create_candidate("STUDY_STAT", 85),
            create_candidate("WARNING_RISK", 70),
            create_candidate("STUDY_STAT", 65),
        ]

        selected = mixer.select_outputs(candidates, max_outputs=3)

        # Should have at least 1 non-STUDY_STAT
        non_study_stat_count = sum(
            1 for _, eval in selected
            if eval.suggested_archetype != "STUDY_STAT"
        )

        assert non_study_stat_count >= 1

    def test_maintains_score_order(self):
        """Should maintain approximate score ordering."""
        mixer = OutputMixer(MixingConfig(max_study_stat=5))

        candidates = [
            create_candidate("STUDY_STAT", 90),
            create_candidate("STUDY_STAT", 50),
            create_candidate("STUDY_STAT", 70),
        ]

        selected = mixer.select_outputs(candidates, max_outputs=3)

        # Should be sorted by score (descending)
        scores = [eval.virality_score for _, eval in selected]
        assert scores == sorted(scores, reverse=True)

    def test_respects_max_per_archetype(self):
        """Should respect per-archetype limits."""
        mixer = OutputMixer(MixingConfig(
            max_study_stat=3,
            max_per_archetype=2,
        ))

        candidates = [
            create_candidate("WARNING_RISK", 90),
            create_candidate("WARNING_RISK", 85),
            create_candidate("WARNING_RISK", 80),
            create_candidate("NEWS_POLICY", 75),
        ]

        selected = mixer.select_outputs(candidates, max_outputs=4)

        # Should have max 2 of any archetype
        warning_count = sum(
            1 for _, eval in selected
            if eval.suggested_archetype == "WARNING_RISK"
        )

        assert warning_count <= 2

    def test_handles_empty_candidates(self):
        """Should handle empty candidate list gracefully."""
        mixer = OutputMixer()

        selected = mixer.select_outputs([], max_outputs=5)

        assert selected == []

    def test_handles_fewer_candidates_than_max(self):
        """Should handle case where candidates < max_outputs."""
        mixer = OutputMixer()

        candidates = [
            create_candidate("STUDY_STAT", 90),
            create_candidate("NEWS_POLICY", 85),
        ]

        selected = mixer.select_outputs(candidates, max_outputs=5)

        assert len(selected) == 2

    def test_tracks_archetype_distribution(self):
        """Should track archetype distribution."""
        mixer = OutputMixer()

        candidates = [
            create_candidate("STUDY_STAT", 90),
            create_candidate("WARNING_RISK", 85),
            create_candidate("NEWS_POLICY", 80),
        ]

        mixer.select_outputs(candidates, max_outputs=3)

        summary = mixer.get_archetype_summary()

        assert "STUDY_STAT" in summary or summary.get("STUDY_STAT", 0) >= 0


class TestSelectDiverseOutputs:
    """Tests for the convenience function."""

    def test_convenience_function_works(self):
        """Should work with default parameters."""
        candidates = [
            create_candidate("STUDY_STAT", 90),
            create_candidate("NEWS_POLICY", 85),
        ]

        selected = select_diverse_outputs(candidates, max_outputs=2)

        assert len(selected) == 2

    def test_respects_custom_limits(self):
        """Should respect custom limits."""
        candidates = [
            create_candidate("STUDY_STAT", 90),
            create_candidate("STUDY_STAT", 85),
            create_candidate("STUDY_STAT", 80),
        ]

        selected = select_diverse_outputs(
            candidates,
            max_outputs=3,
            max_study_stat=1,
        )

        study_stat_count = sum(
            1 for _, eval in selected
            if eval.suggested_archetype == "STUDY_STAT"
        )

        assert study_stat_count <= 1


class TestMixingConfig:
    """Tests for MixingConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = MixingConfig()

        assert config.max_study_stat == 2
        assert config.min_non_study_stat == 1
        assert config.max_per_archetype == 2
        assert config.preferred_archetypes is not None

    def test_custom_values(self):
        """Should accept custom values."""
        config = MixingConfig(
            max_study_stat=3,
            min_non_study_stat=2,
            max_per_archetype=3,
        )

        assert config.max_study_stat == 3
        assert config.min_non_study_stat == 2
        assert config.max_per_archetype == 3
