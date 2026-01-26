"""
Tests for anti-generic filtering module.

Tests:
- Generic truism detection
- Differentiator requirement logic
- Pre-AI and post-AI gates
"""

import pytest
from datetime import datetime, timezone

from viral_bot.normalize import NormalizedItem, ContentType
from viral_bot.anti_generic import (
    check_pre_ai_generic,
    check_post_ai_differentiators,
    find_differentiators,
    GenericFilter,
    format_differentiator_summary,
)


class TestGenericTruismDetection:
    """Tests for generic truism pattern matching."""

    def test_rejects_generic_exercise_claim_without_numbers(self):
        """Should reject 'exercise is good for health' without specifics."""
        item = NormalizedItem(
            title="Regular exercise is good for your health",
            body_text="Exercise is beneficial for health. Everyone should exercise more. "
                     "Physical activity improves overall wellbeing and quality of life.",
            content_type=ContentType.NEWS,
        )

        result = check_pre_ai_generic(item, min_body_chars_paper=100, min_body_chars_news=100)

        assert result.passed is False
        assert "generic_truism" in result.reason.lower() or "truism" in result.reason.lower()

    def test_accepts_exercise_claim_with_specific_numbers(self):
        """Should accept exercise claim with specific numbers."""
        item = NormalizedItem(
            title="Regular exercise reduces mortality risk",
            body_text="A study of 50,000 participants found that 30 minutes of daily walking "
                     "reduces all-cause mortality by 27% over a 10-year follow-up period. "
                     "Adults aged 50-70 showed the greatest benefit.",
            content_type=ContentType.NEWS,
        )

        result = check_pre_ai_generic(item, min_body_chars_paper=100, min_body_chars_news=100)

        assert result.passed is True
        assert len(result.differentiators_found) > 0

    def test_rejects_generic_sleep_claim(self):
        """Should reject generic sleep importance claim."""
        item = NormalizedItem(
            title="Sleep is important for health",
            body_text="Getting good sleep is essential for health. Sleep helps the body recover. "
                     "Everyone should try to sleep better.",
            content_type=ContentType.NEWS,
        )

        result = check_pre_ai_generic(item, min_body_chars_paper=100, min_body_chars_news=100)

        assert result.passed is False

    def test_accepts_specific_sleep_study(self):
        """Should accept specific sleep study with numbers."""
        item = NormalizedItem(
            title="Sleep study reveals surprising finding",
            body_text="Adults who slept 7-8 hours nightly had a 23% lower risk of cardiovascular "
                     "disease compared to those sleeping less than 6 hours, according to a "
                     "15-year cohort study of 123,000 participants.",
            content_type=ContentType.NEWS,
        )

        result = check_pre_ai_generic(item, min_body_chars_paper=100, min_body_chars_news=100)

        assert result.passed is True

    def test_rejects_generic_diet_claim(self):
        """Should reject generic diet claim."""
        item = NormalizedItem(
            title="Eating healthy is important",
            body_text="A healthy diet is good for you. Eating fruits and vegetables is beneficial. "
                     "Everyone should eat healthier.",
            content_type=ContentType.NEWS,
        )

        result = check_pre_ai_generic(item, min_body_chars_paper=100, min_body_chars_news=100)

        assert result.passed is False


class TestInsufficientContent:
    """Tests for minimum content length requirements."""

    def test_rejects_short_paper_abstract(self):
        """Should reject papers with abstracts < 600 chars."""
        item = NormalizedItem(
            title="Important research finding",
            body_text="This is a short abstract.",  # Too short
            content_type=ContentType.PAPER,
        )

        result = check_pre_ai_generic(item, min_body_chars_paper=600, min_body_chars_news=1200)

        assert result.passed is False
        assert "insufficient_content" in result.reason

    def test_rejects_short_news_article(self):
        """Should reject news with body < 1200 chars."""
        item = NormalizedItem(
            title="Breaking health news",
            body_text="Short news article body with not much content." * 10,  # ~450 chars
            content_type=ContentType.NEWS,
        )

        result = check_pre_ai_generic(item, min_body_chars_paper=600, min_body_chars_news=1200)

        assert result.passed is False
        assert "insufficient_content" in result.reason

    def test_accepts_sufficient_paper_abstract(self):
        """Should accept papers with abstracts >= 600 chars."""
        item = NormalizedItem(
            title="Meta-analysis of cardiovascular outcomes",
            body_text="This comprehensive meta-analysis examined cardiovascular outcomes " * 50,
            content_type=ContentType.PAPER,
        )

        result = check_pre_ai_generic(item, min_body_chars_paper=600, min_body_chars_news=1200)

        # Should pass content length check (may still fail truism check if applicable)
        assert "insufficient_content" not in (result.reason or "")


class TestDifferentiatorExtraction:
    """Tests for finding differentiators in text."""

    def test_finds_percentage(self):
        """Should find percentage values."""
        text = "The treatment reduced risk by 27%"
        differentiators = find_differentiators(text)

        assert any("27%" in d for d in differentiators)

    def test_finds_fold_increase(self):
        """Should find X-fold values."""
        text = "Showed a 3.5-fold increase in biomarker levels"
        differentiators = find_differentiators(text)

        assert any("fold" in d.lower() for d in differentiators)

    def test_finds_population_age(self):
        """Should find population age ranges."""
        text = "Adults aged 50-70 were included in the study"
        differentiators = find_differentiators(text)

        assert any("50" in d or "70" in d for d in differentiators)

    def test_finds_sample_size(self):
        """Should find sample sizes."""
        text = "Study included 10,000 participants over 5 years"
        differentiators = find_differentiators(text)

        assert len(differentiators) > 0

    def test_finds_time_duration(self):
        """Should find time durations."""
        text = "After 10 years of follow-up, researchers found significant differences"
        differentiators = find_differentiators(text)

        assert len(differentiators) > 0


class TestPostAIDifferentiators:
    """Tests for post-AI differentiator validation."""

    def test_passes_with_multiple_differentiators(self):
        """Should pass with >= 2 differentiators."""
        evaluation = {
            "most_surprising_finding": "27% reduction in mortality over 10 years",
            "must_include_numbers": ["27%", "10 years"],
            "population": "adults 50-70",
            "time_window": "10-year follow-up",
        }

        passed, reason, reqs = check_post_ai_differentiators(evaluation, min_differentiators=2)

        assert passed is True
        assert reqs.count >= 2

    def test_fails_with_insufficient_differentiators(self):
        """Should fail with < 2 differentiators."""
        evaluation = {
            "most_surprising_finding": "Exercise reduces mortality",
            "must_include_numbers": [],
            "population": None,
            "time_window": None,
        }

        passed, reason, reqs = check_post_ai_differentiators(evaluation, min_differentiators=2)

        assert passed is False
        assert "generic/unspecific" in reason

    def test_numbers_count_as_differentiator(self):
        """Numbers should count as a differentiator."""
        evaluation = {
            "most_surprising_finding": "Study found 35% improvement",
            "must_include_numbers": ["35%"],
            "population": None,
            "time_window": None,
        }

        passed, reason, reqs = check_post_ai_differentiators(evaluation, min_differentiators=1)

        assert reqs.has_number is True
        assert reqs.count >= 1

    def test_population_counts_as_differentiator(self):
        """Population should count as a differentiator."""
        evaluation = {
            "most_surprising_finding": "Finding in specific group",
            "must_include_numbers": [],
            "population": "adults over 65 with hypertension",
            "time_window": None,
        }

        passed, reason, reqs = check_post_ai_differentiators(evaluation, min_differentiators=1)

        assert reqs.has_population is True

    def test_time_window_counts_as_differentiator(self):
        """Time window should count as a differentiator."""
        evaluation = {
            "most_surprising_finding": "Long-term study finding",
            "must_include_numbers": [],
            "population": None,
            "time_window": "15-year follow-up",
        }

        passed, reason, reqs = check_post_ai_differentiators(evaluation, min_differentiators=1)

        assert reqs.has_time_window is True


class TestGenericFilter:
    """Tests for the GenericFilter class."""

    def test_tracks_statistics(self):
        """Should track filtering statistics."""
        filter = GenericFilter(
            min_body_chars_paper=100,
            min_body_chars_news=100,
            min_differentiators=2,
        )

        # Filter a few items
        items = [
            NormalizedItem(
                title="Generic health tip",
                body_text="Exercise is good for health. " * 20,
                content_type=ContentType.NEWS,
            ),
            NormalizedItem(
                title="Specific study finding",
                body_text="Study of 50,000 participants found 27% reduction over 10 years. " * 20,
                content_type=ContentType.NEWS,
            ),
        ]

        for item in items:
            filter.pre_ai_filter(item)

        stats = filter.get_stats()

        assert stats["pre_ai_total"] == 2
        assert stats["pre_ai_passed"] + stats["pre_ai_rejected_truism"] + stats["pre_ai_rejected_insufficient"] == 2


class TestDifferentiatorSummary:
    """Tests for differentiator summary formatting."""

    def test_formats_summary_correctly(self):
        """Should format a readable summary."""
        from viral_bot.anti_generic import DifferentiatorRequirements

        reqs = DifferentiatorRequirements(
            has_number=True,
            number_value="27%",
            has_population=True,
            population_value="adults 50-70",
            has_time_window=False,
            has_comparison=False,
        )

        summary = format_differentiator_summary(reqs)

        assert "27%" in summary
        assert "adults 50-70" in summary

    def test_returns_none_for_empty(self):
        """Should return 'none' for no differentiators."""
        from viral_bot.anti_generic import DifferentiatorRequirements

        reqs = DifferentiatorRequirements()
        summary = format_differentiator_summary(reqs)

        assert summary == "none"
