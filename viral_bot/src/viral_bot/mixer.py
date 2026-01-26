"""
Mixing controller for output diversity.

Ensures variety in generated outputs by:
- Limiting STUDY_STAT to max 2 per run
- Requiring at least 1 non-STUDY_STAT when available
- Balancing archetypes across runs
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from .logging_conf import get_logger
from .openai_eval import EvaluationResult

logger = get_logger(__name__)


@dataclass
class MixingConfig:
    """Configuration for output mixing."""
    max_study_stat: int = 2  # Max STUDY_STAT outputs per run
    min_non_study_stat: int = 1  # Min non-STUDY_STAT when available
    max_per_archetype: int = 2  # Max of any single archetype
    preferred_archetypes: list[str] = None  # Archetypes to prefer

    def __post_init__(self):
        if self.preferred_archetypes is None:
            # Prefer diverse archetypes
            self.preferred_archetypes = [
                "COUNTERINTUITIVE",
                "WARNING_RISK",
                "NEWS_POLICY",
                "SIMPLE_HABIT",
            ]


class OutputMixer:
    """
    Selects diverse outputs from evaluated candidates.

    Prevents 15 STUDY_STAT in a row by enforcing archetype limits
    and preferring variety.
    """

    def __init__(self, config: Optional[MixingConfig] = None):
        """
        Initialize mixer.

        Args:
            config: Mixing configuration (uses defaults if not provided)
        """
        self.config = config or MixingConfig()
        self.archetype_counts = defaultdict(int)

    def select_outputs(
        self,
        candidates: list[tuple],
        max_outputs: int = 5,
    ) -> list[tuple]:
        """
        Select diverse outputs from candidates.

        Args:
            candidates: List of (item, evaluation, ...) tuples, sorted by score
            max_outputs: Maximum outputs to select

        Returns:
            Selected outputs with enforced diversity
        """
        if not candidates:
            return []

        selected = []
        study_stat_count = 0
        non_study_stat_count = 0
        archetype_counts = defaultdict(int)

        # First pass: categorize candidates
        study_stat_candidates = []
        non_study_stat_candidates = []

        for candidate in candidates:
            eval_result = candidate[1]  # Second element is EvaluationResult
            archetype = eval_result.suggested_archetype or "STUDY_STAT"

            if archetype == "STUDY_STAT":
                study_stat_candidates.append(candidate)
            else:
                non_study_stat_candidates.append(candidate)

        logger.debug(
            "mixer_candidates",
            study_stat=len(study_stat_candidates),
            non_study_stat=len(non_study_stat_candidates),
        )

        # Strategy: Interleave non-STUDY_STAT with STUDY_STAT
        # to ensure diversity

        # Add non-STUDY_STAT first (up to a point)
        for candidate in non_study_stat_candidates:
            if len(selected) >= max_outputs:
                break

            eval_result = candidate[1]
            archetype = eval_result.suggested_archetype or "OTHER"

            # Check archetype limit
            if archetype_counts[archetype] >= self.config.max_per_archetype:
                continue

            selected.append(candidate)
            archetype_counts[archetype] += 1
            non_study_stat_count += 1

            # After getting min non-STUDY_STAT, start mixing in STUDY_STAT
            if non_study_stat_count >= self.config.min_non_study_stat:
                # Add a STUDY_STAT if we have room
                if study_stat_candidates and study_stat_count < self.config.max_study_stat:
                    if len(selected) < max_outputs:
                        ss_candidate = study_stat_candidates.pop(0)
                        selected.append(ss_candidate)
                        archetype_counts["STUDY_STAT"] += 1
                        study_stat_count += 1

        # Fill remaining slots with STUDY_STAT (up to limit)
        while (
            len(selected) < max_outputs
            and study_stat_candidates
            and study_stat_count < self.config.max_study_stat
        ):
            candidate = study_stat_candidates.pop(0)
            selected.append(candidate)
            archetype_counts["STUDY_STAT"] += 1
            study_stat_count += 1

        # If still need more, add remaining non-STUDY_STAT
        for candidate in non_study_stat_candidates:
            if len(selected) >= max_outputs:
                break

            if candidate not in selected:
                eval_result = candidate[1]
                archetype = eval_result.suggested_archetype or "OTHER"

                if archetype_counts[archetype] < self.config.max_per_archetype:
                    selected.append(candidate)
                    archetype_counts[archetype] += 1

        # If STILL need more and have more STUDY_STAT (exceeding limit)
        # only add if nothing else available
        while len(selected) < max_outputs and study_stat_candidates:
            candidate = study_stat_candidates.pop(0)
            selected.append(candidate)
            archetype_counts["STUDY_STAT"] += 1
            study_stat_count += 1

        # Re-sort by virality score to maintain quality ranking
        selected.sort(key=lambda x: x[1].virality_score or 0, reverse=True)

        # Update running counts
        self.archetype_counts = archetype_counts

        logger.info(
            "mixer_selected",
            total_candidates=len(candidates),
            selected=len(selected),
            archetype_distribution=dict(archetype_counts),
        )

        return selected

    def get_archetype_summary(self) -> dict:
        """Get summary of archetype distribution in last selection."""
        return dict(self.archetype_counts)


def select_diverse_outputs(
    candidates: list[tuple],
    max_outputs: int = 5,
    max_study_stat: int = 2,
    min_non_study_stat: int = 1,
) -> list[tuple]:
    """
    Convenience function to select diverse outputs.

    Args:
        candidates: List of (item, evaluation, ...) tuples
        max_outputs: Maximum outputs to select
        max_study_stat: Maximum STUDY_STAT outputs
        min_non_study_stat: Minimum non-STUDY_STAT outputs

    Returns:
        Selected outputs with enforced diversity
    """
    config = MixingConfig(
        max_study_stat=max_study_stat,
        min_non_study_stat=min_non_study_stat,
    )
    mixer = OutputMixer(config)
    return mixer.select_outputs(candidates, max_outputs)
