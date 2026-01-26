"""
Anti-generic filtering module.

Rejects "gym poster" content (exercise good, sleep important, eat healthy)
unless it contains strong differentiators like specific numbers, populations,
or surprising findings.

Two-stage filtering:
1. Pre-AI gate: Cheap deterministic checks
2. Post-AI gate: Validates AI output has required differentiators
"""

import re
from dataclasses import dataclass
from typing import Optional

from .logging_conf import get_logger
from .normalize import NormalizedItem, ContentType

logger = get_logger(__name__)


# Generic truism patterns - reject if title matches AND no differentiators in body
GENERIC_TRUISM_PATTERNS = [
    # Exercise
    r"exercise\s+(?:is\s+)?(?:good|beneficial|important|helps?|improves?)\s+(?:for\s+)?(?:health|you)",
    r"regular\s+exercise\s+(?:can\s+)?(?:reduce|lower|cut)s?\s+(?:risk|mortality)",
    r"physical\s+activity\s+(?:is\s+)?(?:good|beneficial|important|linked)",
    r"(?:walking|running|jogging)\s+(?:is\s+)?good\s+for",

    # Sleep
    r"sleep\s+(?:is\s+)?(?:important|essential|crucial|vital)\s+(?:for\s+)?health",
    r"(?:good|better|more)\s+sleep\s+(?:is\s+)?(?:important|good|beneficial)",
    r"lack\s+of\s+sleep\s+(?:is\s+)?(?:bad|harmful|linked)",

    # Diet
    r"healthy\s+diet\s+(?:is\s+)?(?:important|good|beneficial|helps)",
    r"(?:eating|diet)\s+(?:healthy|well)\s+(?:is\s+)?(?:good|important)",
    r"(?:fruits?|vegetables?)\s+(?:are\s+)?good\s+for",
    r"(?:processed\s+food|junk\s+food|sugar)\s+(?:is\s+)?bad",

    # Stress
    r"stress\s+(?:is\s+)?(?:bad|harmful|linked\s+to)",
    r"(?:managing|reducing)\s+stress\s+(?:is\s+)?(?:important|good)",

    # General health
    r"healthy\s+lifestyle\s+(?:is\s+)?(?:important|good|beneficial)",
    r"(?:staying|being)\s+(?:active|healthy)\s+(?:is\s+)?(?:important|good)",
]

# Keywords that indicate potential differentiators
DIFFERENTIATOR_KEYWORDS = [
    # Numbers/percentages
    r"\d+(?:\.\d+)?\s*%",
    r"\d+(?:\.\d+)?-fold",
    r"\d+(?:\.\d+)?\s*(?:times|x)\s+(?:more|less|higher|lower)",
    r"(?:equivalent\s+to|same\s+as)\s+\d+",

    # Time/duration specific
    r"\d+\s*(?:minutes?|hours?|days?|weeks?|months?|years?)\s+(?:of|per)",
    r"(?:after|within|over)\s+\d+\s*(?:minutes?|hours?|days?|weeks?|months?|years?)",

    # Population specific
    r"(?:aged?|ages?)\s+\d+(?:\s*-\s*\d+)?",
    r"(?:adults?|seniors?|elderly|children|teenagers?|men|women)\s+(?:aged?|over|under)\s+\d+",
    r"\d+[,\d]*\s*(?:participants?|subjects?|patients?|people|individuals?)",

    # Study specifics
    r"(?:randomized|randomised)\s+(?:controlled\s+)?trial",
    r"meta-analysis",
    r"cohort\s+study",
    r"(?:follow-up|followed)\s+(?:for\s+)?\d+",

    # Specific effects/outcomes
    r"(?:reduces?|lowers?|cuts?|increases?)\s+(?:risk\s+)?(?:by\s+)?\d+",
    r"(?:mortality|death)\s+(?:risk\s+)?(?:by\s+)?\d+",
    r"(?:risk|chance|odds)\s+(?:of\s+)?(?:\w+\s+)?\d+",
]

# Compile patterns for efficiency
_TRUISM_PATTERNS = [re.compile(p, re.IGNORECASE) for p in GENERIC_TRUISM_PATTERNS]
_DIFFERENTIATOR_PATTERNS = [re.compile(p, re.IGNORECASE) for p in DIFFERENTIATOR_KEYWORDS]


@dataclass
class GenericCheckResult:
    """Result of generic content check."""
    passed: bool
    reason: Optional[str] = None
    matched_truism: Optional[str] = None
    differentiators_found: list[str] = None

    def __post_init__(self):
        if self.differentiators_found is None:
            self.differentiators_found = []


def check_pre_ai_generic(
    item: NormalizedItem,
    min_body_chars_paper: int = 600,
    min_body_chars_news: int = 1200,
) -> GenericCheckResult:
    """
    Pre-AI generic content gate (cheap, deterministic).

    Rejects items if:
    1. Body text is too short
    2. Title matches generic truism AND no differentiators in body

    Args:
        item: NormalizedItem to check
        min_body_chars_paper: Minimum chars for paper abstracts
        min_body_chars_news: Minimum chars for news articles

    Returns:
        GenericCheckResult with pass/fail and reason
    """
    # Check 1: Minimum body length
    min_chars = min_body_chars_paper if item.content_type == ContentType.PAPER else min_body_chars_news

    if item.body_length < min_chars:
        return GenericCheckResult(
            passed=False,
            reason=f"insufficient_content: {item.body_length} chars < {min_chars} required",
        )

    # Check 2: Generic truism in title without differentiators
    title_lower = item.title.lower()
    body_text = item.body_text.lower()
    full_text = f"{title_lower} {body_text}"

    # Check if title matches any truism pattern
    matched_truism = None
    for pattern in _TRUISM_PATTERNS:
        match = pattern.search(title_lower)
        if match:
            matched_truism = match.group(0)
            break

    if matched_truism:
        # If truism matched, require at least one differentiator
        differentiators = find_differentiators(full_text)

        if not differentiators:
            return GenericCheckResult(
                passed=False,
                reason=f"generic_truism_no_differentiator: '{matched_truism}'",
                matched_truism=matched_truism,
            )

        return GenericCheckResult(
            passed=True,
            reason=f"truism_with_differentiator: '{matched_truism}'",
            matched_truism=matched_truism,
            differentiators_found=differentiators,
        )

    # Find differentiators for logging (even if not required)
    differentiators = find_differentiators(full_text)

    return GenericCheckResult(
        passed=True,
        reason="no_truism_match",
        differentiators_found=differentiators,
    )


def find_differentiators(text: str) -> list[str]:
    """
    Find differentiator phrases in text.

    Args:
        text: Text to search

    Returns:
        List of found differentiator matches
    """
    found = []
    for pattern in _DIFFERENTIATOR_PATTERNS:
        matches = pattern.findall(text)
        found.extend(matches)
    return found[:10]  # Limit to first 10


@dataclass
class DifferentiatorRequirements:
    """Required differentiators for post-AI validation."""
    has_number: bool = False
    has_population: bool = False
    has_time_window: bool = False
    has_comparison: bool = False
    number_value: Optional[str] = None
    population_value: Optional[str] = None
    time_window_value: Optional[str] = None
    comparison_value: Optional[str] = None

    @property
    def count(self) -> int:
        """Count of differentiators present."""
        return sum([
            self.has_number,
            self.has_population,
            self.has_time_window,
            self.has_comparison,
        ])

    @property
    def meets_minimum(self) -> bool:
        """Check if minimum requirements are met (at least 2 differentiators)."""
        return self.count >= 2


def check_post_ai_differentiators(
    evaluation_result: dict,
    min_differentiators: int = 2,
) -> tuple[bool, Optional[str], DifferentiatorRequirements]:
    """
    Post-AI generic gate - validate that AI output has required differentiators.

    Requires at least min_differentiators from:
    - A number (%, X-fold, "equivalent to", minutes/day, years, etc.)
    - Specific population (age range, condition, cohort)
    - Time window (weeks, years, follow-up)
    - Strong comparison (equivalent to smoking X/day, etc.)

    Args:
        evaluation_result: AI evaluation result dict
        min_differentiators: Minimum required (default 2)

    Returns:
        Tuple of (passed, reason, requirements)
    """
    reqs = DifferentiatorRequirements()

    # Extract fields from evaluation
    most_surprising = evaluation_result.get("most_surprising_finding", "")
    must_include_numbers = evaluation_result.get("must_include_numbers", [])
    population = evaluation_result.get("population")
    time_window = evaluation_result.get("time_window")
    sample_size = evaluation_result.get("sample_size")

    # Check for numbers
    if must_include_numbers:
        reqs.has_number = True
        reqs.number_value = ", ".join(must_include_numbers[:3])
    elif most_surprising:
        # Look for numbers in the surprising finding
        number_patterns = [
            r"\d+(?:\.\d+)?\s*%",
            r"\d+(?:\.\d+)?-fold",
            r"\d+(?:\.\d+)?\s*(?:times|x)\s+(?:more|less|higher|lower)",
        ]
        for pattern in number_patterns:
            match = re.search(pattern, most_surprising, re.IGNORECASE)
            if match:
                reqs.has_number = True
                reqs.number_value = match.group(0)
                break

    # Check for population
    if population and len(str(population)) > 3:
        reqs.has_population = True
        reqs.population_value = str(population)
    elif sample_size:
        # Sample size counts as population info
        reqs.has_population = True
        reqs.population_value = f"n={sample_size}"

    # Check for time window
    if time_window and len(str(time_window)) > 3:
        reqs.has_time_window = True
        reqs.time_window_value = str(time_window)
    elif most_surprising:
        time_patterns = [
            r"(\d+\s*(?:years?|months?|weeks?)\s+(?:of\s+)?(?:follow-?up|study|period))",
            r"(over\s+\d+\s*(?:years?|months?|weeks?))",
            r"(after\s+\d+\s*(?:years?|months?|weeks?))",
        ]
        for pattern in time_patterns:
            match = re.search(pattern, most_surprising, re.IGNORECASE)
            if match:
                reqs.has_time_window = True
                reqs.time_window_value = match.group(1)
                break

    # Check for comparison
    if most_surprising:
        comparison_patterns = [
            r"(equivalent\s+to\s+(?:smoking|drinking|eating|[\w\s]+)\s+[\d\w]+)",
            r"(same\s+(?:effect|benefit|risk)\s+as\s+[\w\s]+)",
            r"(compared\s+to\s+[\w\s]+,?\s+\d+[%x])",
            r"(like\s+(?:smoking|drinking|eating)\s+[\d\w]+)",
        ]
        for pattern in comparison_patterns:
            match = re.search(pattern, most_surprising, re.IGNORECASE)
            if match:
                reqs.has_comparison = True
                reqs.comparison_value = match.group(1)
                break

    # Determine if passed
    passed = reqs.count >= min_differentiators

    if not passed:
        reason = f"generic/unspecific: only {reqs.count}/{min_differentiators} differentiators"
        return False, reason, reqs

    return True, None, reqs


def format_differentiator_summary(reqs: DifferentiatorRequirements) -> str:
    """Format a summary of found differentiators for logging."""
    parts = []
    if reqs.has_number:
        parts.append(f"number: {reqs.number_value}")
    if reqs.has_population:
        parts.append(f"population: {reqs.population_value}")
    if reqs.has_time_window:
        parts.append(f"time: {reqs.time_window_value}")
    if reqs.has_comparison:
        parts.append(f"comparison: {reqs.comparison_value}")
    return " | ".join(parts) if parts else "none"


class GenericFilter:
    """
    Combined generic content filter with statistics tracking.
    """

    def __init__(
        self,
        min_body_chars_paper: int = 600,
        min_body_chars_news: int = 1200,
        min_differentiators: int = 2,
    ):
        """
        Initialize filter.

        Args:
            min_body_chars_paper: Min chars for paper abstracts
            min_body_chars_news: Min chars for news articles
            min_differentiators: Min differentiators for post-AI gate
        """
        self.min_body_chars_paper = min_body_chars_paper
        self.min_body_chars_news = min_body_chars_news
        self.min_differentiators = min_differentiators

        # Statistics
        self.stats = {
            "pre_ai_total": 0,
            "pre_ai_passed": 0,
            "pre_ai_rejected_insufficient": 0,
            "pre_ai_rejected_truism": 0,
            "post_ai_total": 0,
            "post_ai_passed": 0,
            "post_ai_rejected": 0,
        }
        self.rejection_examples = []

    def pre_ai_filter(self, item: NormalizedItem) -> GenericCheckResult:
        """
        Apply pre-AI generic filter.

        Args:
            item: Item to check

        Returns:
            GenericCheckResult
        """
        self.stats["pre_ai_total"] += 1

        result = check_pre_ai_generic(
            item,
            self.min_body_chars_paper,
            self.min_body_chars_news,
        )

        if result.passed:
            self.stats["pre_ai_passed"] += 1
        else:
            if "insufficient" in (result.reason or ""):
                self.stats["pre_ai_rejected_insufficient"] += 1
            else:
                self.stats["pre_ai_rejected_truism"] += 1

            # Store rejection example
            if len(self.rejection_examples) < 10:
                self.rejection_examples.append({
                    "stage": "pre_ai",
                    "title": item.title[:100],
                    "reason": result.reason,
                })

            logger.debug(
                "pre_ai_rejected",
                title=item.title[:60],
                reason=result.reason,
            )

        return result

    def post_ai_filter(
        self,
        evaluation_result: dict,
        item_title: str = "",
    ) -> tuple[bool, Optional[str], DifferentiatorRequirements]:
        """
        Apply post-AI differentiator filter.

        Args:
            evaluation_result: AI evaluation result dict
            item_title: Item title for logging

        Returns:
            Tuple of (passed, reason, requirements)
        """
        self.stats["post_ai_total"] += 1

        passed, reason, reqs = check_post_ai_differentiators(
            evaluation_result,
            self.min_differentiators,
        )

        if passed:
            self.stats["post_ai_passed"] += 1
        else:
            self.stats["post_ai_rejected"] += 1

            # Store rejection example
            if len(self.rejection_examples) < 10:
                self.rejection_examples.append({
                    "stage": "post_ai",
                    "title": item_title[:100],
                    "reason": reason,
                    "differentiators": format_differentiator_summary(reqs),
                })

            logger.debug(
                "post_ai_rejected",
                title=item_title[:60],
                reason=reason,
                differentiators=format_differentiator_summary(reqs),
            )

        return passed, reason, reqs

    def get_stats(self) -> dict:
        """Get filtering statistics."""
        return {
            **self.stats,
            "rejection_examples": self.rejection_examples,
        }

    def get_rejection_summary(self) -> str:
        """Get a formatted summary of rejections."""
        lines = [
            f"Pre-AI: {self.stats['pre_ai_passed']}/{self.stats['pre_ai_total']} passed",
            f"  - Insufficient content: {self.stats['pre_ai_rejected_insufficient']}",
            f"  - Generic truism: {self.stats['pre_ai_rejected_truism']}",
            f"Post-AI: {self.stats['post_ai_passed']}/{self.stats['post_ai_total']} passed",
            f"  - Rejected (< {self.min_differentiators} differentiators): {self.stats['post_ai_rejected']}",
        ]
        return "\n".join(lines)
