"""
Anti-generic filtering module for story-compressed headlines.

Rejects "gym poster" content (exercise good, sleep important, eat healthy)
unless it contains strong differentiators like specific numbers, populations,
or surprising findings.

Three-stage filtering:
1. Pre-AI gate: Cheap deterministic checks
2. Narrative quality gate: Validates spine extraction quality
3. Post-AI gate: Validates headline has required differentiators

Quality Requirements:
- standalone_clarity_score >= 7
- emotional_hook != "none"
- No admin policy sludge (budget approvals, org changes)
- Output diversity: max 2 STUDY_STAT per run
"""

import re
from dataclasses import dataclass
from typing import Optional

from typing import TYPE_CHECKING

from .logging_conf import get_logger
from .normalize import NormalizedItem, ContentType

if TYPE_CHECKING:
    from .narrative_extractor import NarrativeSpine

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


# =====================================================
# ADMIN SLUDGE DETECTION
# =====================================================

# Keywords that indicate administrative/policy sludge (boring, not health impact)
ADMIN_SLUDGE_PATTERNS = [
    r"budget\s+(?:approval|increase|cut|allocation)",
    r"(?:appointed|appoints?|appointment)\s+(?:to|as|new)",
    r"(?:resigns?|resignation|stepping\s+down)",
    r"(?:merger|acquisition|reorganization|restructuring)",
    r"committee\s+(?:meeting|formation|review)",
    r"(?:board|council)\s+(?:approves?|votes?|elects?)",
    r"funding\s+(?:round|announcement|cut)",
    r"office\s+(?:opens?|closes?|moves?|relocat)",
    r"partnership\s+(?:announced|signed|formed)",
    r"strategic\s+(?:plan|initiative|review)",
    r"(?:quarterly|annual)\s+(?:report|meeting|review)",
    r"leadership\s+(?:change|transition|announcement)",
    r"organizational\s+(?:change|update|news)",
]

_ADMIN_SLUDGE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in ADMIN_SLUDGE_PATTERNS]

# Spending/lifestyle trend patterns (NOT actionable health advice)
LIFESTYLE_TREND_PATTERNS = [
    r"(?:spending|spend|spent)\s+(?:up\s+to\s+)?[£$€]\d+",  # "spending £2,000"
    r"(?:millennials?|gen\s*z|boomers?|generation)\s+(?:are\s+)?(?:spending|prioritiz)",  # "Gen Z spending"
    r"(?:prioritiz(?:e|ing|es?))\s+(?:\w+\s+)?over\s+",  # "prioritizing X over Y"
    r"(?:fitness|wellness)\s+(?:industry|market|trend|boom)",  # industry trends
    r"(?:how\s+much|average)\s+(?:\w+\s+)?(?:spend|cost)",  # "how much people spend"
    r"(?:per\s+event|per\s+session|per\s+class|per\s+month)\s*[£$€]?\d+",  # cost per event
    r"(?:survey|poll)\s+(?:finds?|shows?|reveals?)\s+(?:\w+\s+)?(?:spending|prefer)",  # survey spending
    r"hyrox",  # Specific: Hyrox events (fitness competition spending)
    r"(?:boutique|premium|luxury)\s+(?:fitness|gym|wellness)",  # premium fitness trends
    r"(?:fitness|gym)\s+(?:membership|subscription)\s+(?:cost|price|fee)",  # membership costs
]

_LIFESTYLE_TREND_PATTERNS = [re.compile(p, re.IGNORECASE) for p in LIFESTYLE_TREND_PATTERNS]

# Product recall / too-specific patterns (not general health advice)
TOO_SPECIFIC_PATTERNS = [
    # Product recalls with specific details
    r"(?:stop\s+using|recall(?:ed)?|withdraw[n]?)\s+\w+.*(?:batch|lot|expir)",
    r"(?:batch|lot)\s*(?:number|#|no\.?)?\s*[A-Z0-9-]+",
    r"expir(?:y|ing|es?)\s+\d{1,2}[-/]\d{1,2}[-/]\d{2,4}",
    r"\d+g\s+packs?\s+expiring",
    # Specific product names with recall context
    r"(?:formula|milk|food)\s+\d+g\s+(?:packs?|containers?)",
    # Geographic specificity (not universally applicable)
    r"(?:in|at)\s+(?:Bradford|Manchester|Leeds|Birmingham|London)\s+(?:mosques?|churches?|temples?|centers?)",
    r"(?:mosques?|churches?|temples?)\s+(?:in|at|across)\s+[A-Z][a-z]+",
    r"(?:local|community)\s+(?:mosques?|churches?|centers?)\s+(?:in|at)",
    # Named individuals (case studies that are too specific)
    r"like\s+[A-Z][a-z]+\s+[A-Z][a-z]+,?\s+(?:who|diagnosed|aged)",
    r"[A-Z][a-z]+\s+[A-Z][a-z]+'s\s+(?:cancer|diagnosis|symptoms|story)",
]

_TOO_SPECIFIC_PATTERNS = [re.compile(p, re.IGNORECASE) for p in TOO_SPECIFIC_PATTERNS]

# Wishy-washy / non-actionable advice patterns
WISHY_WASHY_PATTERNS = [
    r"(?:should|must)\s+(?:demand|insist|push\s+for|advocate)",
    r"(?:focus\s+on|prioritize)\s+(?:\w+\s+){0,2}(?:for|to)\s+",  # "Focus on X for Y"
    r"(?:be\s+aware|stay\s+informed|keep\s+in\s+mind)",
    r"(?:consider|think\s+about)\s+(?:your|the)",
]

_WISHY_WASHY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in WISHY_WASHY_PATTERNS]


def is_too_specific(text: str) -> tuple[bool, Optional[str]]:
    """Check if content is too specific (product recalls, locations, named individuals)."""
    for pattern in _TOO_SPECIFIC_PATTERNS:
        match = pattern.search(text)
        if match:
            return True, match.group(0)
    return False, None


def is_wishy_washy(text: str) -> tuple[bool, Optional[str]]:
    """Check if advice is wishy-washy (not concrete eat/do/avoid format)."""
    for pattern in _WISHY_WASHY_PATTERNS:
        match = pattern.search(text)
        if match:
            return True, match.group(0)
    return False, None


def contains_banned_newswords(text: str) -> tuple[bool, list[str]]:
    """
    Check if text contains banned newswords from config.

    Args:
        text: Text to check

    Returns:
        Tuple of (has_banned, list of matched words)
    """
    from .config import get_settings
    settings = get_settings()

    text_lower = text.lower()
    matched = []

    for word in settings.banned_newswords_list:
        if word in text_lower:
            matched.append(word)

    return len(matched) >= 2, matched  # Require 2+ matches to reject


def is_admin_sludge(text: str) -> tuple[bool, Optional[str]]:
    """
    Check if text contains administrative sludge (not health impact content).

    Also checks against banned newswords from config.

    Args:
        text: Title or body text to check

    Returns:
        Tuple of (is_sludge, matched_pattern)
    """
    for pattern in _ADMIN_SLUDGE_PATTERNS:
        match = pattern.search(text)
        if match:
            return True, match.group(0)
    return False, None


def is_lifestyle_trend(text: str) -> tuple[bool, Optional[str]]:
    """
    Check if text is about spending/lifestyle trends (not actionable health).

    Examples that should be rejected:
    - "Young millennials spend £2,000 per Hyrox event"
    - "Gen Z prioritizes fitness over leisure"
    - "Premium fitness industry booming"

    Args:
        text: Title or body text to check

    Returns:
        Tuple of (is_trend, matched_pattern)
    """
    for pattern in _LIFESTYLE_TREND_PATTERNS:
        match = pattern.search(text)
        if match:
            return True, match.group(0)
    return False, None


# =====================================================
# NARRATIVE SPINE QUALITY GATE
# =====================================================

@dataclass
class NarrativeQualityResult:
    """Result of narrative spine quality check."""
    passed: bool
    reason: Optional[str] = None
    clarity_score: int = 0
    emotional_hook: str = "none"
    has_numbers: bool = False
    has_consequence: bool = False


def check_narrative_quality(
    spine: "NarrativeSpine",
    min_clarity_score: int = 7,
    min_lay_relevance: int = 6,
    require_emotional_hook: bool = True,
) -> NarrativeQualityResult:
    """
    Check if a narrative spine meets quality requirements.

    Requirements:
    - standalone_clarity_score >= min_clarity_score (default 7)
    - lay_audience_relevance >= min_lay_relevance (default 6) - filters medical pro content
    - emotional_hook != "none" (if require_emotional_hook is True)
    - Has at least one key number for STUDY_STAT archetype
    - Has real-world consequence
    - HUMAN_INTEREST must have actionable lesson
    - NEWS_POLICY must have controversy potential (moderate or high)

    Args:
        spine: NarrativeSpine to check
        min_clarity_score: Minimum clarity score (default 7)
        min_lay_relevance: Minimum lay audience relevance (default 6)
        require_emotional_hook: Whether to require emotional hook

    Returns:
        NarrativeQualityResult
    """
    # Check clarity score
    if spine.standalone_clarity_score < min_clarity_score:
        return NarrativeQualityResult(
            passed=False,
            reason=f"clarity_too_low: {spine.standalone_clarity_score} < {min_clarity_score}",
            clarity_score=spine.standalone_clarity_score,
            emotional_hook=spine.emotional_hook,
        )

    # Check lay audience relevance (reject medical professional content)
    lay_relevance = getattr(spine, 'lay_audience_relevance', 10)  # Default 10 for backward compat
    if lay_relevance < min_lay_relevance:
        return NarrativeQualityResult(
            passed=False,
            reason=f"not_relevant_to_lay_audience: {lay_relevance}/10 < {min_lay_relevance}",
            clarity_score=spine.standalone_clarity_score,
            emotional_hook=spine.emotional_hook,
        )

    # Check emotional hook
    if require_emotional_hook and spine.emotional_hook == "none":
        return NarrativeQualityResult(
            passed=False,
            reason="no_emotional_hook",
            clarity_score=spine.standalone_clarity_score,
            emotional_hook=spine.emotional_hook,
        )

    # Check key numbers for STUDY_STAT
    has_numbers = len(spine.key_numbers) > 0
    if spine.content_archetype == "STUDY_STAT" and not has_numbers:
        return NarrativeQualityResult(
            passed=False,
            reason="study_stat_without_numbers",
            clarity_score=spine.standalone_clarity_score,
            emotional_hook=spine.emotional_hook,
            has_numbers=False,
        )

    # Check real-world consequence
    has_consequence = bool(spine.real_world_consequence and len(spine.real_world_consequence) >= 10)
    if not has_consequence:
        return NarrativeQualityResult(
            passed=False,
            reason="no_real_world_consequence",
            clarity_score=spine.standalone_clarity_score,
            emotional_hook=spine.emotional_hook,
            has_numbers=has_numbers,
            has_consequence=False,
        )

    # Check for admin sludge
    is_sludge, sludge_match = is_admin_sludge(spine.hook)
    if is_sludge:
        return NarrativeQualityResult(
            passed=False,
            reason=f"admin_sludge: {sludge_match}",
            clarity_score=spine.standalone_clarity_score,
            emotional_hook=spine.emotional_hook,
        )

    # Check for lifestyle/spending trends (not actionable health advice)
    is_trend, trend_match = is_lifestyle_trend(spine.hook)
    if is_trend:
        return NarrativeQualityResult(
            passed=False,
            reason=f"lifestyle_trend_not_health: {trend_match}",
            clarity_score=spine.standalone_clarity_score,
            emotional_hook=spine.emotional_hook,
        )

    # Also check is_actionable field - reject if explicitly marked as not actionable
    is_actionable = getattr(spine, 'is_actionable', True)  # Default True for backward compat
    if not is_actionable:
        return NarrativeQualityResult(
            passed=False,
            reason="not_actionable_content",
            clarity_score=spine.standalone_clarity_score,
            emotional_hook=spine.emotional_hook,
        )

    # HUMAN_INTEREST must have actionable lesson
    actionable_lesson = getattr(spine, 'actionable_lesson', '')
    if spine.content_archetype == "HUMAN_INTEREST" and not actionable_lesson:
        return NarrativeQualityResult(
            passed=False,
            reason="human_interest_without_actionable_lesson",
            clarity_score=spine.standalone_clarity_score,
            emotional_hook=spine.emotional_hook,
        )

    # NEWS_POLICY must have controversy potential
    controversy = getattr(spine, 'controversy_potential', 'none')
    if spine.content_archetype == "NEWS_POLICY" and controversy in ("none", "low"):
        return NarrativeQualityResult(
            passed=False,
            reason=f"news_policy_not_viral_enough: controversy={controversy}",
            clarity_score=spine.standalone_clarity_score,
            emotional_hook=spine.emotional_hook,
        )

    # Check for too-specific content (product recalls, geographic niche, named individuals)
    too_specific, specific_match = is_too_specific(spine.hook)
    if too_specific:
        return NarrativeQualityResult(
            passed=False,
            reason=f"too_specific: {specific_match}",
            clarity_score=spine.standalone_clarity_score,
            emotional_hook=spine.emotional_hook,
        )

    # Check for wishy-washy advice (not concrete do/eat/avoid format)
    wishy, wishy_match = is_wishy_washy(spine.hook)
    if wishy:
        return NarrativeQualityResult(
            passed=False,
            reason=f"wishy_washy_advice: {wishy_match}",
            clarity_score=spine.standalone_clarity_score,
            emotional_hook=spine.emotional_hook,
        )

    return NarrativeQualityResult(
        passed=True,
        clarity_score=spine.standalone_clarity_score,
        emotional_hook=spine.emotional_hook,
        has_numbers=has_numbers,
        has_consequence=has_consequence,
    )


# =====================================================
# STORY COMPRESSION QUALITY GATE
# =====================================================

@dataclass
class HeadlineQualityResult:
    """Result of headline quality check."""
    passed: bool
    reason: Optional[str] = None
    has_number: bool = False
    has_consequence: bool = False
    headline_length: int = 0


def check_headline_quality(
    headline: str,
    min_length: int = 30,
    max_length: int = 300,
) -> HeadlineQualityResult:
    """
    Check if a compressed headline meets quality requirements.

    Requirements:
    - Contains at least one number
    - No source label prefixes
    - Within length bounds
    - No exclamation marks (clickbait indicator)

    Args:
        headline: The compressed headline
        min_length: Minimum character length
        max_length: Maximum character length

    Returns:
        HeadlineQualityResult
    """
    if not headline:
        return HeadlineQualityResult(
            passed=False,
            reason="empty_headline",
        )

    # Check length
    headline_length = len(headline)
    if headline_length < min_length:
        return HeadlineQualityResult(
            passed=False,
            reason=f"too_short: {headline_length} < {min_length}",
            headline_length=headline_length,
        )

    if headline_length > max_length:
        return HeadlineQualityResult(
            passed=False,
            reason=f"too_long: {headline_length} > {max_length}",
            headline_length=headline_length,
        )

    # Check for numbers
    has_number = any(char.isdigit() for char in headline)
    if not has_number:
        return HeadlineQualityResult(
            passed=False,
            reason="no_numbers",
            headline_length=headline_length,
            has_number=False,
        )

    # Check for source prefix patterns
    source_prefix_patterns = [
        r'^[A-Z][a-zA-Z\s]+:\s',  # "Guardian Health: "
        r'^Study:\s',             # "Study: "
        r'^Research:\s',          # "Research: "
    ]
    for pattern in source_prefix_patterns:
        if re.match(pattern, headline):
            return HeadlineQualityResult(
                passed=False,
                reason=f"has_source_prefix",
                headline_length=headline_length,
                has_number=has_number,
            )

    # Check for exclamation marks (clickbait indicator)
    if '!' in headline:
        return HeadlineQualityResult(
            passed=False,
            reason="has_exclamation_mark",
            headline_length=headline_length,
            has_number=has_number,
        )

    # Check for consequence (should have some impact statement)
    consequence_patterns = [
        r"\d+%",  # Percentage
        r"risk",
        r"chance",
        r"likely",
        r"linked to",
        r"associated with",
        r"leads? to",
        r"causes?",
        r"reduces?",
        r"increases?",
    ]
    has_consequence = any(re.search(p, headline, re.IGNORECASE) for p in consequence_patterns)

    return HeadlineQualityResult(
        passed=True,
        has_number=has_number,
        has_consequence=has_consequence,
        headline_length=headline_length,
    )


# =====================================================
# OUTPUT DIVERSITY ENFORCEMENT
# =====================================================

class ArchetypeDiversityEnforcer:
    """
    Enforces output diversity by archetype.

    Prioritizes actionable archetypes (SIMPLE_HABIT, WARNING_RISK, IF_THEN).
    No longer caps STUDY_STAT - instead prioritizes actionable content.
    """

    # Actionable archetypes that should be prioritized
    ACTIONABLE_ARCHETYPES = {"SIMPLE_HABIT", "WARNING_RISK", "IF_THEN"}

    def __init__(
        self,
        max_study_stat: int = 10,  # Effectively no limit
        max_per_archetype: int = 5,  # Generous limit
    ):
        """
        Initialize enforcer.

        Args:
            max_study_stat: Maximum STUDY_STAT items per run (no longer enforced strictly)
            max_per_archetype: Maximum items per any archetype
        """
        self.max_study_stat = max_study_stat
        self.max_per_archetype = max_per_archetype
        self.archetype_counts: dict[str, int] = {}

    def can_add(self, archetype: str) -> tuple[bool, Optional[str]]:
        """
        Check if we can add an item with this archetype.

        Args:
            archetype: The content archetype

        Returns:
            Tuple of (can_add, reason_if_not)
        """
        current_count = self.archetype_counts.get(archetype, 0)

        # General archetype limit (generous)
        if current_count >= self.max_per_archetype:
            return False, f"max_archetype_reached ({self.max_per_archetype})"

        return True, None

    def add(self, archetype: str) -> None:
        """Record that an item with this archetype was added."""
        self.archetype_counts[archetype] = self.archetype_counts.get(archetype, 0) + 1

    def get_counts(self) -> dict[str, int]:
        """Get current archetype counts."""
        return dict(self.archetype_counts)

    def reset(self) -> None:
        """Reset counts for a new run."""
        self.archetype_counts = {}

    @classmethod
    def is_actionable_archetype(cls, archetype: str) -> bool:
        """Check if archetype is an actionable type (prioritized)."""
        return archetype in cls.ACTIONABLE_ARCHETYPES

    @classmethod
    def sort_by_priority(cls, items_with_spines: list) -> list:
        """
        Sort items to prioritize actionable archetypes.

        Items with actionable archetypes (SIMPLE_HABIT, WARNING_RISK, IF_THEN)
        come first, followed by other archetypes sorted by clarity score.

        Args:
            items_with_spines: List of (item, spine, ...) tuples

        Returns:
            Sorted list with actionable content first
        """
        def sort_key(item_tuple):
            spine = item_tuple[1]
            archetype = spine.content_archetype
            is_actionable = archetype in cls.ACTIONABLE_ARCHETYPES

            # Primary: actionable first (0 for actionable, 1 for not)
            # Secondary: clarity score descending
            return (0 if is_actionable else 1, -spine.standalone_clarity_score)

        return sorted(items_with_spines, key=sort_key)
