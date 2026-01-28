"""
Viral primitives extraction and matching.

Deterministic extraction of viral elements from content before LLM generation.
These primitives identify content with high viral potential based on patterns
that consistently perform well on Instagram health accounts.
"""

import re
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

from .logging_conf import get_logger

logger = get_logger(__name__)


class ViralPrimitive(str, Enum):
    """Types of viral primitives that drive engagement."""
    STUDY_SHOCK_COMPARISON = "STUDY_SHOCK_COMPARISON"  # Study + shocking comparison
    SIMPLE_HACK_PAIN_RELIEF = "SIMPLE_HACK_PAIN_RELIEF"  # Small hack -> pain relief
    FOOD_SYMPTOM_BENEFIT = "FOOD_SYMPTOM_BENEFIT"  # Food/ingredient -> benefit
    PARENT_CHILD_BIO = "PARENT_CHILD_BIO"  # Baby/parent + bio markers
    AUTHORITY_CLASSIFICATION = "AUTHORITY_CLASSIFICATION"  # WHO/FDA classification
    CULTURE_CONTROVERSY = "CULTURE_CONTROVERSY"  # Health-relevant controversy
    TIME_REVERSAL = "TIME_REVERSAL"  # Age reversal, longevity findings
    BODY_PART_SPECIFIC = "BODY_PART_SPECIFIC"  # Specific body part benefit
    NONE = "NONE"  # No strong primitive match


@dataclass
class ExtractedElements:
    """Extracted viral elements from content."""
    # Numbers and statistics
    percentages: list[str] = field(default_factory=list)
    years: list[str] = field(default_factory=list)
    times_multipliers: list[str] = field(default_factory=list)
    sample_sizes: list[str] = field(default_factory=list)
    age_ranges: list[str] = field(default_factory=list)

    # Time windows
    time_windows: list[str] = field(default_factory=list)

    # Comparisons
    equivalent_comparisons: list[str] = field(default_factory=list)
    versus_comparisons: list[str] = field(default_factory=list)

    # Body parts and symptoms
    body_parts: list[str] = field(default_factory=list)
    symptoms: list[str] = field(default_factory=list)

    # Actions
    action_verbs: list[str] = field(default_factory=list)
    foods: list[str] = field(default_factory=list)

    # Authority mentions
    authorities: list[str] = field(default_factory=list)

    # Parent/child terms
    parent_child_terms: list[str] = field(default_factory=list)
    bio_markers: list[str] = field(default_factory=list)

    # Study indicators
    study_indicators: list[str] = field(default_factory=list)

    @property
    def has_numbers(self) -> bool:
        return bool(
            self.percentages or self.years or
            self.times_multipliers or self.sample_sizes
        )

    @property
    def has_time_window(self) -> bool:
        return bool(self.time_windows or self.years)

    @property
    def has_comparison(self) -> bool:
        return bool(self.equivalent_comparisons or self.versus_comparisons)

    @property
    def has_authority(self) -> bool:
        return bool(self.authorities)

    @property
    def has_body_part(self) -> bool:
        return bool(self.body_parts)

    @property
    def has_action(self) -> bool:
        return bool(self.action_verbs or self.foods)


@dataclass
class PrimitiveMatch:
    """Result of primitive matching."""
    primitive: ViralPrimitive
    score: int  # 0-100
    confidence: float  # 0.0-1.0
    elements: ExtractedElements
    match_reasons: list[str] = field(default_factory=list)

    @property
    def passes_threshold(self) -> bool:
        from .config import get_settings
        settings = get_settings()
        return self.score >= settings.primitive_threshold


# Extraction patterns

# Numbers
PERCENTAGE_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*%', re.IGNORECASE)
YEARS_PATTERN = re.compile(r'(\d+)\s*(?:years?|yrs?)\b', re.IGNORECASE)
TIMES_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*(?:times?|x)\s+(?:more|less|higher|lower|greater|faster)', re.IGNORECASE)
FOLD_PATTERN = re.compile(r'(\d+(?:\.\d+)?)-?fold', re.IGNORECASE)
SAMPLE_SIZE_PATTERN = re.compile(r'(?:n\s*=\s*|among\s+|studied\s+|included\s+)(\d{1,3}(?:,\d{3})*|\d+)\s*(?:participants?|people|subjects?|adults?|patients?)?', re.IGNORECASE)
AGE_PATTERN = re.compile(r'(?:aged?|ages?)\s+(\d+(?:\s*-\s*\d+)?)', re.IGNORECASE)

# Time windows
TIME_WINDOW_PATTERNS = [
    re.compile(r'(?:over|within|after|for)\s+(\d+)\s*(?:weeks?|months?|years?|days?)', re.IGNORECASE),
    re.compile(r'(\d+)\s*(?:weeks?|months?|years?|days?)\s+(?:of\s+)?(?:follow-?up|study|later)', re.IGNORECASE),
    re.compile(r'(?:daily|weekly|monthly)\s+for\s+(\d+)\s*(?:weeks?|months?|years?)', re.IGNORECASE),
]

# Comparisons
EQUIVALENT_PATTERNS = [
    re.compile(r'equivalent\s+to\s+([^\.]{10,60})', re.IGNORECASE),
    re.compile(r'same\s+(?:as|effect)\s+(?:as\s+)?([^\.]{10,60})', re.IGNORECASE),
    re.compile(r'like\s+(?:smoking|drinking|eating)\s+([^\.]{10,40})', re.IGNORECASE),
]
VERSUS_PATTERN = re.compile(r'(?:compared\s+to|versus|vs\.?)\s+([^\.]{5,50})', re.IGNORECASE)

# Body parts
BODY_PARTS = [
    'brain', 'heart', 'liver', 'kidney', 'gut', 'stomach', 'intestine',
    'lung', 'skin', 'bone', 'muscle', 'joint', 'knee', 'hip', 'back',
    'spine', 'neck', 'shoulder', 'eye', 'ear', 'blood', 'artery', 'vein',
    'cells?', 'dna', 'genes?', 'mitochondria', 'immune\s+system',
]
BODY_PART_PATTERN = re.compile(r'\b(' + '|'.join(BODY_PARTS) + r')\b', re.IGNORECASE)

# Symptoms
SYMPTOMS = [
    'pain', 'ache', 'inflammation', 'swelling', 'fatigue', 'tiredness',
    'anxiety', 'depression', 'stress', 'insomnia', 'headache', 'migraine',
    'nausea', 'bloating', 'constipation', 'diarrhea', 'cramp',
    'stiffness', 'soreness', 'weakness', 'dizziness', 'brain\s+fog',
]
SYMPTOM_PATTERN = re.compile(r'\b(' + '|'.join(SYMPTOMS) + r')\b', re.IGNORECASE)

# Action verbs
ACTION_VERBS = [
    'eat', 'drink', 'take', 'consume', 'avoid', 'reduce', 'increase',
    'walk', 'run', 'exercise', 'sleep', 'stretch', 'massage',
    'freeze', 'heat', 'ice', 'rub', 'roll', 'blend', 'mix',
    'breathe', 'meditate', 'fast', 'skip', 'limit', 'cut',
]
ACTION_VERB_PATTERN = re.compile(r'\b(' + '|'.join(ACTION_VERBS) + r')(?:ing|ed|s)?\b', re.IGNORECASE)

# Foods
FOODS = [
    'coffee', 'tea', 'chocolate', 'cheese', 'yogurt', 'milk', 'egg',
    'fish', 'salmon', 'tuna', 'chicken', 'beef', 'pork', 'meat',
    'vegetable', 'fruit', 'apple', 'banana', 'berry', 'blueberry',
    'spinach', 'kale', 'broccoli', 'carrot', 'tomato', 'avocado',
    'olive\s+oil', 'coconut\s+oil', 'butter', 'nuts?', 'almond', 'walnut',
    'rice', 'bread', 'pasta', 'oat', 'quinoa', 'legume', 'bean',
    'garlic', 'onion', 'ginger', 'turmeric', 'cinnamon', 'honey',
    'wine', 'alcohol', 'beer', 'soda', 'juice', 'water',
    'supplement', 'vitamin', 'mineral', 'protein', 'fiber', 'probiotic',
]
FOOD_PATTERN = re.compile(r'\b(' + '|'.join(FOODS) + r')s?\b', re.IGNORECASE)

# Authorities
AUTHORITIES = [
    'WHO', 'FDA', 'CDC', 'NIH', 'IARC', 'NHS', 'USDA',
    'World\s+Health\s+Organization',
    'Food\s+and\s+Drug\s+Administration',
    'Centers?\s+for\s+Disease\s+Control',
    'National\s+Institutes?\s+of\s+Health',
]
AUTHORITY_PATTERN = re.compile(r'\b(' + '|'.join(AUTHORITIES) + r')\b', re.IGNORECASE)

# Parent/child
PARENT_CHILD_TERMS = [
    'baby', 'babies', 'infant', 'newborn', 'toddler', 'child', 'children',
    'parent', 'mother', 'father', 'maternal', 'paternal', 'pregnancy',
    'pregnant', 'breastfeed', 'nursing', 'womb', 'fetal', 'fetus',
]
PARENT_CHILD_PATTERN = re.compile(r'\b(' + '|'.join(PARENT_CHILD_TERMS) + r')\b', re.IGNORECASE)

# Bio markers
BIO_MARKERS = [
    'cortisol', 'oxytocin', 'dopamine', 'serotonin', 'melatonin',
    'testosterone', 'estrogen', 'insulin', 'glucose', 'cholesterol',
    'blood\s+pressure', 'heart\s+rate', 'inflammation', 'cytokine',
    'telomere', 'epigenetic', 'microbiome', 'gut\s+bacteria',
]
BIO_MARKER_PATTERN = re.compile(r'\b(' + '|'.join(BIO_MARKERS) + r')\b', re.IGNORECASE)

# Study indicators
STUDY_INDICATORS = [
    'study', 'research', 'researchers?', 'scientists?', 'found',
    'discovered', 'reveals?', 'shows?', 'suggests?', 'linked',
    'associated', 'trial', 'experiment', 'analysis', 'meta-analysis',
]
STUDY_INDICATOR_PATTERN = re.compile(r'\b(' + '|'.join(STUDY_INDICATORS) + r')\b', re.IGNORECASE)


def extract_elements(text: str) -> ExtractedElements:
    """
    Extract all viral elements from text.

    Args:
        text: Article title + body text

    Returns:
        ExtractedElements with all found elements
    """
    elements = ExtractedElements()

    # Numbers
    elements.percentages = PERCENTAGE_PATTERN.findall(text)
    elements.years = YEARS_PATTERN.findall(text)

    times_matches = TIMES_PATTERN.findall(text)
    fold_matches = FOLD_PATTERN.findall(text)
    elements.times_multipliers = times_matches + [f"{m}-fold" for m in fold_matches]

    elements.sample_sizes = SAMPLE_SIZE_PATTERN.findall(text)
    elements.age_ranges = AGE_PATTERN.findall(text)

    # Time windows
    for pattern in TIME_WINDOW_PATTERNS:
        elements.time_windows.extend(pattern.findall(text))

    # Comparisons
    for pattern in EQUIVALENT_PATTERNS:
        elements.equivalent_comparisons.extend(pattern.findall(text))
    elements.versus_comparisons = VERSUS_PATTERN.findall(text)

    # Body parts and symptoms
    elements.body_parts = list(set(BODY_PART_PATTERN.findall(text.lower())))
    elements.symptoms = list(set(SYMPTOM_PATTERN.findall(text.lower())))

    # Actions and foods
    elements.action_verbs = list(set(ACTION_VERB_PATTERN.findall(text.lower())))
    elements.foods = list(set(FOOD_PATTERN.findall(text.lower())))

    # Authority mentions
    elements.authorities = list(set(AUTHORITY_PATTERN.findall(text)))

    # Parent/child
    elements.parent_child_terms = list(set(PARENT_CHILD_PATTERN.findall(text.lower())))
    elements.bio_markers = list(set(BIO_MARKER_PATTERN.findall(text.lower())))

    # Study indicators
    elements.study_indicators = list(set(STUDY_INDICATOR_PATTERN.findall(text.lower())))

    return elements


def match_primitive(text: str, elements: Optional[ExtractedElements] = None) -> PrimitiveMatch:
    """
    Match text to the best viral primitive.

    Args:
        text: Article text to analyze
        elements: Pre-extracted elements (optional, will extract if not provided)

    Returns:
        PrimitiveMatch with best primitive and score
    """
    if elements is None:
        elements = extract_elements(text)

    scores = {}
    reasons = {}

    # STUDY_SHOCK_COMPARISON: Study + numbers + comparison
    study_shock_score = 0
    study_shock_reasons = []
    if elements.study_indicators:
        study_shock_score += 20
        study_shock_reasons.append("has_study_indicator")
    if elements.percentages:
        study_shock_score += 25
        study_shock_reasons.append(f"has_percentage:{elements.percentages[0]}%")
    if elements.has_comparison:
        study_shock_score += 30
        study_shock_reasons.append("has_comparison")
    if elements.times_multipliers:
        study_shock_score += 25
        study_shock_reasons.append(f"has_multiplier:{elements.times_multipliers[0]}")
    scores[ViralPrimitive.STUDY_SHOCK_COMPARISON] = study_shock_score
    reasons[ViralPrimitive.STUDY_SHOCK_COMPARISON] = study_shock_reasons

    # SIMPLE_HACK_PAIN_RELIEF: Action verb + symptom relief
    hack_score = 0
    hack_reasons = []
    if elements.action_verbs:
        hack_score += 30
        hack_reasons.append(f"has_action:{elements.action_verbs[0]}")
    if elements.symptoms:
        hack_score += 35
        hack_reasons.append(f"targets_symptom:{elements.symptoms[0]}")
    if elements.time_windows:
        hack_score += 20
        hack_reasons.append("has_time_window")
    if elements.body_parts:
        hack_score += 15
        hack_reasons.append(f"body_part:{elements.body_parts[0]}")
    scores[ViralPrimitive.SIMPLE_HACK_PAIN_RELIEF] = hack_score
    reasons[ViralPrimitive.SIMPLE_HACK_PAIN_RELIEF] = hack_reasons

    # FOOD_SYMPTOM_BENEFIT: Food + health benefit
    food_score = 0
    food_reasons = []
    if elements.foods:
        food_score += 35
        food_reasons.append(f"has_food:{elements.foods[0]}")
    if elements.symptoms or elements.body_parts:
        food_score += 30
        food_reasons.append("targets_health_area")
    if elements.percentages:
        food_score += 20
        food_reasons.append("has_percentage")
    if elements.time_windows:
        food_score += 15
        food_reasons.append("has_time_window")
    scores[ViralPrimitive.FOOD_SYMPTOM_BENEFIT] = food_score
    reasons[ViralPrimitive.FOOD_SYMPTOM_BENEFIT] = food_reasons

    # PARENT_CHILD_BIO: Parent/child terms + bio markers
    parent_score = 0
    parent_reasons = []
    if elements.parent_child_terms:
        parent_score += 40
        parent_reasons.append(f"has_parent_child:{elements.parent_child_terms[0]}")
    if elements.bio_markers:
        parent_score += 35
        parent_reasons.append(f"has_bio_marker:{elements.bio_markers[0]}")
    if elements.percentages:
        parent_score += 15
        parent_reasons.append("has_percentage")
    if elements.time_windows:
        parent_score += 10
        parent_reasons.append("has_time_window")
    scores[ViralPrimitive.PARENT_CHILD_BIO] = parent_score
    reasons[ViralPrimitive.PARENT_CHILD_BIO] = parent_reasons

    # AUTHORITY_CLASSIFICATION: Authority + classification language
    authority_score = 0
    authority_reasons = []
    if elements.authorities:
        authority_score += 50
        authority_reasons.append(f"has_authority:{elements.authorities[0]}")
    # Check for classification language
    classification_patterns = [
        r'classif(?:y|ied|ication)',
        r'carcinogen',
        r'group\s+\d',
        r'ban(?:ned)?',
        r'warning',
        r'approved',
    ]
    for pattern in classification_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            authority_score += 25
            authority_reasons.append(f"has_classification_language")
            break
    if elements.has_comparison:
        authority_score += 15
        authority_reasons.append("has_comparison")
    scores[ViralPrimitive.AUTHORITY_CLASSIFICATION] = authority_score
    reasons[ViralPrimitive.AUTHORITY_CLASSIFICATION] = authority_reasons

    # BODY_PART_SPECIFIC: Specific body part + numbers
    body_score = 0
    body_reasons = []
    if elements.body_parts:
        body_score += 35
        body_reasons.append(f"has_body_part:{elements.body_parts[0]}")
    if elements.percentages:
        body_score += 25
        body_reasons.append("has_percentage")
    if elements.action_verbs:
        body_score += 20
        body_reasons.append("has_action")
    if elements.time_windows:
        body_score += 20
        body_reasons.append("has_time_window")
    scores[ViralPrimitive.BODY_PART_SPECIFIC] = body_score
    reasons[ViralPrimitive.BODY_PART_SPECIFIC] = body_reasons

    # TIME_REVERSAL: Age/longevity + reversal language
    time_score = 0
    time_reasons = []
    reversal_patterns = [
        r'revers(?:e|ed|ing|al)',
        r'younger',
        r'anti-?aging',
        r'longevity',
        r'lifespan',
        r'years?\s+(?:longer|younger|older)',
        r'biological\s+age',
    ]
    for pattern in reversal_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            time_score += 40
            time_reasons.append("has_reversal_language")
            break
    if elements.years:
        time_score += 30
        time_reasons.append(f"has_years:{elements.years[0]}")
    if elements.percentages:
        time_score += 20
        time_reasons.append("has_percentage")
    if elements.study_indicators:
        time_score += 10
        time_reasons.append("has_study")
    scores[ViralPrimitive.TIME_REVERSAL] = time_score
    reasons[ViralPrimitive.TIME_REVERSAL] = time_reasons

    # Find best match
    best_primitive = max(scores, key=scores.get)
    best_score = scores[best_primitive]

    # If no strong match, return NONE
    if best_score < 30:
        best_primitive = ViralPrimitive.NONE
        best_score = 0

    # Calculate confidence based on score distribution
    total_score = sum(scores.values())
    confidence = best_score / total_score if total_score > 0 else 0.0

    return PrimitiveMatch(
        primitive=best_primitive,
        score=best_score,
        confidence=confidence,
        elements=elements,
        match_reasons=reasons.get(best_primitive, []),
    )


def check_hard_rejects(text: str) -> tuple[bool, Optional[str]]:
    """
    Check for hard reject patterns.

    Args:
        text: Article text to check

    Returns:
        Tuple of (should_reject, reason)
    """
    text_lower = text.lower()

    # Null results
    null_patterns = [
        r'(?:no|not?)\s+(?:significant|effect|difference|benefit|improvement)',
        r'(?:may\s+)?not\s+(?:work|help|improve)',
        r'failed\s+to\s+(?:show|find|demonstrate)',
        r'not\s+better\s+than\s+placebo',
        r'no\s+evidence\s+(?:of|that|for)',
    ]
    for pattern in null_patterns:
        if re.search(pattern, text_lower):
            return True, f"null_result:{pattern}"

    # Admin/policy sludge
    admin_patterns = [
        r'\bobjectives?\b',
        r'\bstakeholders?\b',
        r'\bframework\b',
        r'\bgovernance\b',
        r'\bcommissioning\b',
        r'\bprocurement\b',
        r'local\s+authorities?',
        r'action\s+plan\b',
        r'key\s+performance',
    ]
    admin_count = sum(1 for p in admin_patterns if re.search(p, text_lower))
    if admin_count >= 2:
        return True, "admin_sludge"

    # Promotional content
    promo_patterns = [
        r'\bfilm\s+premiere\b',
        r'\bdocumentary\s+(?:release|screening)\b',
        r'\bevent\s+registration\b',
        r'\bsign\s+up\s+(?:now|today)\b',
        r'\btickets?\s+(?:available|on\s+sale)\b',
    ]
    for pattern in promo_patterns:
        if re.search(pattern, text_lower):
            return True, f"promotional:{pattern}"

    # Small local programs
    local_patterns = [
        r'\d{1,3}\s+(?:people|participants?|members?)\s+(?:joined|signed|enrolled)',
        r'community\s+(?:program|initiative|project)\s+(?:launches?|starts?)',
        r'local\s+(?:club|group|organization)\s+(?:offers?|hosts?)',
    ]
    for pattern in local_patterns:
        if re.search(pattern, text_lower):
            return True, f"local_program:{pattern}"

    return False, None


def analyze_content(text: str) -> dict:
    """
    Full analysis of content for viral potential.

    Args:
        text: Article title + body text

    Returns:
        Dict with elements, primitive match, and rejection info
    """
    # Check hard rejects first
    should_reject, reject_reason = check_hard_rejects(text)

    if should_reject:
        return {
            "rejected": True,
            "rejection_reason": reject_reason,
            "elements": None,
            "primitive_match": None,
        }

    # Extract elements
    elements = extract_elements(text)

    # Match primitive
    primitive_match = match_primitive(text, elements)

    return {
        "rejected": False,
        "rejection_reason": None,
        "elements": elements,
        "primitive_match": primitive_match,
    }
