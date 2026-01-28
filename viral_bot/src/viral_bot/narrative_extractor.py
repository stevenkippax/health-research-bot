"""
Narrative Spine Extraction - AI Stage #1

Extracts the story structure from articles before headline generation.
This produces structured narrative elements that feed into slide copy generation.

V3 Output Schema (Viral Likeness Pipeline):
- relevant: Whether this content is suitable for Instagram health audience
- rejection_reason: Why content was rejected (null if relevant)
- primitive: Viral primitive type (STUDY_SHOCK_COMPARISON, SIMPLE_HACK_PAIN_RELIEF, etc.)
- hook: Plain English hook
- action: What to do (null if not actionable)
- outcome: Expected result
- numbers: Specific numbers from text
- time_window: Duration/timeframe
- who_it_applies_to: Target population
- mechanism_clause: Brief causation (<=12 words)
- why_it_matters: One sentence impact
- standalone_clarity: 0-10 score
- tone: shock/awe/concern/warmth/humor
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal
from enum import Enum

from openai import OpenAI
from pydantic import BaseModel, Field

from .config import get_settings
from .logging_conf import get_logger
from .normalize import NormalizedItem

logger = get_logger(__name__)


class EmotionalHook(str, Enum):
    """Types of emotional hooks that drive engagement."""
    FEAR = "fear"
    HOPE = "hope"
    SURPRISE = "surprise"
    VALIDATION = "validation"
    CURIOSITY = "curiosity"
    OUTRAGE = "outrage"
    NONE = "none"


class Tone(str, Enum):
    """Tone categories for slide copy."""
    SHOCK = "shock"
    AWE = "awe"
    CONCERN = "concern"
    WARMTH = "warmth"
    HUMOR = "humor"


# Pydantic model for structured output (V3)
class NarrativeSpineResponse(BaseModel):
    """Structured response from narrative extraction - V3 with viral primitives."""

    relevant: bool = Field(
        description="Is this content suitable for Instagram health audience? False for admin sludge, null results, generic truisms, promo content."
    )

    rejection_reason: Optional[str] = Field(
        default=None,
        description="If not relevant, why? (admin_sludge, null_result, generic_truism, promo_content, too_technical, local_program)"
    )

    primitive: Literal[
        "STUDY_SHOCK_COMPARISON",
        "SIMPLE_HACK_PAIN_RELIEF",
        "FOOD_SYMPTOM_BENEFIT",
        "PARENT_CHILD_BIO",
        "AUTHORITY_CLASSIFICATION",
        "CULTURE_CONTROVERSY",
        "TIME_REVERSAL",
        "BODY_PART_SPECIFIC",
        "NONE"
    ] = Field(
        description="Best viral primitive for this content"
    )

    hook: str = Field(
        description="The surprising or attention-grabbing core element in plain English (1-2 sentences)"
    )

    action: Optional[str] = Field(
        default=None,
        description="What specific action can people take? (eat X, avoid Y, do Z) - null if not actionable"
    )

    outcome: Optional[str] = Field(
        default=None,
        description="Expected result of the action or finding (reduce pain, lower risk, improve sleep)"
    )

    numbers: list[str] = Field(
        default_factory=list,
        description="Specific numbers from the text (e.g., '27%', '3.7 years', '12,000 participants')"
    )

    time_window: Optional[str] = Field(
        default=None,
        description="Duration or time frame (e.g., '10-year follow-up', 'within 6 weeks', 'daily for 3 months')"
    )

    who_it_applies_to: Optional[str] = Field(
        default=None,
        description="Specific population (e.g., 'adults over 50', 'pregnant women') - null if general"
    )

    mechanism_clause: str = Field(
        description="Brief causation explanation - MUST BE 12 WORDS OR FEWER"
    )

    why_it_matters: str = Field(
        description="One sentence explaining real-world impact in plain language"
    )

    standalone_clarity: int = Field(
        ge=0, le=10,
        description="0-10: How clear is this without context? 10 = crystal clear, 0 = incomprehensible"
    )

    tone: Literal["shock", "awe", "concern", "warmth", "humor"] = Field(
        description="Primary tone for the content"
    )

    emotional_hook: Literal["fear", "hope", "surprise", "validation", "curiosity", "outrage", "none"] = Field(
        description="Primary emotional driver"
    )

    support_level: Literal["strong", "moderate", "emerging", "preliminary"] = Field(
        description="Evidence strength: strong (RCT/meta), moderate (cohort), emerging (single study), preliminary (animal/cell)"
    )

    extraction_notes: str = Field(
        default="",
        description="Any caveats about extraction quality"
    )


@dataclass
class NarrativeSpine:
    """Extracted narrative structure from an article - V3."""
    # Relevance gate
    relevant: bool = True
    rejection_reason: Optional[str] = None

    # Viral primitive
    primitive: str = "NONE"

    # Core narrative elements
    hook: str = ""
    action: Optional[str] = None
    outcome: Optional[str] = None
    numbers: list[str] = field(default_factory=list)
    time_window: Optional[str] = None
    who_it_applies_to: Optional[str] = None
    mechanism_clause: str = ""
    why_it_matters: str = ""

    # Quality metrics
    standalone_clarity: int = 0
    tone: str = "concern"
    emotional_hook: str = "none"
    support_level: str = "moderate"
    extraction_notes: str = ""

    # Legacy compatibility aliases
    @property
    def key_numbers(self) -> list[str]:
        return self.numbers

    @property
    def standalone_clarity_score(self) -> int:
        return self.standalone_clarity

    @property
    def real_world_consequence(self) -> str:
        return self.why_it_matters

    @property
    def mechanism_or_why(self) -> str:
        return self.mechanism_clause

    @property
    def content_archetype(self) -> str:
        """Map primitive to content archetype for backward compatibility."""
        primitive_to_archetype = {
            "STUDY_SHOCK_COMPARISON": "STUDY_STAT",
            "SIMPLE_HACK_PAIN_RELIEF": "SIMPLE_HABIT",
            "FOOD_SYMPTOM_BENEFIT": "SIMPLE_HABIT",
            "PARENT_CHILD_BIO": "STUDY_STAT",
            "AUTHORITY_CLASSIFICATION": "WARNING_RISK",
            "CULTURE_CONTROVERSY": "NEWS_POLICY",
            "TIME_REVERSAL": "COUNTERINTUITIVE",
            "BODY_PART_SPECIFIC": "SIMPLE_HABIT",
            "NONE": "STUDY_STAT",
        }
        return primitive_to_archetype.get(self.primitive, "STUDY_STAT")

    @property
    def is_actionable(self) -> bool:
        return self.action is not None and len(self.action) > 0

    @property
    def passes_quality_gate(self) -> bool:
        """Check if this spine passes minimum quality requirements."""
        if not self.relevant:
            return False
        if self.standalone_clarity < 7:
            return False
        if self.emotional_hook == "none":
            return False
        if self.primitive in ("STUDY_SHOCK_COMPARISON", "PARENT_CHILD_BIO") and not self.numbers:
            return False
        if not self.why_it_matters or len(self.why_it_matters) < 10:
            return False
        return True

    @property
    def quality_failure_reason(self) -> Optional[str]:
        """Get the reason for quality gate failure."""
        if not self.relevant:
            return f"not_relevant: {self.rejection_reason}"
        if self.standalone_clarity < 7:
            return f"standalone_clarity too low ({self.standalone_clarity})"
        if self.emotional_hook == "none":
            return "no emotional hook"
        if self.primitive in ("STUDY_SHOCK_COMPARISON", "PARENT_CHILD_BIO") and not self.numbers:
            return f"{self.primitive} without numbers"
        if not self.why_it_matters or len(self.why_it_matters) < 10:
            return "missing why_it_matters"
        return None

    @property
    def is_actionable_archetype(self) -> bool:
        """Check if this is an actionable primitive."""
        return self.primitive in (
            "SIMPLE_HACK_PAIN_RELIEF",
            "FOOD_SYMPTOM_BENEFIT",
            "BODY_PART_SPECIFIC",
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_response(cls, response: NarrativeSpineResponse) -> "NarrativeSpine":
        """Create from Pydantic response."""
        return cls(
            relevant=response.relevant,
            rejection_reason=response.rejection_reason,
            primitive=response.primitive,
            hook=response.hook,
            action=response.action,
            outcome=response.outcome,
            numbers=response.numbers,
            time_window=response.time_window,
            who_it_applies_to=response.who_it_applies_to,
            mechanism_clause=response.mechanism_clause,
            why_it_matters=response.why_it_matters,
            standalone_clarity=response.standalone_clarity,
            tone=response.tone,
            emotional_hook=response.emotional_hook,
            support_level=response.support_level,
            extraction_notes=response.extraction_notes,
        )


NARRATIVE_EXTRACTION_PROMPT = """You are extracting narrative elements for an Instagram health account (@avoidaging / aging.ai).

TARGET STYLE (from our top-performing posts):
- "STUDY SAYS EATING ONE EGG A DAY CAN REDUCE YOUR RISK OF STROKE BY 12%"
- "IF YOU WAKE UP BETWEEN 3-5 AM REGULARLY... YOUR LUNGS MAY BE TRYING TO TELL YOU SOMETHING"
- "PROCESSED DELI MEATS ARE NOW CLASSIFIED AS GROUP 1 CARCINOGENS BY THE WHO... THE SAME GROUP AS TOBACCO"
- "WALKING JUST 11 MINUTES A DAY REDUCES YOUR RISK OF EARLY DEATH BY 23%"

HARD REJECTS (set relevant=false):
1. NULL RESULTS: "no effect", "may not work", "not better than placebo", "failed to show"
2. ADMIN SLUDGE: objectives, stakeholders, frameworks, governance, local authorities, procurement
3. GENERIC TRUISMS: "exercise is good", "sleep matters", "eat healthy" (unless shocking numbers)
4. PROMO CONTENT: film premieres, event registrations, documentary releases
5. SMALL LOCAL PROGRAMS: "150 people joined community walking group"
6. NICHE SUBGROUPS: content only relevant to rare medical conditions with no broad appeal
7. DENSE JARGON: technical abbreviations that don't translate to Instagram slides

VIRAL PRIMITIVES (choose the best fit):
- STUDY_SHOCK_COMPARISON: Study + shocking number/comparison ("equivalent to smoking X cigarettes")
- SIMPLE_HACK_PAIN_RELIEF: Small physical action → symptom relief ("roll tennis ball under foot")
- FOOD_SYMPTOM_BENEFIT: Specific food → health benefit ("eating walnuts reduces inflammation")
- PARENT_CHILD_BIO: Baby/parent content + biological markers + numbers
- AUTHORITY_CLASSIFICATION: WHO/FDA/IARC classification + dramatic comparison
- CULTURE_CONTROVERSY: Health-relevant controversy (verifiable, not pure politics)
- TIME_REVERSAL: Age reversal, longevity, "add X years to your life"
- BODY_PART_SPECIFIC: Specific body part + improvement/risk
- NONE: No clear viral primitive (likely reject)

EXTRACTION RULES:
1. hook: Plain English, 1-2 sentences, must be attention-grabbing
2. action: What can people DO? (eat X, avoid Y, walk for Z minutes) - null if not actionable
3. outcome: The result/benefit/risk
4. numbers: ONLY numbers explicitly stated in text - do not infer
5. time_window: Duration mentioned (weeks, months, years)
6. mechanism_clause: Brief causation - MUST BE 12 WORDS OR FEWER
7. why_it_matters: One sentence, plain language impact
8. standalone_clarity: Would someone understand this without reading the article? (0-10)
9. tone: shock (alarming finding), awe (amazing discovery), concern (warning), warmth (heartwarming), humor (surprising/funny)

ARTICLE TO ANALYZE:
"""


class NarrativeExtractor:
    """
    Extracts narrative spine from articles using OpenAI.

    This is the first AI stage in the story-compression pipeline.
    """

    def __init__(self):
        """Initialize extractor with OpenAI client."""
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.model = self.settings.openai_model

    def extract(self, item: NormalizedItem) -> Optional[NarrativeSpine]:
        """
        Extract narrative spine from a normalized item.

        Args:
            item: NormalizedItem with body text

        Returns:
            NarrativeSpine or None if extraction fails
        """
        if not item.body_text or len(item.body_text) < 200:
            logger.warning("insufficient_content_for_extraction", url=item.url)
            return None

        # Build the prompt
        full_text = item.get_full_text_for_evaluation()
        prompt = NARRATIVE_EXTRACTION_PROMPT + full_text

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a story structure analyst. Extract narrative elements in the specified JSON format.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                response_format=NarrativeSpineResponse,
                temperature=0.3,  # Lower temp for consistent extraction
            )

            parsed = response.choices[0].message.parsed

            if parsed is None:
                logger.warning("narrative_extraction_returned_none", url=item.url)
                return None

            spine = NarrativeSpine.from_response(parsed)

            logger.debug(
                "narrative_extracted",
                url=item.url,
                clarity_score=spine.standalone_clarity_score,
                emotional_hook=spine.emotional_hook,
                archetype=spine.content_archetype,
                key_numbers_count=len(spine.key_numbers),
            )

            return spine

        except Exception as e:
            logger.error(
                "narrative_extraction_failed",
                url=item.url,
                error=str(e),
            )
            return None

    def extract_batch(
        self,
        items: list[NormalizedItem],
        require_quality_gate: bool = True,
    ) -> list[tuple[NormalizedItem, NarrativeSpine]]:
        """
        Extract narrative spines from multiple items.

        Args:
            items: List of NormalizedItems
            require_quality_gate: If True, filter out items that fail quality gate

        Returns:
            List of (item, spine) tuples that pass quality requirements
        """
        results = []

        for item in items:
            spine = self.extract(item)

            if spine is None:
                continue

            if require_quality_gate and not spine.passes_quality_gate:
                logger.debug(
                    "narrative_failed_quality_gate",
                    url=item.url,
                    reason=spine.quality_failure_reason,
                )
                continue

            results.append((item, spine))

        logger.info(
            "narrative_extraction_batch_complete",
            input_count=len(items),
            output_count=len(results),
        )

        return results


def extract_narrative(item: NormalizedItem) -> Optional[NarrativeSpine]:
    """
    Convenience function to extract narrative spine.

    Args:
        item: NormalizedItem to process

    Returns:
        NarrativeSpine or None
    """
    extractor = NarrativeExtractor()
    return extractor.extract(item)
