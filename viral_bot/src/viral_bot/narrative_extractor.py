"""
Narrative Spine Extraction - AI Stage #1

Extracts the story structure from articles before headline generation.
This produces structured narrative elements that feed into story compression.

Output schema:
- hook: The surprising/attention-grabbing element
- key_numbers: Specific numbers that make the story concrete
- who_it_applies_to: The population this affects
- time_window: Duration/time frame of the effect
- mechanism_or_why: Brief explanation of causation
- real_world_consequence: What this means for people's lives
- standalone_clarity_score: 1-10, how clear is this without context?
- emotional_hook: fear/hope/surprise/validation/curiosity/outrage/none
- lay_audience_relevance: 1-10, is this useful for regular people (not medical pros)?
- actionable_lesson: For human interest stories, what can people learn/do?
- controversy_potential: For news/policy, is this controversial/viral?
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
    FEAR = "fear"                    # Danger, risk, warning
    HOPE = "hope"                    # Promise, possibility, cure
    SURPRISE = "surprise"            # Counterintuitive, unexpected
    VALIDATION = "validation"        # Confirms existing beliefs
    CURIOSITY = "curiosity"          # Mystery, "how" questions
    OUTRAGE = "outrage"              # Injustice, scandal
    NONE = "none"                    # No strong emotional hook


# Pydantic model for structured output
class NarrativeSpineResponse(BaseModel):
    """Structured response from narrative extraction."""

    hook: str = Field(
        description="The surprising or attention-grabbing core element (1-2 sentences)"
    )

    key_numbers: list[str] = Field(
        default_factory=list,
        description="Specific numbers that make the story concrete (e.g., '27% reduction', '3.7 years longer', '12,000 participants')"
    )

    who_it_applies_to: str = Field(
        description="The specific population this affects (e.g., 'adults over 50', 'pregnant women', 'people with diabetes')"
    )

    time_window: str = Field(
        description="Duration or time frame of the effect (e.g., '10-year follow-up', 'within 6 weeks', 'daily for 3 months')"
    )

    mechanism_or_why: str = Field(
        description="Brief explanation of causation or mechanism (1 sentence)"
    )

    real_world_consequence: str = Field(
        description="What this means for people's lives in plain language"
    )

    standalone_clarity_score: int = Field(
        ge=1, le=10,
        description="1-10: How clear is this story without any additional context? 10 = crystal clear to anyone"
    )

    emotional_hook: Literal["fear", "hope", "surprise", "validation", "curiosity", "outrage", "none"] = Field(
        description="Primary emotional driver"
    )

    content_archetype: Literal[
        "STUDY_STAT",           # Research finding with numbers
        "WARNING_RISK",         # Health warning/danger
        "SIMPLE_HABIT",         # Easy actionable advice
        "IF_THEN",              # Conditional relationship
        "COUNTERINTUITIVE",     # Surprising/unexpected
        "HUMAN_INTEREST",       # Personal story/case
        "NEWS_POLICY",          # Policy change/announcement
    ] = Field(
        description="Best content archetype for this story"
    )

    support_level: Literal["strong", "moderate", "emerging", "preliminary"] = Field(
        description="How strong is the evidence supporting this claim?"
    )

    is_actionable: bool = Field(
        description="Can regular people act on this information?"
    )

    lay_audience_relevance: int = Field(
        ge=1, le=10,
        description="1-10: How relevant is this to regular people (not medical professionals)? 10 = perfect for Instagram health audience, 1 = only useful for doctors/researchers"
    )

    actionable_lesson: str = Field(
        default="",
        description="For HUMAN_INTEREST stories: What specific action can readers take based on this story? (e.g., 'Get suspicious moles checked early'). Leave empty if not a human interest story."
    )

    controversy_potential: Literal["high", "moderate", "low", "none"] = Field(
        default="none",
        description="For NEWS_POLICY: Does this have viral controversy potential? High = sparks debate (like paid period leave), Moderate = interesting policy, Low = routine announcement, None = not applicable"
    )

    extraction_notes: str = Field(
        default="",
        description="Any caveats or notes about extraction quality"
    )


@dataclass
class NarrativeSpine:
    """Extracted narrative structure from an article."""
    hook: str = ""
    key_numbers: list[str] = field(default_factory=list)
    who_it_applies_to: str = ""
    time_window: str = ""
    mechanism_or_why: str = ""
    real_world_consequence: str = ""
    standalone_clarity_score: int = 0
    emotional_hook: str = "none"
    content_archetype: str = "STUDY_STAT"
    support_level: str = "moderate"
    is_actionable: bool = False
    lay_audience_relevance: int = 5  # NEW: 1-10 score for lay audience
    actionable_lesson: str = ""  # NEW: For HUMAN_INTEREST stories
    controversy_potential: str = "none"  # NEW: For NEWS_POLICY stories
    extraction_notes: str = ""

    # Quality gates
    @property
    def passes_quality_gate(self) -> bool:
        """Check if this spine passes minimum quality requirements."""
        # Must have clarity score >= 7
        if self.standalone_clarity_score < 7:
            return False
        # Must have some emotional hook
        if self.emotional_hook == "none":
            return False
        # Must have at least one key number for STUDY_STAT
        if self.content_archetype == "STUDY_STAT" and len(self.key_numbers) == 0:
            return False
        # Must have real-world consequence
        if not self.real_world_consequence or len(self.real_world_consequence) < 10:
            return False
        # Must be relevant to lay audience (not medical professionals)
        if self.lay_audience_relevance < 6:
            return False
        # HUMAN_INTEREST must have actionable lesson
        if self.content_archetype == "HUMAN_INTEREST" and not self.actionable_lesson:
            return False
        # NEWS_POLICY must have some controversy potential to be viral
        if self.content_archetype == "NEWS_POLICY" and self.controversy_potential in ("none", "low"):
            return False
        return True

    @property
    def quality_failure_reason(self) -> Optional[str]:
        """Get the reason for quality gate failure."""
        if self.standalone_clarity_score < 7:
            return f"standalone_clarity_score too low ({self.standalone_clarity_score})"
        if self.emotional_hook == "none":
            return "no emotional hook"
        if self.content_archetype == "STUDY_STAT" and len(self.key_numbers) == 0:
            return "STUDY_STAT without key numbers"
        if not self.real_world_consequence or len(self.real_world_consequence) < 10:
            return "missing real-world consequence"
        if self.lay_audience_relevance < 6:
            return f"not_relevant_to_lay_audience ({self.lay_audience_relevance}/10)"
        if self.content_archetype == "HUMAN_INTEREST" and not self.actionable_lesson:
            return "human_interest_without_actionable_lesson"
        if self.content_archetype == "NEWS_POLICY" and self.controversy_potential in ("none", "low"):
            return f"news_policy_not_viral_enough (controversy: {self.controversy_potential})"
        return None

    @property
    def is_actionable_archetype(self) -> bool:
        """Check if this is an actionable archetype (prioritized)."""
        return self.content_archetype in ("SIMPLE_HABIT", "WARNING_RISK", "IF_THEN")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_response(cls, response: NarrativeSpineResponse) -> "NarrativeSpine":
        """Create from Pydantic response."""
        return cls(
            hook=response.hook,
            key_numbers=response.key_numbers,
            who_it_applies_to=response.who_it_applies_to,
            time_window=response.time_window,
            mechanism_or_why=response.mechanism_or_why,
            real_world_consequence=response.real_world_consequence,
            standalone_clarity_score=response.standalone_clarity_score,
            emotional_hook=response.emotional_hook,
            content_archetype=response.content_archetype,
            support_level=response.support_level,
            is_actionable=response.is_actionable,
            lay_audience_relevance=response.lay_audience_relevance,
            actionable_lesson=response.actionable_lesson,
            controversy_potential=response.controversy_potential,
            extraction_notes=response.extraction_notes,
        )


NARRATIVE_EXTRACTION_PROMPT = """Extract narrative elements for an Instagram health account that gives ACTIONABLE advice.

GOAL: "Do X to ease Y", "Eat X for Y effect", "Avoid X because Y" format content.
AUDIENCE: Regular people wanting practical health tips, NOT medical professionals.

EXTRACT:
1. HOOK: One surprising attention-grabber with numbers. NOT vague like "study examined".
2. KEY_NUMBERS: All specific stats (%, years, sample sizes, effect sizes).
3. WHO_IT_APPLIES_TO: Specific population, not "people".
4. TIME_WINDOW: Duration/timeframe of effect.
5. MECHANISM_OR_WHY: One sentence on causation.
6. REAL_WORLD_CONSEQUENCE: Plain language impact on daily life.
7. STANDALONE_CLARITY_SCORE (1-10): 10=crystal clear to anyone, 1=needs full paper.
8. EMOTIONAL_HOOK: fear/hope/surprise/validation/curiosity/outrage/none
9. CONTENT_ARCHETYPE (prioritize actionable):
   - SIMPLE_HABIT: "Do X to get Y" - PRIORITIZE
   - WARNING_RISK: "X causes Y bad effect" - PRIORITIZE
   - IF_THEN: "If X then Y" - PRIORITIZE
   - STUDY_STAT: Research finding with numbers
   - COUNTERINTUITIVE: Surprising findings
   - HUMAN_INTEREST: Personal story (needs actionable_lesson)
   - NEWS_POLICY: Policy change (needs high controversy_potential)
10. SUPPORT_LEVEL: strong/moderate/emerging/preliminary
11. IS_ACTIONABLE: Can regular people act on this?
    TRUE: eat/avoid something, exercise, change habit
    FALSE: spending trends, lifestyle observations, medical-pro only, requires Rx
12. LAY_AUDIENCE_RELEVANCE (1-10): 10=perfect actionable advice, 1=only for doctors.
    REJECT <6: clinical thresholds, spending trends, diagnostic info.
13. ACTIONABLE_LESSON: For HUMAN_INTEREST only - what action can readers take? Empty if none.
14. CONTROVERSY_POTENTIAL: For NEWS_POLICY only - high/moderate/low/none. Reject low/none.

REJECT: spending habits, lifestyle trends, medical-pro content, product recalls, geographic-specific advice.

ARTICLE:
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
