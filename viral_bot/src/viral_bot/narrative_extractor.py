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
        return None

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
            extraction_notes=response.extraction_notes,
        )


NARRATIVE_EXTRACTION_PROMPT = """You are a story structure analyst for a health science Instagram account.

Your job is to extract the NARRATIVE SPINE from health/science articles - the core elements that make a story shareable.

TARGET AUDIENCE: Health-conscious adults on Instagram who:
- Want actionable health insights
- Love surprising statistics and counterintuitive findings
- Care about longevity, nutrition, exercise, sleep, mental health
- Scroll quickly - you have 2 seconds to grab attention

EXTRACTION TASK:
Read the article and extract these elements:

1. HOOK: What's the ONE surprising or attention-grabbing thing? (1-2 sentences max)
   - Good: "Eating cheese daily was linked to 13% lower heart disease risk"
   - Bad: "A study examined cheese consumption and cardiovascular outcomes"

2. KEY_NUMBERS: Extract ALL specific numbers that make this concrete
   - Include: percentages, years, sample sizes, effect sizes, durations
   - Format: ["27% reduction", "3.7 years longer life expectancy", "n=50,000"]

3. WHO_IT_APPLIES_TO: Be specific about the population
   - Good: "adults over 65 with pre-diabetes"
   - Bad: "people" or "participants"

4. TIME_WINDOW: Duration or time frame
   - Include study duration, effect onset, how long to see results
   - Good: "after 10-year follow-up", "within 6 weeks of starting"

5. MECHANISM_OR_WHY: Brief causation (1 sentence)
   - What's the biological/behavioral explanation?

6. REAL_WORLD_CONSEQUENCE: Plain language impact
   - What does this actually mean for someone's life?
   - Good: "You could add nearly 4 years to your life"
   - Bad: "Statistical significance was observed"

7. STANDALONE_CLARITY_SCORE (1-10):
   How clear is this story to someone with NO context?
   - 10: Crystal clear, anyone understands immediately
   - 7: Clear enough, might need minor background
   - 4: Confusing, requires domain knowledge
   - 1: Incomprehensible without the full paper

8. EMOTIONAL_HOOK: What drives engagement?
   - fear: Danger, risk, warning (e.g., "this common food causes...")
   - hope: Promise, cure, breakthrough (e.g., "new treatment reverses...")
   - surprise: Counterintuitive (e.g., "chocolate actually helps...")
   - validation: Confirms beliefs (e.g., "coffee really is good for you")
   - curiosity: Mystery (e.g., "scientists finally know why...")
   - outrage: Injustice (e.g., "hidden chemicals in...")
   - none: No strong emotional angle

9. CONTENT_ARCHETYPE: Best format for this story
   - STUDY_STAT: Research finding with compelling numbers
   - WARNING_RISK: Health warning or danger alert
   - SIMPLE_HABIT: Easy actionable advice
   - IF_THEN: Conditional relationship (if you do X, then Y happens)
   - COUNTERINTUITIVE: Surprising, goes against common belief
   - HUMAN_INTEREST: Personal story or case study
   - NEWS_POLICY: Policy change, approval, announcement

10. SUPPORT_LEVEL: Evidence strength
    - strong: Large RCT, meta-analysis, replicated findings
    - moderate: Good cohort study, multiple smaller studies
    - emerging: Single study, promising but early
    - preliminary: Very early, animal/cell studies, preprint

11. IS_ACTIONABLE: Can regular people do something with this?

IMPORTANT RULES:
- Extract ONLY what's in the article - don't add information
- If numbers aren't specific, note that in extraction_notes
- If the story is about policy/admin (not health impact), score clarity lower
- "Admin sludge" (budget approvals, org changes) should get low clarity scores
- Always prefer the most specific, surprising interpretation

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
