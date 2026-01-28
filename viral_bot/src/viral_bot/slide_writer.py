"""
Slide Copy Writer - Instagram On-Image Text Generation

Generates ALL CAPS slide copy that matches the winner corpus style.
This is NOT a news headline - it's Instagram on-image text that stands alone.
"""

import re
from typing import Optional
from dataclasses import dataclass

from openai import OpenAI
from pydantic import BaseModel, Field

from .config import get_settings
from .logging_conf import get_logger
from .normalize import NormalizedItem
from .narrative_extractor import NarrativeSpine

logger = get_logger(__name__)


class SlideWriterResponse(BaseModel):
    """Structured response from slide copy generation."""

    image_headline: Optional[str] = Field(
        description="The on-image text for Instagram slide. ALL CAPS. Self-contained. Must include a number or strong comparison. Null if cannot meet requirements."
    )

    reason: Optional[str] = Field(
        default=None,
        description="If image_headline is null, explain why this content cannot be converted to viral slide copy."
    )


@dataclass
class SlideCopyResult:
    """Result of slide copy generation."""
    image_headline: Optional[str] = None
    reason: Optional[str] = None
    was_generated: bool = False

    @property
    def success(self) -> bool:
        return self.image_headline is not None


# Study-related opener phrases to rotate
STUDY_OPENERS = [
    "STUDY SAYS",
    "STUDY REVEALS",
    "STUDY FINDS",
    "NEW STUDY SHOWS",
    "RESEARCH REVEALS",
    "SCIENTISTS FIND",
    "RESEARCHERS DISCOVER",
]


SLIDE_WRITER_SYSTEM = """You are a slide copy writer for an Instagram health account (@avoidaging / aging.ai).

Your job is to write ON-IMAGE TEXT that matches our top-performing style. This is NOT a news headline - it's Instagram slide text.

WINNER CORPUS STYLE (you MUST match this):
- ALL CAPS (always)
- Templates like: "STUDY SAYS", "STUDY REVEALS", "ONE [FOOD] CAN", "THIS IS THE [BODY PART]", "IF YOU [ACTION]"
- Mini-story compression: hook → numbers/time → outcome
- Use commas for pacing and flow, NOT ellipsis (no "...")
- NO source labels or prefixes (no "Guardian Health:", "Harvard Study:")
- Self-contained and readable without a caption

EXCELLENT EXAMPLES:
1. "STUDY SAYS EATING ONE EGG A DAY CAN REDUCE YOUR RISK OF STROKE BY 12%"
2. "PROCESSED DELI MEATS ARE NOW CLASSIFIED AS GROUP 1 CARCINOGENS BY THE WHO, THE SAME GROUP AS TOBACCO AND ASBESTOS"
3. "IF YOU WAKE UP BETWEEN 3-5 AM REGULARLY, YOUR LUNGS MAY BE TRYING TO TELL YOU SOMETHING"
4. "WALKING JUST 11 MINUTES A DAY REDUCES YOUR RISK OF EARLY DEATH BY 23%"
5. "ONE ALCOHOLIC DRINK A DAY IS NOW LINKED TO A 51% INCREASE IN MOUTH CANCER RISK"
6. "STUDY REVEALS THAT PEOPLE WHO EAT FERMENTED FOODS 5+ TIMES A WEEK HAVE 34% FEWER SICK DAYS"

BAD EXAMPLES (DO NOT DO):
- "New study reveals surprising findings about sleep" (no specifics, lowercase)
- "Harvard researchers find health benefits" (no numbers, source prefix)
- "Study: Exercise is good for you" (generic, uses colon prefix)
- "You won't believe what scientists found!" (clickbait)

HARD RULES:
1. ALL CAPS (entire text)
2. Must include at least ONE: specific number OR dramatic comparison OR time-based claim
3. NO source prefixes ("Study:", "Research:", "According to...")
4. Use "LINKED TO" or "ASSOCIATED WITH" for correlational findings
5. No exclamation marks
6. 1-3 sentences maximum
7. Complete, standalone statement
8. Plain language - no medical jargon
9. If evidence is correlational, do NOT use causal language like "causes"

If you cannot meet these rules, return null for image_headline and explain why in reason."""


def build_slide_prompt(spine: NarrativeSpine, item: NormalizedItem) -> str:
    """Build the user prompt for slide copy generation."""

    numbers_str = ", ".join(spine.numbers) if spine.numbers else "(none)"

    prompt = f"""Generate Instagram slide copy for this narrative:

PRIMITIVE: {spine.primitive}
HOOK: {spine.hook}
ACTION: {spine.action or "(none)"}
OUTCOME: {spine.outcome or "(none)"}
NUMBERS: {numbers_str}
TIME WINDOW: {spine.time_window or "(none)"}
WHO: {spine.who_it_applies_to or "general population"}
WHY IT MATTERS: {spine.why_it_matters}
TONE: {spine.tone}
SUPPORT LEVEL: {spine.support_level}

Generate ALL CAPS slide copy that:
1. Starts with an appropriate opener for the primitive type
2. Includes at least one number from the NUMBERS list
3. Is self-contained and readable without context
4. Matches the winner corpus style

Return JSON with image_headline (or null if cannot meet requirements) and reason."""

    return prompt


class SlideWriter:
    """Generates Instagram slide copy from narrative spines."""

    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.model = self.settings.openai_model

    def write(
        self,
        item: NormalizedItem,
        spine: NarrativeSpine,
    ) -> SlideCopyResult:
        """
        Generate slide copy from a narrative spine.

        Args:
            item: NormalizedItem for source context
            spine: NarrativeSpine with extracted narrative elements

        Returns:
            SlideCopyResult with image_headline or rejection reason
        """
        # Skip if spine was already rejected
        if not spine.relevant:
            return SlideCopyResult(
                reason=f"spine_rejected: {spine.rejection_reason}",
                was_generated=False,
            )

        # Skip if clarity too low
        if spine.standalone_clarity < 7:
            return SlideCopyResult(
                reason=f"clarity_too_low: {spine.standalone_clarity}",
                was_generated=False,
            )

        user_prompt = build_slide_prompt(spine, item)

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": SLIDE_WRITER_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=SlideWriterResponse,
                temperature=0.7,
            )

            parsed = response.choices[0].message.parsed

            if parsed is None:
                return SlideCopyResult(
                    reason="llm_returned_none",
                    was_generated=True,
                )

            # Post-process and validate
            headline = parsed.image_headline
            if headline:
                headline = self._sanitize_headline(headline)
                if not self._validate_headline(headline):
                    return SlideCopyResult(
                        reason="failed_validation",
                        was_generated=True,
                    )

            return SlideCopyResult(
                image_headline=headline,
                reason=parsed.reason,
                was_generated=True,
            )

        except Exception as e:
            logger.error("slide_writer_failed", url=item.url, error=str(e))
            return SlideCopyResult(
                reason=f"error: {str(e)}",
                was_generated=False,
            )

    def _sanitize_headline(self, headline: str) -> str:
        """Clean and format the headline."""
        # Ensure ALL CAPS
        headline = headline.upper()

        # Strip leading source prefixes
        prefix_patterns = [
            r'^[A-Z\s]+:\s*',  # "STUDY: " or "GUARDIAN HEALTH: "
            r'^[A-Z\s]+-\s*',  # "RESEARCH - "
            r'^ACCORDING\s+TO\s+[A-Z\s,]+,?\s*',  # "ACCORDING TO RESEARCHERS,"
        ]
        for pattern in prefix_patterns:
            headline = re.sub(pattern, '', headline, flags=re.IGNORECASE)

        # Remove exclamation marks
        headline = headline.replace('!', '')

        # Replace ellipsis with comma (user preference)
        headline = re.sub(r'\.{2,}', ',', headline)
        headline = headline.replace('…', ',')
        # Clean up double commas or comma-space issues
        headline = re.sub(r',\s*,', ',', headline)
        headline = re.sub(r'\s+,', ',', headline)

        # Clean up whitespace
        headline = ' '.join(headline.split())

        return headline.strip()

    def _validate_headline(self, headline: str) -> bool:
        """Validate headline meets requirements."""
        if not headline:
            return False

        # Must have some length
        if len(headline) < 30:
            return False

        # Must be ALL CAPS (allow numbers and punctuation)
        alpha_chars = [c for c in headline if c.isalpha()]
        if alpha_chars and not all(c.isupper() for c in alpha_chars):
            return False

        # Should have at least one number OR comparison word
        has_number = any(c.isdigit() for c in headline)
        comparison_words = ['SAME', 'EQUIVALENT', 'LIKE', 'AS MUCH AS', 'MORE THAN', 'LESS THAN']
        has_comparison = any(word in headline for word in comparison_words)

        if not has_number and not has_comparison:
            # Check for time-based claims
            time_words = ['DAILY', 'WEEKLY', 'EVERY DAY', 'PER WEEK', 'MINUTES', 'HOURS']
            has_time = any(word in headline for word in time_words)
            if not has_time:
                return False

        # No source prefixes (after sanitization, double-check)
        if ':' in headline[:30]:  # Colon in first 30 chars suggests prefix
            return False

        return True


# Singleton writer
_writer: Optional[SlideWriter] = None


def get_slide_writer() -> SlideWriter:
    """Get singleton slide writer instance."""
    global _writer
    if _writer is None:
        _writer = SlideWriter()
    return _writer


def write_slide_copy(item: NormalizedItem, spine: NarrativeSpine) -> SlideCopyResult:
    """Convenience function to write slide copy."""
    writer = get_slide_writer()
    return writer.write(item, spine)
