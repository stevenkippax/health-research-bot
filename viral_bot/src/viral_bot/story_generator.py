"""
Story Compression Headline Generator - AI Stage #2

Takes a NarrativeSpine and compresses it into Instagram-ready slide text.

Output format: Complete statements, NOT clickbait
- Hook + Numbers + Consequence
- "Babies who get skin-to-skin contact with their fathers in the first hour
   have 42% better temperature regulation and stronger cardiac stability."
- "Deli ham is officially classified as a Group 1 carcinogen by the WHO...
   the same group as tobacco and asbestos."

NO source label prefixes like "Guardian Health:", "Study:", etc.
"""

import json
import re
from typing import Optional
from dataclasses import dataclass, field, asdict

from openai import OpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import get_settings
from .logging_conf import get_logger
from .normalize import NormalizedItem
from .narrative_extractor import NarrativeSpine

logger = get_logger(__name__)


# =====================================================
# HEADLINE SANITIZER
# =====================================================

# Patterns to strip from the beginning of headlines
SOURCE_PREFIX_PATTERNS = [
    r'^[A-Z][a-zA-Z\s]+:\s*',          # "Guardian Health: ", "BBC News: "
    r'^Study:\s*',                       # "Study: "
    r'^New study:\s*',                   # "New study: "
    r'^Research:\s*',                    # "Research: "
    r'^Breaking:\s*',                    # "Breaking: "
    r'^BREAKING:\s*',                    # "BREAKING: "
    r'^Exclusive:\s*',                   # "Exclusive: "
    r'^Update:\s*',                      # "Update: "
    r'^\[.*?\]\s*',                      # "[Source] "
    r'^From\s+[A-Z][a-zA-Z\s]+:\s*',    # "From Harvard: "
]


def sanitize_headline(headline: str) -> str:
    """
    Remove source label prefixes and clean up the headline.

    Examples:
        "Guardian Health: Study finds..." -> "Study finds..."
        "Study: Walking 10 minutes..." -> "Walking 10 minutes..."
        "[BBC] New research shows..." -> "New research shows..."

    Args:
        headline: Raw headline text

    Returns:
        Cleaned headline without source prefixes
    """
    if not headline:
        return headline

    cleaned = headline.strip()

    # Apply all prefix patterns
    for pattern in SOURCE_PREFIX_PATTERNS:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

    # Remove any leading/trailing whitespace
    cleaned = cleaned.strip()

    # Ensure first character is capitalized
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]

    # Remove exclamation marks
    cleaned = cleaned.replace('!', '.')

    # Remove double periods
    cleaned = re.sub(r'\.{2,}', '.', cleaned)

    return cleaned


# =====================================================
# STORY COMPRESSION PROMPT
# =====================================================

STORY_COMPRESSION_SYSTEM = """You are a story compressor for a quasi-scientific health Instagram account that gives ACTIONABLE advice.

Your job is to take structured narrative elements and compress them into shareable slide text that helps people take action.

THIS IS A BRAND THAT GIVES:
- Actionable advice to followers
- Warnings about health risks
- A sprinkle of breakthroughs and health AI developments

PRIORITIZE THESE FORMATS (in order of preference):

1. "Do X to ease/improve Y" format:
   - "Walking just 20 minutes daily reduces heart disease risk by 31%"
   - "Eating 2 servings of fermented foods daily lowers inflammation by 34%"

2. "Eat X to cause Y effect" format:
   - "Eating low-glycemic foods like fruits and whole grains can reduce dementia risk by 16%"
   - "Consuming olive oil daily linked to 28% lower risk of death from all causes"

3. "Eating/Doing X causes Y bad effect" (warnings):
   - "Consuming processed foods with preservatives like potassium sorbate linked to 14% higher cancer risk"
   - "Sitting for more than 8 hours daily increases diabetes risk by 90%"
   - "Ultra-processed foods increase depression risk by 33%"

OUTPUT FORMAT: Complete statements that stand alone
- NOT clickbait ("You won't believe...")
- NOT questions ("Did you know...?")
- NOT source labels ("Harvard study:", "New research:")

IDEAL OUTPUT STRUCTURE:
[Action/Food/Habit] + [Specific amount/frequency] + [Effect with numbers] + [Who it applies to if specific]

EXCELLENT EXAMPLES:
1. "Consuming processed foods with preservatives like potassium sorbate linked to 14% higher cancer risk over 7.5 years... potentially increasing prostate cancer risk by 32%."

2. "Eating low-glycemic foods like fruits and whole grains can reduce dementia risk by 16%, supporting long-term brain health."

3. "Adults who sleep less than 6 hours per night have a 27% higher risk of atherosclerosis, even if they exercise regularly and eat well."

4. "Deli ham is officially classified as a Group 1 carcinogen by the WHO... the same group as tobacco and asbestos."

5. "People who eat fermented foods 5+ times per week have measurably lower inflammation markers and 34% fewer sick days."

6. "Taking a 10-minute walk after meals reduces blood sugar spikes by up to 22%."

BAD EXAMPLES (DO NOT DO):
- "New study reveals surprising findings about sleep" (no specifics)
- "Scientists discover link between diet and health" (too vague)
- "You won't believe what researchers found about exercise!" (clickbait)
- "Study: Exercise is good for you" (boring, uses "Study:" prefix)
- "Harvard research shows health benefits" (no numbers, vague)
- "Lowering the blood detection threshold to 80 micrograms..." (too technical for lay audience)
- "A mismatch between two common blood tests can signal..." (diagnostic info for doctors, not actionable)

RULES:
1. MUST include at least one specific number from key_numbers
2. MUST be actionable or a clear warning for regular people
3. MUST include the real-world consequence in plain language
4. Use ellipses (...) for dramatic pause before the punchline
5. NO source label prefixes (no "Study:", "Research:", "According to...")
6. NO exclamation marks
7. 1-3 sentences maximum
8. Plain language - avoid medical jargon
9. Use "linked to" or "associated with" for correlational findings
10. Complete, standalone statements that make sense without context
11. Focus on WHAT PEOPLE CAN DO or WHAT THEY SHOULD AVOID

IMPORTANT: Return ONLY valid JSON."""


# Pydantic model for structured output
class StoryCompressionResponse(BaseModel):
    """Structured response from story compression."""

    headline: str = Field(
        description="The compressed slide text (1-3 sentences, no source prefixes)"
    )

    highlight_words: list[str] = Field(
        default_factory=list,
        description="Key words/numbers to highlight in colored text (3-5 max)"
    )

    image_suggestion: str = Field(
        description="Concrete visual concept for the background image"
    )

    layout_notes: list[str] = Field(
        default_factory=list,
        description="Notes about text placement and visual design"
    )

    generation_notes: str = Field(
        default="",
        description="Any notes about the compression process"
    )


@dataclass
class StoryCompressionResult:
    """Result of story compression."""
    headline: str = ""
    highlight_words: list[str] = field(default_factory=list)
    image_suggestion: str = ""
    layout_notes: list[str] = field(default_factory=list)
    generation_notes: str = ""

    # Quality check
    @property
    def is_valid(self) -> bool:
        """Check if the compression result is valid."""
        if not self.headline or len(self.headline) < 30:
            return False
        # Must have at least one number
        if not any(char.isdigit() for char in self.headline):
            return False
        # Must not start with source prefix
        if re.match(r'^[A-Z][a-z]+:', self.headline):
            return False
        return True

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_response(cls, response: StoryCompressionResponse) -> "StoryCompressionResult":
        """Create from Pydantic response."""
        # Apply sanitizer to the headline
        cleaned_headline = sanitize_headline(response.headline)

        return cls(
            headline=cleaned_headline,
            highlight_words=response.highlight_words,
            image_suggestion=response.image_suggestion,
            layout_notes=response.layout_notes,
            generation_notes=response.generation_notes,
        )


def build_compression_prompt(spine: NarrativeSpine, item: NormalizedItem) -> str:
    """Build the user prompt for story compression."""

    numbers_str = "\n".join([f"  - {num}" for num in spine.key_numbers]) if spine.key_numbers else "  (none extracted)"

    return f"""Compress this narrative into Instagram slide text:

HOOK: {spine.hook}

KEY NUMBERS (must include at least one):
{numbers_str}

WHO IT APPLIES TO: {spine.who_it_applies_to}

TIME WINDOW: {spine.time_window}

MECHANISM: {spine.mechanism_or_why}

REAL-WORLD CONSEQUENCE: {spine.real_world_consequence}

EMOTIONAL HOOK: {spine.emotional_hook}
CONTENT ARCHETYPE: {spine.content_archetype}
SUPPORT LEVEL: {spine.support_level}

SOURCE NAME: {item.source_name}
CREDIBILITY TIER: {item.credibility_tier.value}

Return JSON:
{{
  "headline": "Your compressed slide text here (1-3 sentences, include numbers)",
  "highlight_words": ["word1", "number1", "impact_word"],
  "image_suggestion": "Concrete visual concept",
  "layout_notes": ["note1", "note2"],
  "generation_notes": "Any notes"
}}

REMINDER:
- Include at least ONE number from key_numbers
- NO source prefixes ("Study:", "Research:", etc.)
- Complete statement that stands alone
- Ellipses (...) for dramatic effect before punchline"""


class StoryCompressor:
    """
    Compresses narrative spines into Instagram-ready headlines.

    This is the second AI stage in the story-compression pipeline.
    """

    def __init__(self):
        """Initialize compressor with OpenAI client."""
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.model = self.settings.openai_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def compress(
        self,
        item: NormalizedItem,
        spine: NarrativeSpine,
    ) -> Optional[StoryCompressionResult]:
        """
        Compress a narrative spine into slide text.

        Args:
            item: NormalizedItem for source context
            spine: NarrativeSpine with structured narrative elements

        Returns:
            StoryCompressionResult or None if compression fails
        """
        user_prompt = build_compression_prompt(spine, item)

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": STORY_COMPRESSION_SYSTEM,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                response_format=StoryCompressionResponse,
                temperature=0.7,  # Some creativity for compelling text
            )

            parsed = response.choices[0].message.parsed

            if parsed is None:
                logger.warning("story_compression_returned_none", url=item.url)
                return None

            result = StoryCompressionResult.from_response(parsed)

            # Validate the result
            if not result.is_valid:
                logger.warning(
                    "story_compression_invalid",
                    url=item.url,
                    headline=result.headline[:50] if result.headline else None,
                )
                return None

            logger.debug(
                "story_compressed",
                url=item.url,
                headline_length=len(result.headline),
                highlight_count=len(result.highlight_words),
            )

            return result

        except Exception as e:
            logger.error(
                "story_compression_failed",
                url=item.url,
                error=str(e),
            )
            return None

    def compress_batch(
        self,
        items_with_spines: list[tuple[NormalizedItem, NarrativeSpine]],
        max_outputs: int = 10,
    ) -> list[tuple[NormalizedItem, NarrativeSpine, StoryCompressionResult]]:
        """
        Compress multiple items.

        Args:
            items_with_spines: List of (NormalizedItem, NarrativeSpine) tuples
            max_outputs: Maximum outputs to generate

        Returns:
            List of (item, spine, result) tuples
        """
        results = []
        count = 0

        for item, spine in items_with_spines:
            if count >= max_outputs:
                break

            result = self.compress(item, spine)

            if result is not None:
                results.append((item, spine, result))
                count += 1

                logger.info(
                    "story_compression_success",
                    title=item.title[:40],
                    headline=result.headline[:50],
                )

        logger.info(
            "story_compression_batch_complete",
            input_count=len(items_with_spines),
            output_count=len(results),
        )

        return results


# =====================================================
# COMBINED GENERATOR (backward compatibility)
# =====================================================

@dataclass
class GenerationResult:
    """Combined result for backward compatibility."""
    headline: "HeadlineResult"
    image: "ImageSuggestionResult"

    def to_dict(self) -> dict:
        return {
            "headline": self.headline.to_dict(),
            "image": self.image.to_dict(),
        }


@dataclass
class HeadlineResult:
    """Headline result for backward compatibility."""
    image_headline: Optional[str] = None
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ImageSuggestionResult:
    """Image suggestion for backward compatibility."""
    image_suggestion: str = ""
    layout_notes: list[str] = field(default_factory=list)
    highlight_words: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def story_compression_to_generation_result(
    result: StoryCompressionResult,
) -> GenerationResult:
    """Convert story compression result to legacy GenerationResult format."""
    return GenerationResult(
        headline=HeadlineResult(
            image_headline=result.headline,
            reason=None,
        ),
        image=ImageSuggestionResult(
            image_suggestion=result.image_suggestion,
            layout_notes=result.layout_notes,
            highlight_words=result.highlight_words,
        ),
    )
