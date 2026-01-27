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

STORY_COMPRESSION_SYSTEM = """Compress narrative into Instagram slide text for an actionable health account.

FORMATS (priority order):
1. "Do X to improve Y": "Walking 20min daily reduces heart disease risk by 31%"
2. "Eat X for Y effect": "Low-glycemic foods reduce dementia risk by 16%"
3. "X causes Y bad effect": "Sitting 8+ hours daily increases diabetes risk by 90%"

STRUCTURE: [Action/Food] + [Amount/Frequency] + [Effect with number] + [Who if specific]

RULES:
- Include specific number from key_numbers
- Plain language, no jargon
- 1-3 sentences, use ellipses (...) for dramatic pause
- NO source prefixes ("Study:", "Harvard:")
- NO clickbait, questions, or exclamation marks
- Use "linked to" for correlational findings
- Focus on WHAT TO DO or AVOID

Return valid JSON only."""


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

    numbers_str = ", ".join(spine.key_numbers) if spine.key_numbers else "(none)"

    return f"""HOOK: {spine.hook}
NUMBERS: {numbers_str}
WHO: {spine.who_it_applies_to}
TIME: {spine.time_window}
CONSEQUENCE: {spine.real_world_consequence}
ARCHETYPE: {spine.content_archetype}"""


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
