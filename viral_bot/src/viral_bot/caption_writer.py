"""
Caption Writer - Instagram Post Caption Generation

Generates supporting captions for Instagram slides.
Captions add context but remain short, with source attribution.
"""

from typing import Optional
from dataclasses import dataclass

from openai import OpenAI
from pydantic import BaseModel, Field

from .config import get_settings
from .logging_conf import get_logger
from .normalize import NormalizedItem
from .narrative_extractor import NarrativeSpine

logger = get_logger(__name__)


class CaptionWriterResponse(BaseModel):
    """Structured response from caption generation."""

    caption: str = Field(
        description="The Instagram caption. 2-4 short paragraphs with line breaks. Must end with Source line."
    )


@dataclass
class CaptionResult:
    """Result of caption generation."""
    caption: Optional[str] = None
    was_generated: bool = False

    @property
    def success(self) -> bool:
        return self.caption is not None and len(self.caption) > 20


CAPTION_WRITER_SYSTEM = """You are a caption writer for an Instagram health account (@avoidaging / aging.ai).

Your job is to write SHORT supporting captions that add context to the slide image.

CAPTION RULES:
1. 2-4 SHORT paragraphs (separated by blank lines)
2. NO hashtags
3. NO emojis (unless very minimal)
4. NO long lists or bullet points
5. Plain, conversational language
6. If evidence is weak/associative, include ONE softening clause ("early evidence suggests", "may be linked to", etc.)
7. MUST end with "Source: [source_name]" and URL on the last line

STRUCTURE:
- Paragraph 1: Brief expansion on the slide's claim
- Paragraph 2: Additional context or mechanism (optional)
- Paragraph 3: Practical implication or caveat
- Last line: Source: [Name] - [URL]

EXAMPLE CAPTION:
"A new study found that consuming just one egg per day was associated with a significantly lower risk of stroke, particularly hemorrhagic stroke.

The researchers followed over 400,000 participants for nearly a decade. Eggs contain choline and other nutrients that may support vascular health.

While this is observational data, it adds to growing evidence that eggs can be part of a heart-healthy diet.

Source: American Journal of Clinical Nutrition - https://example.com/study"

AVOID:
- Long paragraphs (keep each under 3 sentences)
- Medical jargon
- Overclaiming ("this WILL prevent heart disease")
- Multiple hashtags
- Bullet point lists
- Repetition of the slide text verbatim"""


def build_caption_prompt(
    spine: NarrativeSpine,
    item: NormalizedItem,
    image_headline: str,
) -> str:
    """Build the user prompt for caption generation."""

    numbers_str = ", ".join(spine.numbers) if spine.numbers else "(none)"

    prompt = f"""Write a caption to accompany this Instagram slide:

SLIDE TEXT: {image_headline}

NARRATIVE CONTEXT:
- Hook: {spine.hook}
- Mechanism: {spine.mechanism_clause}
- Why it matters: {spine.why_it_matters}
- Numbers: {numbers_str}
- Time window: {spine.time_window or "(not specified)"}
- Population: {spine.who_it_applies_to or "general"}
- Support level: {spine.support_level}

SOURCE INFORMATION:
- Source name: {item.source_name}
- URL: {item.url}

Write a 2-4 paragraph caption that:
1. Adds context without repeating the slide verbatim
2. Uses softening language if support_level is "emerging" or "preliminary"
3. Ends with "Source: {item.source_name} - {item.url}"

Return JSON with the caption."""

    return prompt


class CaptionWriter:
    """Generates Instagram captions from narrative spines."""

    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.model = self.settings.openai_model

    def write(
        self,
        item: NormalizedItem,
        spine: NarrativeSpine,
        image_headline: str,
    ) -> CaptionResult:
        """
        Generate caption for a slide.

        Args:
            item: NormalizedItem for source context
            spine: NarrativeSpine with extracted narrative elements
            image_headline: The slide text to write caption for

        Returns:
            CaptionResult with caption text
        """
        if not image_headline:
            return CaptionResult(was_generated=False)

        user_prompt = build_caption_prompt(spine, item, image_headline)

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": CAPTION_WRITER_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=CaptionWriterResponse,
                temperature=0.7,
            )

            parsed = response.choices[0].message.parsed

            if parsed is None:
                return CaptionResult(was_generated=True)

            # Post-process caption
            caption = self._sanitize_caption(parsed.caption, item)

            return CaptionResult(
                caption=caption,
                was_generated=True,
            )

        except Exception as e:
            logger.error("caption_writer_failed", url=item.url, error=str(e))
            return CaptionResult(was_generated=False)

    def _sanitize_caption(self, caption: str, item: NormalizedItem) -> str:
        """Clean and format the caption."""
        # Remove excessive hashtags (keep at most 0)
        import re
        caption = re.sub(r'#\w+\s*', '', caption)

        # Ensure source line is present
        source_patterns = [
            f"Source: {item.source_name}",
            f"Source:",
            "source:",
        ]

        has_source = any(p.lower() in caption.lower() for p in source_patterns)

        if not has_source:
            # Add source line
            caption = caption.rstrip()
            if not caption.endswith('\n'):
                caption += '\n'
            caption += f"\nSource: {item.source_name} - {item.url}"

        # Clean up excessive newlines
        caption = re.sub(r'\n{3,}', '\n\n', caption)

        return caption.strip()


# Singleton writer
_writer: Optional[CaptionWriter] = None


def get_caption_writer() -> CaptionWriter:
    """Get singleton caption writer instance."""
    global _writer
    if _writer is None:
        _writer = CaptionWriter()
    return _writer


def write_caption(
    item: NormalizedItem,
    spine: NarrativeSpine,
    image_headline: str,
) -> CaptionResult:
    """Convenience function to write caption."""
    writer = get_caption_writer()
    return writer.write(item, spine, image_headline)
