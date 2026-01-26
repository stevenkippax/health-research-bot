"""
OpenAI-powered headline and image suggestion generator.

Creates viral on-image headlines and visual concepts for Instagram posts.
Enforces differentiator requirements - refuses to generate generic headlines.
"""

import json
from typing import Optional
from dataclasses import dataclass, field, asdict
import time

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import get_settings
from .logging_conf import get_logger
from .openai_eval import EvaluationResult

logger = get_logger(__name__)


@dataclass
class HeadlineResult:
    """Result of headline generation."""
    image_headline: Optional[str] = None
    reason: Optional[str] = None  # Reason if headline is null

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ImageSuggestionResult:
    """Result of image suggestion generation."""
    image_suggestion: str = ""
    layout_notes: list[str] = field(default_factory=list)
    highlight_words: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GenerationResult:
    """Combined generation result."""
    headline: HeadlineResult
    image: ImageSuggestionResult

    def to_dict(self) -> dict:
        return {
            "headline": self.headline.to_dict(),
            "image": self.image.to_dict(),
        }


# =====================================================
# STRICT HEADLINE SYSTEM PROMPT
# =====================================================
HEADLINE_SYSTEM_PROMPT = """You write short, viral on-image headlines for aging.ai and avoidaging Instagram pages.

CRITICAL RULES:
1. If NO differentiators are provided (numbers, population, time_window, comparison), output null for image_headline
2. MUST include at least one differentiator in the headline
3. NO exclamation marks
4. NO generic statements without specifics
5. Use correlational language for observational studies ("linked to", "associated with")
6. 1-2 sentences MAXIMUM - must be readable as large bold text on an image

HEADLINE REQUIREMENTS:
- Include the specific numbers/stats when provided
- Include population or time frame when it adds impact
- Plain language, no medical jargon
- Credible framing when appropriate ("New study:", "Harvard research:", "WHO:")

ARCHETYPE-SPECIFIC GUIDANCE:
- NEWS_POLICY: Lead with the authority ("WHO just announced...", "New CDC guidelines...")
- STUDY_STAT: Lead with the striking statistic ("Study of 50,000 people finds 27%...")
- WARNING_RISK: Frame as important awareness ("This common habit linked to 35% higher risk...")
- SIMPLE_HABIT: Focus on the actionable benefit with numbers ("20 minutes of X linked to...")
- IF_THEN: Clear conditional with specifics ("People who do X for 30 minutes have...")
- COUNTERINTUITIVE: Highlight the surprise ("The unexpected finding: X actually...")
- HUMAN_INTEREST: Personal angle with specifics ("This 95-year-old's daily habit...")

IMPORTANT: Return ONLY valid JSON. No markdown, no code blocks."""


HEADLINE_USER_TEMPLATE = """Generate an on-image headline from this evaluated finding:

ARCHETYPE: {archetype}
STUDY TYPE: {study_type}
MOST SURPRISING FINDING: {most_surprising_finding}
MUST INCLUDE NUMBERS: {must_include_numbers}
POPULATION: {population}
TIME WINDOW: {time_window}
SOURCE: {source_name}

Return JSON in this EXACT format:
{{
  "image_headline": "Your 1-2 sentence headline here, or null if insufficient differentiators",
  "reason": "If headline is null, explain why. Otherwise null."
}}

RULES:
- If must_include_numbers is empty AND population is null AND time_window is null, output null
- Include at least one differentiator (number, population, or time) in the headline
- No exclamation marks
- Use "linked to"/"associated with" for observational studies, not "causes" or "prevents\""""


# =====================================================
# IMAGE SUGGESTION SYSTEM PROMPT
# =====================================================
IMAGE_SYSTEM_PROMPT = """You propose concrete image concepts for Instagram posts about health and longevity.

Your suggestions should be:
- SIMPLE: Easy to find as a stock photo or create as a simple graphic
- VISUALLY CLEAR: One main subject, not cluttered
- EMOTIONALLY RESONANT: Hopeful, thought-provoking, or relatable
- APPROPRIATE: No graphic medical imagery, no negative imagery

For each headline, provide:
1. What to show (concrete visual description)
2. Layout notes (background type, text placement suggestions)
3. Words to highlight (for colored text overlay)

GOOD EXAMPLES:
- "Close-up of hands holding a coffee cup, warm morning light"
- "Split-screen: sleeping person (left) vs tired person at desk (right)"
- "Senior person jogging in a park, early morning, happy expression"

BAD EXAMPLES:
- "Abstract representation of health" (too vague)
- "Medical illustration of brain pathways" (too clinical)
- "Person lying in hospital bed" (too negative)

IMPORTANT: Return ONLY valid JSON. No markdown, no code blocks."""


IMAGE_USER_TEMPLATE = """Suggest an image concept for this headline:

HEADLINE: {headline}
ARCHETYPE: {archetype}
KEY NUMBERS TO HIGHLIGHT: {highlight_numbers}

Return JSON in this EXACT format:
{{
  "image_suggestion": "What to show: [concrete visual description]",
  "layout_notes": ["note about background", "note about text placement"],
  "highlight_words": ["word1", "word2", "number1"]
}}

The highlight_words should include key numbers and impactful words that could be shown in a contrasting color."""


class ContentGenerator:
    """
    Generates headlines and image suggestions using OpenAI.

    Enforces differentiator requirements - refuses generic headlines.
    """

    def __init__(self):
        """Initialize generator with OpenAI client."""
        settings = get_settings()
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.call_delay = settings.openai_call_delay

        logger.info("content_generator_initialized", model=self.model)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def generate_headline(
        self,
        evaluation: EvaluationResult,
        source_name: str,
    ) -> HeadlineResult:
        """
        Generate a viral on-image headline.

        Args:
            evaluation: Evaluation result with extracted data
            source_name: Name of the content source

        Returns:
            HeadlineResult with the generated headline (or null with reason)
        """
        # Check if we have required differentiators
        has_numbers = bool(evaluation.must_include_numbers)
        has_population = bool(evaluation.population and len(evaluation.population) > 3)
        has_time = bool(evaluation.time_window and len(evaluation.time_window) > 3)

        if not (has_numbers or has_population or has_time):
            logger.debug(
                "headline_rejected_no_differentiators",
                finding=evaluation.most_surprising_finding[:50] if evaluation.most_surprising_finding else None,
            )
            return HeadlineResult(
                image_headline=None,
                reason="No differentiators available (no numbers, population, or time window)",
            )

        user_prompt = HEADLINE_USER_TEMPLATE.format(
            archetype=evaluation.suggested_archetype or "STUDY_STAT",
            study_type=evaluation.study_type or "observational",
            most_surprising_finding=evaluation.most_surprising_finding or evaluation.extracted_claim or "",
            must_include_numbers=", ".join(evaluation.must_include_numbers) if evaluation.must_include_numbers else "None",
            population=evaluation.population or "Not specified",
            time_window=evaluation.time_window or "Not specified",
            source_name=source_name,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": HEADLINE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,  # Slightly higher for creativity
                max_tokens=300,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            data = json.loads(content)

            headline = data.get("image_headline")
            reason = data.get("reason")

            if not headline:
                return HeadlineResult(
                    image_headline=None,
                    reason=reason or "Generator returned null headline",
                )

            # Post-validation: check headline has a differentiator
            headline_lower = headline.lower()
            has_differentiator = any([
                any(num.lower() in headline_lower for num in evaluation.must_include_numbers) if evaluation.must_include_numbers else False,
                evaluation.population and evaluation.population.lower()[:10] in headline_lower if evaluation.population else False,
                evaluation.time_window and any(word in headline_lower for word in evaluation.time_window.lower().split()) if evaluation.time_window else False,
                any(char.isdigit() for char in headline),  # Any number
            ])

            if not has_differentiator:
                logger.warning(
                    "headline_missing_differentiator",
                    headline=headline[:60],
                )
                # Still return it but log the warning

            logger.debug("headline_generated", headline=headline[:50])

            time.sleep(self.call_delay)

            return HeadlineResult(image_headline=headline, reason=None)

        except json.JSONDecodeError as e:
            logger.error("headline_json_error", error=str(e))
            return HeadlineResult(
                image_headline=None,
                reason=f"JSON parse error: {str(e)}",
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def generate_image_suggestion(
        self,
        headline: str,
        archetype: str,
        numbers: list[str] = None,
    ) -> ImageSuggestionResult:
        """
        Generate an image concept suggestion.

        Args:
            headline: The generated headline
            archetype: Content archetype
            numbers: Numbers to potentially highlight

        Returns:
            ImageSuggestionResult with visual concept
        """
        user_prompt = IMAGE_USER_TEMPLATE.format(
            headline=headline,
            archetype=archetype,
            highlight_numbers=", ".join(numbers[:5]) if numbers else "None",
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": IMAGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=400,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            data = json.loads(content)

            suggestion = data.get("image_suggestion", "")

            if not suggestion:
                # Provide a fallback
                suggestion = "Simple, clean background with headline text overlay"

            logger.debug("image_suggestion_generated", suggestion=suggestion[:50])

            time.sleep(self.call_delay)

            return ImageSuggestionResult(
                image_suggestion=suggestion,
                layout_notes=data.get("layout_notes", []),
                highlight_words=data.get("highlight_words", []),
            )

        except json.JSONDecodeError as e:
            logger.error("image_json_error", error=str(e))
            return ImageSuggestionResult(
                image_suggestion="Clean gradient background with bold text overlay",
                layout_notes=["Center the text", "Use contrasting colors"],
                highlight_words=[],
            )

    def generate(
        self,
        evaluation: EvaluationResult,
        source_name: str,
    ) -> Optional[GenerationResult]:
        """
        Generate both headline and image suggestion.

        Args:
            evaluation: Evaluation result
            source_name: Name of content source

        Returns:
            GenerationResult with both headline and image, or None if headline failed
        """
        # Generate headline first
        headline_result = self.generate_headline(evaluation, source_name)

        if not headline_result.image_headline:
            logger.debug(
                "generation_skipped_no_headline",
                reason=headline_result.reason,
            )
            return None

        # Then generate image suggestion based on headline
        image_result = self.generate_image_suggestion(
            headline_result.image_headline,
            evaluation.suggested_archetype or "STUDY_STAT",
            evaluation.must_include_numbers,
        )

        return GenerationResult(
            headline=headline_result,
            image=image_result,
        )

    def generate_batch(
        self,
        items_with_evals: list[tuple],
        max_outputs: int = 5,
    ) -> list[tuple]:
        """
        Generate content for multiple items.

        Args:
            items_with_evals: List of (NormalizedItem, EvaluationResult) tuples
            max_outputs: Maximum number of outputs to generate

        Returns:
            List of (NormalizedItem, EvaluationResult, GenerationResult) tuples
        """
        logger.info(
            "starting_batch_generation",
            items=len(items_with_evals),
            max_outputs=max_outputs,
        )

        results = []
        generated_count = 0

        for item, evaluation in items_with_evals:
            if generated_count >= max_outputs:
                break

            try:
                generation = self.generate(evaluation, item.source_name)

                if generation is None:
                    logger.debug(
                        "generation_failed_no_headline",
                        title=item.title[:40],
                    )
                    continue

                results.append((item, evaluation, generation))
                generated_count += 1

                logger.info(
                    "content_generated",
                    title=item.title[:40],
                    headline=generation.headline.image_headline[:40] if generation.headline.image_headline else None,
                )

            except Exception as e:
                logger.error(
                    "generation_failed",
                    title=item.title[:50],
                    error=str(e),
                )
                continue

        logger.info("batch_generation_complete", total=len(results))

        return results
