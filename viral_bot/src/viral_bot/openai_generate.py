"""
OpenAI-powered headline and image suggestion generator.

Creates viral on-image headlines and visual concepts for Instagram posts.
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
    image_headline: str
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ImageSuggestionResult:
    """Result of image suggestion generation."""
    image_suggestion: str
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


# System prompt for headline generation
HEADLINE_SYSTEM_PROMPT = """You write short, viral on-image headlines for the Instagram pages aging.ai and avoidaging.

HEADLINE REQUIREMENTS:
- 1-2 sentences maximum
- Easy to read in LARGE BOLD TEXT on an image
- Plain language, no jargon
- Include credible framing when appropriate (e.g., "New study:", "WHO:", "Harvard research:")
- Include numbers ONLY if provided in must_include_numbers
- No clickbait exaggeration - stick to the facts

ARCHETYPE-SPECIFIC GUIDANCE:
- NEWS_POLICY: Lead with the authority ("WHO just announced...")
- STUDY_STAT: Lead with the striking number ("Study of 50,000 people finds...")
- WARNING_RISK: Frame as important awareness ("This common habit linked to...")
- SIMPLE_HABIT: Focus on the actionable benefit ("One daily habit that...")
- IF_THEN: Clear conditional ("If you do X before bed, your brain...")
- COUNTERINTUITIVE: Highlight the surprise ("The unexpected food that...")
- HUMAN_INTEREST: Personal angle ("This 95-year-old's secret...")

Return ONLY valid JSON with no additional text."""


HEADLINE_USER_TEMPLATE = """Generate an on-image headline:

ARCHETYPE: {archetype}
EXTRACTED CLAIM: {extracted_claim}
SOURCE: {source_name}
MUST INCLUDE NUMBERS: {must_include_numbers}

Return JSON:
{{"image_headline": "Your headline here"}}"""


# System prompt for image suggestion
IMAGE_SYSTEM_PROMPT = """You propose concrete image concepts for Instagram posts about health and longevity.

Your suggestions should be:
- Simple and visually clear
- Easy to create (stock photos, simple graphics, or photo composites)
- Emotionally resonant (hopeful, thought-provoking, relatable)
- Appropriate for health content (no graphic medical imagery)

For each headline, suggest:
1. What to show (e.g., "close-up of hands holding vegetables", "person jogging at sunrise", "split-screen of brain scans")
2. Optional layout notes (background type, icon placement)
3. Optional words to highlight in color (for text overlay)

DO NOT generate actual images. Just describe the concept.

Return ONLY valid JSON with no additional text."""


IMAGE_USER_TEMPLATE = """Suggest an image concept for this headline:

HEADLINE: {headline}
ARCHETYPE: {archetype}

Return JSON:
{{
  "image_suggestion": "What to show: ...",
  "layout_notes": ["optional note 1", "optional note 2"],
  "highlight_words": ["word1", "word2"]
}}"""


class ContentGenerator:
    """
    Generates headlines and image suggestions using OpenAI.
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
            evaluation: Evaluation result with extracted claim and archetype
            source_name: Name of the content source
        
        Returns:
            HeadlineResult with the generated headline
        """
        user_prompt = HEADLINE_USER_TEMPLATE.format(
            archetype=evaluation.suggested_archetype or "STUDY_STAT",
            extracted_claim=evaluation.extracted_claim or "",
            source_name=source_name,
            must_include_numbers=", ".join(evaluation.must_include_numbers) or "None",
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": HEADLINE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,  # Slightly higher for creativity
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            
            headline = data.get("image_headline", "")
            
            if not headline:
                raise ValueError("Empty headline returned")
            
            logger.debug("headline_generated", headline=headline[:50])
            
            time.sleep(self.call_delay)
            
            return HeadlineResult(image_headline=headline)
            
        except json.JSONDecodeError as e:
            logger.error("headline_json_error", error=str(e))
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def generate_image_suggestion(
        self,
        headline: str,
        archetype: str,
    ) -> ImageSuggestionResult:
        """
        Generate an image concept suggestion.
        
        Args:
            headline: The generated headline
            archetype: Content archetype
        
        Returns:
            ImageSuggestionResult with visual concept
        """
        user_prompt = IMAGE_USER_TEMPLATE.format(
            headline=headline,
            archetype=archetype,
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": IMAGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            
            suggestion = data.get("image_suggestion", "")
            
            if not suggestion:
                raise ValueError("Empty image suggestion returned")
            
            logger.debug("image_suggestion_generated", suggestion=suggestion[:50])
            
            time.sleep(self.call_delay)
            
            return ImageSuggestionResult(
                image_suggestion=suggestion,
                layout_notes=data.get("layout_notes", []),
                highlight_words=data.get("highlight_words", []),
            )
            
        except json.JSONDecodeError as e:
            logger.error("image_json_error", error=str(e))
            raise
    
    def generate(
        self,
        evaluation: EvaluationResult,
        source_name: str,
    ) -> GenerationResult:
        """
        Generate both headline and image suggestion.
        
        Args:
            evaluation: Evaluation result
            source_name: Name of content source
        
        Returns:
            GenerationResult with both headline and image
        """
        # Generate headline first
        headline_result = self.generate_headline(evaluation, source_name)
        
        # Then generate image suggestion based on headline
        image_result = self.generate_image_suggestion(
            headline_result.image_headline,
            evaluation.suggested_archetype or "STUDY_STAT",
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
            items_with_evals: List of (FetchedItem, EvaluationResult) tuples
            max_outputs: Maximum number of outputs to generate
        
        Returns:
            List of (FetchedItem, EvaluationResult, GenerationResult) tuples
        """
        logger.info(
            "starting_batch_generation",
            items=len(items_with_evals),
            max_outputs=max_outputs,
        )
        
        results = []
        
        for item, evaluation in items_with_evals[:max_outputs]:
            try:
                generation = self.generate(evaluation, item.source_name)
                results.append((item, evaluation, generation))
                
                logger.info(
                    "content_generated",
                    title=item.title[:40],
                    headline=generation.headline.image_headline[:40],
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
