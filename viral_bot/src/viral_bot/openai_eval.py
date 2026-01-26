"""
OpenAI-powered virality predictor.

Evaluates content items for viral potential on aging.ai/avoidaging.
"""

import json
from typing import Optional
from dataclasses import dataclass, field, asdict
import time

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import get_settings
from .logging_conf import get_logger
from .sources.base import FetchedItem

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Result of virality evaluation."""
    relevant: bool
    reason: str
    suggested_archetype: Optional[str] = None
    extracted_claim: Optional[str] = None
    virality_score: Optional[int] = None  # 0-100
    confidence: Optional[float] = None  # 0.0-1.0
    why_it_will_work: list[str] = field(default_factory=list)
    must_include_numbers: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)


# System prompt for the evaluator
EVALUATOR_SYSTEM_PROMPT = """You evaluate whether a health/science news item can become a viral on-image headline for the Instagram pages aging.ai and avoidaging.

These pages focus on aging, longevity, healthspan, and science-backed health information. Their audience values:
- Authority (WHO, NIH, CDC, peer-reviewed studies, long-term cohort studies)
- Quantification (percentages, risk multipliers, "equivalent to X years of aging")
- Emotional clarity (hope, concern, surprise) without fear-mongering
- Relatability (everyday habits, foods, decisions people can control)
- Compressibility into large bold text for Instagram images

EVALUATION RUBRIC (each contributes to virality_score):
1. Authority Signal (0-20 points): Is there a credible source? Government agency, major study, respected institution?
2. Quantification (0-20 points): Does it have striking numbers? Risk percentages, years of life, study size?
3. Emotional Hook (0-20 points): Does it evoke hope, concern, or surprise without being sensational?
4. Relatability (0-20 points): Can average people act on this? Is it about common habits/foods/activities?
5. Clarity/Compressibility (0-20 points): Can this be expressed in 1-2 bold sentences for an image?

IMPORTANT:
- You do NOT exaggerate or invent statistics
- You reject items that are too technical, too niche, or lack a clear actionable angle
- You prefer correlational language ("linked to", "associated with") for observational studies
- You flag if the source uses absolute claims that aren't justified

ARCHETYPES:
- NEWS_POLICY: Government announcements, WHO guidelines, policy changes
- STUDY_STAT: Research findings with striking statistics
- WARNING_RISK: "This common thing increases risk of..."
- SIMPLE_HABIT: Easy daily habits with health benefits
- IF_THEN: Conditional cause-effect ("If you do X, Y happens")
- COUNTERINTUITIVE: Surprising findings that challenge assumptions
- HUMAN_INTEREST: Stories about real people and longevity

Return ONLY valid JSON with no additional text."""


EVALUATOR_USER_TEMPLATE = """Evaluate this item for viral potential:

TITLE: {title}
SOURCE: {source_name}
DATE: {published_at}
URL: {url}
SUMMARY: {summary}

Return JSON in this exact format:
{{
  "relevant": true/false,
  "reason": "Brief explanation of relevance/irrelevance",
  "suggested_archetype": "NEWS_POLICY|STUDY_STAT|WARNING_RISK|SIMPLE_HABIT|IF_THEN|COUNTERINTUITIVE|HUMAN_INTEREST",
  "extracted_claim": "One-sentence factual claim to base headline on",
  "virality_score": 0-100,
  "confidence": 0.0-1.0,
  "why_it_will_work": ["reason1", "reason2", "reason3"],
  "must_include_numbers": ["27%", "50,000 participants", etc] or []
}}

If not relevant, set relevant=false and only fill reason field."""


class ViralityPredictor:
    """
    Predicts virality potential of content items using OpenAI.
    """
    
    VALID_ARCHETYPES = {
        "NEWS_POLICY",
        "STUDY_STAT",
        "WARNING_RISK",
        "SIMPLE_HABIT",
        "IF_THEN",
        "COUNTERINTUITIVE",
        "HUMAN_INTEREST",
    }
    
    def __init__(self):
        """Initialize predictor with OpenAI client."""
        settings = get_settings()
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.call_delay = settings.openai_call_delay
        
        logger.info("virality_predictor_initialized", model=self.model)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def evaluate(self, item: FetchedItem) -> EvaluationResult:
        """
        Evaluate a single item for viral potential.
        
        Args:
            item: Content item to evaluate
        
        Returns:
            EvaluationResult with scores and extracted data
        """
        # Build the prompt
        user_prompt = EVALUATOR_USER_TEMPLATE.format(
            title=item.title,
            source_name=item.source_name,
            published_at=item.published_at.isoformat() if item.published_at else "Unknown",
            url=item.url,
            summary=item.summary or "No summary available",
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,  # Lower temperature for more consistent scoring
                max_tokens=1000,
                response_format={"type": "json_object"},
            )
            
            # Parse response
            content = response.choices[0].message.content
            data = json.loads(content)
            
            # Validate and build result
            result = self._parse_response(data)
            
            logger.debug(
                "evaluation_complete",
                title=item.title[:50],
                relevant=result.relevant,
                score=result.virality_score,
            )
            
            # Rate limiting delay
            time.sleep(self.call_delay)
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error("evaluation_json_error", error=str(e), content=content[:200])
            return EvaluationResult(
                relevant=False,
                reason=f"Failed to parse AI response: {str(e)}",
            )
        except Exception as e:
            logger.error("evaluation_error", error=str(e))
            raise
    
    def evaluate_batch(
        self,
        items: list[FetchedItem],
        min_score: int = 40,
    ) -> list[tuple[FetchedItem, EvaluationResult]]:
        """
        Evaluate multiple items.
        
        Args:
            items: Items to evaluate
            min_score: Minimum score to include in results
        
        Returns:
            List of (item, result) tuples, sorted by virality score
        """
        logger.info("starting_batch_evaluation", items=len(items))
        
        results = []
        
        for item in items:
            try:
                result = self.evaluate(item)
                
                # Only include relevant items above threshold
                if result.relevant and result.virality_score is not None:
                    if result.virality_score >= min_score:
                        results.append((item, result))
                    else:
                        logger.debug(
                            "item_below_threshold",
                            title=item.title[:50],
                            score=result.virality_score,
                        )
                
            except Exception as e:
                logger.error(
                    "item_evaluation_failed",
                    title=item.title[:50],
                    error=str(e),
                )
                continue
        
        # Sort by virality score (descending)
        results.sort(key=lambda x: x[1].virality_score or 0, reverse=True)
        
        logger.info(
            "batch_evaluation_complete",
            total=len(items),
            relevant=len(results),
        )
        
        return results
    
    def _parse_response(self, data: dict) -> EvaluationResult:
        """Parse and validate AI response."""
        relevant = bool(data.get("relevant", False))
        reason = str(data.get("reason", "No reason provided"))
        
        if not relevant:
            return EvaluationResult(relevant=False, reason=reason)
        
        # Validate archetype
        archetype = data.get("suggested_archetype")
        if archetype and archetype not in self.VALID_ARCHETYPES:
            logger.warning("invalid_archetype", archetype=archetype)
            archetype = None
        
        # Validate score
        score = data.get("virality_score")
        if score is not None:
            score = max(0, min(100, int(score)))
        
        # Validate confidence
        confidence = data.get("confidence")
        if confidence is not None:
            confidence = max(0.0, min(1.0, float(confidence)))
        
        return EvaluationResult(
            relevant=True,
            reason=reason,
            suggested_archetype=archetype,
            extracted_claim=data.get("extracted_claim"),
            virality_score=score,
            confidence=confidence,
            why_it_will_work=data.get("why_it_will_work", []),
            must_include_numbers=data.get("must_include_numbers", []),
        )
