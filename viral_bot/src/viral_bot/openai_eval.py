"""
OpenAI-powered virality predictor with anti-generic requirements.

Evaluates content items for viral potential on aging.ai/avoidaging.
Requires specific, novel, concrete hooks - rejects generic health truisms.
"""

import json
from typing import Optional
from dataclasses import dataclass, field, asdict
import time

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import get_settings
from .logging_conf import get_logger
from .normalize import NormalizedItem

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Result of virality evaluation with anti-generic fields."""
    relevant: bool
    reason: str
    rejection_reason: Optional[str] = None

    # Study/source metadata
    study_type: Optional[str] = None  # RCT, observational, meta_analysis, review, policy, news, other
    population: Optional[str] = None
    sample_size: Optional[str] = None
    time_window: Optional[str] = None
    primary_outcome: Optional[str] = None

    # Core finding
    most_surprising_finding: Optional[str] = None
    extracted_claim: Optional[str] = None  # Legacy field for compatibility

    # Differentiators
    must_include_numbers: list[str] = field(default_factory=list)

    # Archetype and scoring
    suggested_archetype: Optional[str] = None
    virality_score: Optional[int] = None  # 0-100
    confidence: Optional[float] = None  # 0.0-1.0
    why_it_will_work: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def has_required_differentiators(self) -> bool:
        """Check if at least 2 differentiators are present."""
        count = sum([
            bool(self.must_include_numbers),
            bool(self.population and len(self.population) > 3),
            bool(self.time_window and len(self.time_window) > 3),
            bool(self.sample_size),
        ])
        return count >= 2


# =====================================================
# STRICT EVALUATOR SYSTEM PROMPT
# =====================================================
EVALUATOR_SYSTEM_PROMPT = """You are a virality analyst for aging.ai / avoidaging Instagram pages.

Your job is to find SPECIFIC, NOVEL, IMAGE-HEADLINE-WORTHY hooks from credible sources.

CRITICAL RULES:
1. REJECT generic health truisms ("exercise is good", "sleep is important", "eat healthy")
2. REQUIRE concrete, surprising, or quantified findings
3. Base ALL outputs on concrete details from the provided text - DO NOT invent numbers or statistics
4. For observational studies, use correlational language ("linked to", "associated with") not causal claims
5. If the provided text lacks specific differentiators (numbers, populations, timeframes), mark as NOT relevant

WHAT MAKES CONTENT VIRAL (in order of importance):
1. SPECIFICITY: "27% reduction" beats "reduces risk"; "adults 50-70" beats "older adults"
2. SURPRISE: Counterintuitive findings, unexpected comparisons ("equivalent to smoking 15 cigarettes")
3. AUTHORITY: WHO, NIH, major journals, large cohort studies
4. ACTIONABILITY: Something people can actually do differently
5. URGENCY: Time-sensitive or affects a large population

STUDY TYPE CLASSIFICATION:
- RCT: Randomized controlled trial - highest quality, can imply causation
- observational: Cohort, case-control, cross-sectional - correlations only
- meta_analysis: Systematic review of multiple studies - strong evidence
- review: Narrative review - opinion/summary
- policy: Government announcement, guidelines
- news: News report about health topic
- other: Anything else

ARCHETYPES (choose best fit):
- NEWS_POLICY: Government announcements, WHO guidelines, policy changes
- STUDY_STAT: Research findings with striking statistics (most common for papers)
- WARNING_RISK: "This common thing linked to increased risk..."
- SIMPLE_HABIT: Easy daily habits with specific benefits
- IF_THEN: Conditional cause-effect with numbers
- COUNTERINTUITIVE: Surprising findings that challenge assumptions
- HUMAN_INTEREST: Stories about real people and longevity

IMPORTANT: Return ONLY valid JSON. No markdown formatting, no code blocks, just JSON."""


# =====================================================
# STRICT EVALUATOR USER PROMPT
# =====================================================
EVALUATOR_USER_TEMPLATE = """Evaluate this content for viral headline potential.

SOURCE: {source_name}
URL: {url}
DATE: {published_at}

{full_text}

---

Analyze the above and return JSON in this EXACT format:

{{
  "relevant": true or false,
  "rejection_reason": "string explaining why not relevant, or null if relevant",
  "study_type": "RCT|observational|meta_analysis|review|policy|news|other",
  "population": "specific population description or null",
  "sample_size": "number of participants or null",
  "time_window": "study duration/follow-up or null",
  "primary_outcome": "main outcome measured or null",
  "most_surprising_finding": "one sentence grounded ONLY in the provided text, or null",
  "must_include_numbers": ["list", "of", "specific", "numbers", "from", "text"] or [],
  "suggested_archetype": "NEWS_POLICY|STUDY_STAT|WARNING_RISK|SIMPLE_HABIT|IF_THEN|COUNTERINTUITIVE|HUMAN_INTEREST",
  "virality_score": 0-100,
  "confidence": 0.0-1.0,
  "why_it_will_work": ["reason1", "reason2", "reason3"]
}}

REQUIREMENTS:
- If relevant=false, only fill rejection_reason, set other fields to null/empty
- most_surprising_finding must quote or closely paraphrase the source, never invent
- must_include_numbers should contain EXACT numbers from the text (e.g., "27%", "50,000 participants", "10-year follow-up")
- virality_score: 0-40 = not worth posting, 41-60 = maybe, 61-80 = good, 81-100 = excellent
- Set relevant=false if the content is generic without specific numbers or findings"""


class ViralityPredictor:
    """
    Predicts virality potential of content items using OpenAI.

    Enforces anti-generic requirements and extracts structured data.
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

    VALID_STUDY_TYPES = {
        "RCT",
        "observational",
        "meta_analysis",
        "review",
        "policy",
        "news",
        "other",
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
    def evaluate(self, item: NormalizedItem) -> EvaluationResult:
        """
        Evaluate a single item for viral potential.

        Args:
            item: NormalizedItem to evaluate (with full body text)

        Returns:
            EvaluationResult with scores, findings, and differentiators
        """
        # Build the prompt with full text
        full_text = item.get_full_text_for_evaluation()

        user_prompt = EVALUATOR_USER_TEMPLATE.format(
            source_name=item.source_name,
            url=item.url,
            published_at=item.published_at.isoformat() if item.published_at else "Unknown",
            full_text=full_text[:6000],  # Limit to avoid token issues
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,  # Lower temperature for consistent scoring
                max_tokens=1500,
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
                finding=result.most_surprising_finding[:50] if result.most_surprising_finding else None,
            )

            # Rate limiting delay
            time.sleep(self.call_delay)

            return result

        except json.JSONDecodeError as e:
            logger.error("evaluation_json_error", error=str(e), content=content[:200])

            # Try to fix JSON and retry once
            try:
                fixed_content = self._attempt_json_fix(content)
                data = json.loads(fixed_content)
                return self._parse_response(data)
            except Exception:
                return EvaluationResult(
                    relevant=False,
                    reason=f"Failed to parse AI response: {str(e)}",
                    rejection_reason="json_parse_error",
                )

        except Exception as e:
            logger.error("evaluation_error", error=str(e))
            raise

    def evaluate_batch(
        self,
        items: list[NormalizedItem],
        min_score: int = 40,
    ) -> list[tuple[NormalizedItem, EvaluationResult]]:
        """
        Evaluate multiple items.

        Args:
            items: NormalizedItems to evaluate
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
        reason = str(data.get("rejection_reason") or data.get("reason", "No reason provided"))

        if not relevant:
            return EvaluationResult(
                relevant=False,
                reason=reason,
                rejection_reason=reason,
            )

        # Validate archetype
        archetype = data.get("suggested_archetype")
        if archetype and archetype not in self.VALID_ARCHETYPES:
            logger.warning("invalid_archetype", archetype=archetype)
            archetype = "STUDY_STAT"  # Default

        # Validate study type
        study_type = data.get("study_type")
        if study_type and study_type not in self.VALID_STUDY_TYPES:
            logger.warning("invalid_study_type", study_type=study_type)
            study_type = "other"

        # Validate score
        score = data.get("virality_score")
        if score is not None:
            score = max(0, min(100, int(score)))

        # Validate confidence
        confidence = data.get("confidence")
        if confidence is not None:
            confidence = max(0.0, min(1.0, float(confidence)))

        # Extract finding - use most_surprising_finding, fall back to older field
        most_surprising = data.get("most_surprising_finding")
        extracted_claim = data.get("extracted_claim") or most_surprising

        return EvaluationResult(
            relevant=True,
            reason=reason if reason != "No reason provided" else "Content is relevant",
            rejection_reason=None,
            study_type=study_type,
            population=data.get("population"),
            sample_size=data.get("sample_size"),
            time_window=data.get("time_window"),
            primary_outcome=data.get("primary_outcome"),
            most_surprising_finding=most_surprising,
            extracted_claim=extracted_claim,
            must_include_numbers=data.get("must_include_numbers", []),
            suggested_archetype=archetype,
            virality_score=score,
            confidence=confidence,
            why_it_will_work=data.get("why_it_will_work", []),
        )

    def _attempt_json_fix(self, content: str) -> str:
        """Attempt to fix malformed JSON."""
        # Try to extract JSON from markdown code blocks
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if json_match:
            return json_match.group(1)

        # Try to find JSON object
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            return json_match.group(0)

        return content


# Legacy compatibility: Support both NormalizedItem and FetchedItem
def evaluate_fetched_item(predictor: ViralityPredictor, item) -> EvaluationResult:
    """
    Evaluate a FetchedItem (legacy compatibility).

    Converts FetchedItem to NormalizedItem for evaluation.
    """
    from .normalize import from_fetched_item, ContentType

    # Convert to NormalizedItem
    normalized = from_fetched_item(
        item,
        body_text=item.summary or "",
        content_type=ContentType.NEWS,
    )

    return predictor.evaluate(normalized)
