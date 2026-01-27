"""
Main orchestration module with story-compression pipeline.

Coordinates the full workflow:
1. Fetch content from tiered sources
2. Fetch full article/abstract text
3. Pre-AI generic filter
4. Deduplicate (URL + semantic)
5. Extract narrative spine (AI Stage #1)
6. Quality gate on narrative
7. Story compression (AI Stage #2)
8. Headline quality gate
9. Novelty check against history
10. Mix outputs for diversity (max 2 STUDY_STAT)
11. Export to Google Sheets

Produces detailed run reports for debugging.
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional
import uuid
from collections import defaultdict

from .config import get_settings
from .logging_conf import get_logger, bind_context, setup_logging
from .db import get_database, ContentItem, Evaluation, Output
from .sources import get_source_registry, FetchedItem
from .sources.registry import CredibilityTier
from .normalize import NormalizedItem, ContentType
from .content_fetch import ContentFetcher
from .anti_generic import (
    GenericFilter,
    check_narrative_quality,
    check_headline_quality,
    ArchetypeDiversityEnforcer,
    format_differentiator_summary,
)
from .novelty import NoveltyChecker, load_novelty_data_from_db
from .mixer import OutputMixer, MixingConfig
from .dedupe import Deduplicator, load_recent_data_from_db
from .narrative_extractor import NarrativeExtractor, NarrativeSpine
from .story_generator import (
    StoryCompressor,
    StoryCompressionResult,
    story_compression_to_generation_result,
)
from .sheets import SheetsExporter

logger = get_logger(__name__)


class RunReport:
    """Tracks statistics and examples for run reporting."""

    def __init__(self):
        self.stats = defaultdict(int)
        self.rejection_reasons = defaultdict(int)
        self.rejection_examples = []
        self.accepted_items = []
        self.tier_distribution = defaultdict(int)

    def log_rejection(self, stage: str, reason: str, title: str):
        """Log a rejection."""
        self.rejection_reasons[f"{stage}:{reason}"] += 1
        if len(self.rejection_examples) < 10:
            self.rejection_examples.append({
                "stage": stage,
                "reason": reason,
                "title": title[:100],
            })

    def log_acceptance(self, item, spine: NarrativeSpine, result: StoryCompressionResult):
        """Log an accepted item."""
        self.accepted_items.append({
            "title": item.title[:100] if hasattr(item, 'title') else str(item)[:100],
            "headline": result.headline[:80],
            "archetype": spine.content_archetype,
            "clarity_score": spine.standalone_clarity_score,
            "emotional_hook": spine.emotional_hook,
            "tier": item.credibility_tier.value if hasattr(item, 'credibility_tier') else "?",
        })
        # Track tier distribution
        if hasattr(item, 'credibility_tier'):
            self.tier_distribution[item.credibility_tier.value] += 1

    def get_summary(self) -> str:
        """Get formatted summary."""
        lines = [
            "=" * 60,
            "RUN REPORT - STORY COMPRESSION PIPELINE",
            "=" * 60,
            "",
            "PIPELINE STATISTICS:",
        ]

        for key, value in sorted(self.stats.items()):
            lines.append(f"  {key}: {value}")

        lines.extend([
            "",
            "TIER DISTRIBUTION:",
        ])
        for tier, count in sorted(self.tier_distribution.items()):
            lines.append(f"  Tier {tier}: {count}")

        lines.extend([
            "",
            "REJECTION REASONS:",
        ])

        for reason, count in sorted(self.rejection_reasons.items(), key=lambda x: -x[1]):
            lines.append(f"  {reason}: {count}")

        if self.rejection_examples:
            lines.extend([
                "",
                "REJECTION EXAMPLES:",
            ])
            for ex in self.rejection_examples[:5]:
                lines.append(f"  [{ex['stage']}] {ex['reason']}")
                lines.append(f"    Title: {ex['title']}")

        if self.accepted_items:
            lines.extend([
                "",
                "ACCEPTED HEADLINES:",
            ])
            for item in self.accepted_items[:5]:
                lines.append(f"  [{item['archetype']}] Tier {item['tier']}, Clarity: {item['clarity_score']}")
                lines.append(f"    {item['headline']}")

        lines.append("=" * 60)
        return "\n".join(lines)


class BotRunner:
    """
    Main bot runner with story-compression pipeline.
    """

    def __init__(self):
        """Initialize all components."""
        self.settings = get_settings()
        self.db = get_database()
        self.registry = get_source_registry()
        self.content_fetcher = ContentFetcher(
            timeout=self.settings.content_fetch_timeout,
            max_concurrent=self.settings.max_concurrent_content_fetches,
        )
        self.generic_filter = GenericFilter(
            min_body_chars_paper=self.settings.min_body_chars_paper,
            min_body_chars_news=self.settings.min_body_chars_news,
            min_differentiators=self.settings.min_differentiators,
        )
        self.narrative_extractor = NarrativeExtractor()
        self.story_compressor = StoryCompressor()
        self.exporter = SheetsExporter()

        logger.info("bot_runner_initialized")

    def generate_run_id(self) -> str:
        """Generate a unique run ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:8]
        return f"run_{timestamp}_{short_uuid}"

    async def run(
        self,
        freshness_hours: Optional[int] = None,
        max_outputs: Optional[int] = None,
        dry_run: bool = False,
    ) -> dict:
        """
        Execute one full bot run with story-compression pipeline.

        Args:
            freshness_hours: Override default freshness window
            max_outputs: Override default max outputs
            dry_run: If True, don't write to Sheets

        Returns:
            Run statistics dictionary
        """
        run_id = self.generate_run_id()
        bind_context(run_id=run_id)

        freshness_hours = freshness_hours or self.settings.freshness_hours
        max_outputs = max_outputs or self.settings.max_outputs_per_run

        logger.info(
            "run_started",
            freshness_hours=freshness_hours,
            max_outputs=max_outputs,
            dry_run=dry_run,
        )

        report = RunReport()
        stats = {
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "items_fetched": 0,
            "items_with_content": 0,
            "items_after_pre_filter": 0,
            "items_after_dedup": 0,
            "items_with_narrative": 0,
            "items_after_narrative_gate": 0,
            "items_with_headline": 0,
            "items_after_headline_gate": 0,
            "items_after_novelty": 0,
            "items_after_diversity": 0,
            "items_exported": 0,
            "errors": [],
            "dry_run": dry_run,
        }

        session = self.db.get_session()

        try:
            # Record run start
            self.db.start_run(session, run_id)

            # === STEP 1: Fetch content from tiered sources ===
            logger.info("step_1_fetching_content")
            report.stats["step"] = 1

            fetched_items = await self.registry.fetch_all(
                freshness_hours=freshness_hours,
                max_concurrent=self.settings.max_concurrent_fetches,
            )
            stats["items_fetched"] = len(fetched_items)
            report.stats["fetched"] = len(fetched_items)

            if not fetched_items:
                logger.warning("no_items_fetched")
                return self._complete_run(session, run_id, stats, report)

            # === STEP 2: Fetch full content (abstracts/articles) ===
            logger.info("step_2_fetching_full_content")
            report.stats["step"] = 2

            normalized_items = await self.content_fetcher.fetch_batch(
                fetched_items,
                min_body_chars_paper=self.settings.min_body_chars_paper,
                min_body_chars_news=self.settings.min_body_chars_news,
            )
            stats["items_with_content"] = len(normalized_items)
            report.stats["with_content"] = len(normalized_items)

            # Log items rejected for insufficient content
            content_rejected = len(fetched_items) - len(normalized_items)
            if content_rejected > 0:
                report.rejection_reasons["content_fetch:insufficient_body"] = content_rejected

            if not normalized_items:
                logger.warning("no_items_with_sufficient_content")
                return self._complete_run(session, run_id, stats, report)

            # === STEP 3: Pre-AI generic filter ===
            if self.settings.enable_pre_ai_filter:
                logger.info("step_3_pre_ai_filter")
                report.stats["step"] = 3

                filtered_items = []
                for item in normalized_items:
                    result = self.generic_filter.pre_ai_filter(item)
                    if result.passed:
                        filtered_items.append(item)
                    else:
                        report.log_rejection("pre_ai", result.reason or "unknown", item.title)

                stats["items_after_pre_filter"] = len(filtered_items)
                report.stats["after_pre_filter"] = len(filtered_items)
                normalized_items = filtered_items
            else:
                stats["items_after_pre_filter"] = len(normalized_items)
                report.stats["after_pre_filter"] = len(normalized_items)

            if not normalized_items:
                logger.warning("no_items_after_pre_filter")
                return self._complete_run(session, run_id, stats, report)

            # === STEP 4: Deduplicate ===
            logger.info("step_4_deduplicating")
            report.stats["step"] = 4

            # Load recent data for dedup
            recent_urls, recent_embeddings = load_recent_data_from_db(
                self.db, session,
                days=self.settings.dedup_retention_days,
            )

            deduplicator = Deduplicator(
                similarity_threshold=self.settings.dedup_similarity_threshold,
                recent_urls=recent_urls,
                recent_embeddings=recent_embeddings,
            )

            unique_normalized = self._dedupe_normalized(deduplicator, normalized_items)
            stats["items_after_dedup"] = len(unique_normalized)
            report.stats["after_dedup"] = len(unique_normalized)

            dedup_rejected = len(normalized_items) - len(unique_normalized)
            if dedup_rejected > 0:
                report.rejection_reasons["dedup:duplicate"] = dedup_rejected

            if not unique_normalized:
                logger.warning("no_items_after_dedup")
                return self._complete_run(session, run_id, stats, report)

            # === STEP 5: Extract Narrative Spine (AI Stage #1) ===
            logger.info("step_5_narrative_extraction")
            report.stats["step"] = 5

            items_with_spines = []
            for item in unique_normalized:
                spine = self.narrative_extractor.extract(item)
                if spine:
                    items_with_spines.append((item, spine))
                else:
                    report.log_rejection("narrative", "extraction_failed", item.title)

            stats["items_with_narrative"] = len(items_with_spines)
            report.stats["with_narrative"] = len(items_with_spines)

            if not items_with_spines:
                logger.warning("no_items_with_narrative")
                return self._complete_run(session, run_id, stats, report)

            # === STEP 6: Narrative Quality Gate ===
            logger.info("step_6_narrative_quality_gate")
            report.stats["step"] = 6

            quality_passed = []
            for item, spine in items_with_spines:
                result = check_narrative_quality(
                    spine,
                    min_clarity_score=self.settings.min_clarity_score,
                    require_emotional_hook=True,
                )
                if result.passed:
                    quality_passed.append((item, spine))
                else:
                    report.log_rejection("narrative_quality", result.reason or "unknown", item.title)

            stats["items_after_narrative_gate"] = len(quality_passed)
            report.stats["after_narrative_gate"] = len(quality_passed)

            if not quality_passed:
                logger.warning("no_items_after_narrative_gate")
                return self._complete_run(session, run_id, stats, report)

            # === STEP 7: Story Compression (AI Stage #2) ===
            logger.info("step_7_story_compression")
            report.stats["step"] = 7

            items_with_headlines = []
            for item, spine in quality_passed:
                result = self.story_compressor.compress(item, spine)
                if result:
                    items_with_headlines.append((item, spine, result))
                else:
                    report.log_rejection("compression", "generation_failed", item.title)

            stats["items_with_headline"] = len(items_with_headlines)
            report.stats["with_headline"] = len(items_with_headlines)

            if not items_with_headlines:
                logger.warning("no_items_with_headline")
                return self._complete_run(session, run_id, stats, report)

            # === STEP 8: Headline Quality Gate ===
            logger.info("step_8_headline_quality_gate")
            report.stats["step"] = 8

            headline_passed = []
            for item, spine, result in items_with_headlines:
                quality_result = check_headline_quality(result.headline)
                if quality_result.passed:
                    headline_passed.append((item, spine, result))
                else:
                    report.log_rejection("headline_quality", quality_result.reason or "unknown", item.title)

            stats["items_after_headline_gate"] = len(headline_passed)
            report.stats["after_headline_gate"] = len(headline_passed)

            if not headline_passed:
                logger.warning("no_items_after_headline_gate")
                return self._complete_run(session, run_id, stats, report)

            # === STEP 9: Novelty Check ===
            logger.info("step_9_novelty_check")
            report.stats["step"] = 9

            recent_findings = load_novelty_data_from_db(
                self.db, session,
                days=self.settings.novelty_retention_days,
            )

            novelty_checker = NoveltyChecker(
                similarity_threshold=self.settings.novelty_threshold,
                use_embeddings=self.settings.novelty_use_embeddings,
                penalty_factor=self.settings.novelty_penalty_factor,
            )
            novelty_checker.load_recent_findings(recent_findings)

            novelty_passed = []
            for item, spine, result in headline_passed:
                finding = spine.hook or result.headline
                novelty_result, _ = novelty_checker.check_novelty(finding, 1.0)

                if novelty_result.is_novel:
                    novelty_passed.append((item, spine, result))
                    novelty_checker.add_finding(finding)
                else:
                    report.log_rejection(
                        "novelty",
                        f"too_similar ({novelty_result.similarity_score:.2f})",
                        item.title,
                    )

            stats["items_after_novelty"] = len(novelty_passed)
            report.stats["after_novelty"] = len(novelty_passed)

            if not novelty_passed:
                logger.warning("no_items_after_novelty")
                return self._complete_run(session, run_id, stats, report)

            # === STEP 10: Output Diversity (max 2 STUDY_STAT) ===
            logger.info("step_10_diversity_enforcement")
            report.stats["step"] = 10

            diversity_enforcer = ArchetypeDiversityEnforcer(
                max_study_stat=self.settings.max_study_stat_per_run,
                max_per_archetype=self.settings.max_per_archetype,
            )

            # Sort by clarity score (higher is better)
            novelty_passed.sort(key=lambda x: x[1].standalone_clarity_score, reverse=True)

            final_outputs = []
            for item, spine, result in novelty_passed:
                if len(final_outputs) >= max_outputs:
                    break

                can_add, reason = diversity_enforcer.can_add(spine.content_archetype)
                if can_add:
                    diversity_enforcer.add(spine.content_archetype)
                    final_outputs.append((item, spine, result))
                    report.log_acceptance(item, spine, result)
                else:
                    report.log_rejection("diversity", reason or "unknown", item.title)

            stats["items_after_diversity"] = len(final_outputs)
            report.stats["after_diversity"] = len(final_outputs)
            report.stats["archetype_distribution"] = diversity_enforcer.get_counts()

            # Save to database
            self._save_to_database(session, run_id, final_outputs)
            session.commit()

            # === STEP 11: Export to Google Sheets ===
            if not dry_run and final_outputs:
                logger.info("step_11_exporting_to_sheets")
                report.stats["step"] = 11

                try:
                    # Convert to legacy format for sheets exporter
                    export_data = []
                    for item, spine, result in final_outputs:
                        gen_result = story_compression_to_generation_result(result)
                        # Create a mock evaluation for backward compat
                        mock_eval = self._spine_to_mock_eval(spine)
                        export_data.append((item, mock_eval, gen_result))

                    exported = self.exporter.export_outputs_v2(run_id, final_outputs)
                    stats["items_exported"] = exported
                    report.stats["exported"] = exported
                except Exception as e:
                    logger.error("sheets_export_failed", error=str(e))
                    stats["errors"].append(f"Sheets export failed: {str(e)}")
            else:
                logger.info("dry_run_skipping_sheets_export")
                stats["items_exported"] = 0

            return self._complete_run(session, run_id, stats, report)

        except Exception as e:
            logger.error("run_failed", error=str(e))
            stats["errors"].append(str(e))
            stats["status"] = "FAILED"

            self.db.complete_run(
                session, run_id, "FAILED",
                items_fetched=stats["items_fetched"],
                items_evaluated=stats.get("items_with_narrative", 0),
                items_output=0,
                error_message=str(e),
            )

            raise

        finally:
            session.close()

    def _spine_to_mock_eval(self, spine: NarrativeSpine):
        """Create a mock evaluation object from a spine for backward compat."""
        from dataclasses import dataclass, field
        from typing import Optional

        @dataclass
        class MockEval:
            relevant: bool = True
            reason: Optional[str] = None
            virality_score: int = 8
            confidence: float = 0.8
            suggested_archetype: str = ""
            extracted_claim: str = ""
            most_surprising_finding: str = ""
            why_it_will_work: list = field(default_factory=list)
            must_include_numbers: list = field(default_factory=list)
            population: str = ""
            time_window: str = ""
            study_type: str = ""

        return MockEval(
            suggested_archetype=spine.content_archetype,
            extracted_claim=spine.hook,
            most_surprising_finding=spine.hook,
            why_it_will_work=[spine.real_world_consequence],
            must_include_numbers=spine.key_numbers,
            population=spine.who_it_applies_to,
            time_window=spine.time_window,
            study_type=spine.support_level,
        )

    def _save_to_database(
        self,
        session,
        run_id: str,
        outputs: list[tuple[NormalizedItem, NarrativeSpine, StoryCompressionResult]],
    ):
        """Save outputs to database."""
        for item, spine, result in outputs:
            # Save content item
            db_item, created = self.db.get_or_create_item(
                session,
                source=item.source_name,
                url=item.url,
                title=item.title,
                published_at=item.published_at,
                summary=item.snippet or item.body_text[:500],
            )

            if created:
                # Save output
                self.db.save_output(
                    session,
                    run_id=run_id,
                    item_id=db_item.id,
                    headline=result.headline,
                    archetype=spine.content_archetype,
                    image_suggestion=result.image_suggestion,
                    layout_notes=result.layout_notes,
                    highlight_words=result.highlight_words,
                    extracted_claim=spine.hook,
                    virality_score=spine.standalone_clarity_score,  # Using clarity as proxy
                    confidence=0.8,
                    why_it_will_work=[spine.real_world_consequence],
                    sources_json={
                        "source_name": item.source_name,
                        "url": item.url,
                        "title": item.title,
                        "population": spine.who_it_applies_to,
                        "time_window": spine.time_window,
                        "emotional_hook": spine.emotional_hook,
                        "support_level": spine.support_level,
                        "credibility_tier": item.credibility_tier.value,
                    },
                )

    def _complete_run(
        self,
        session,
        run_id: str,
        stats: dict,
        report: RunReport,
    ) -> dict:
        """Complete the run and log final report."""
        self.db.complete_run(
            session, run_id, "SUCCESS",
            items_fetched=stats["items_fetched"],
            items_evaluated=stats.get("items_with_narrative", 0),
            items_output=stats.get("items_after_diversity", 0),
        )

        stats["completed_at"] = datetime.now(timezone.utc).isoformat()
        stats["status"] = "SUCCESS"
        stats["report_summary"] = report.get_summary()

        # Log the report
        logger.info(
            "run_completed",
            **{k: v for k, v in stats.items() if k not in ("errors", "report_summary")},
        )

        # Print report to console for visibility
        print(report.get_summary())

        return stats

    def _dedupe_normalized(
        self,
        deduplicator: Deduplicator,
        items: list[NormalizedItem],
    ) -> list[NormalizedItem]:
        """Deduplicate normalized items."""
        seen_urls = set(deduplicator.recent_urls)
        unique = []

        for item in items:
            url = item.url.lower().strip().rstrip("/")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            unique.append(item)

        return unique


async def run_bot(
    freshness_hours: Optional[int] = None,
    max_outputs: Optional[int] = None,
    dry_run: bool = False,
) -> dict:
    """Convenience function to run the bot."""
    runner = BotRunner()
    return await runner.run(freshness_hours, max_outputs, dry_run)


def run_bot_sync(
    freshness_hours: Optional[int] = None,
    max_outputs: Optional[int] = None,
    dry_run: bool = False,
) -> dict:
    """Synchronous wrapper for run_bot."""
    return asyncio.run(run_bot(freshness_hours, max_outputs, dry_run))


async def debug_single_url(url: str) -> dict:
    """
    Debug the full story-compression pipeline for a single URL.

    Args:
        url: URL to process

    Returns:
        Debug information at each step
    """
    from .sources.base import FetchedItem

    settings = get_settings()
    debug_info = {"url": url, "steps": {}}

    # Create a mock FetchedItem
    item = FetchedItem(
        source_name="debug",
        url=url,
        title="Debug item",
        published_at=datetime.now(timezone.utc),
        summary="",
    )

    # Step 1: Fetch content
    print("Step 1: Fetching content...")
    fetcher = ContentFetcher()
    normalized = await fetcher.fetch_content(item)
    debug_info["steps"]["content_fetch"] = {
        "body_length": normalized.body_length,
        "content_type": normalized.content_type.value,
        "credibility_tier": normalized.credibility_tier.value,
        "title": normalized.title,
        "body_preview": normalized.body_text[:500] if normalized.body_text else None,
    }
    print(f"  Body length: {normalized.body_length}")
    print(f"  Content type: {normalized.content_type.value}")
    print(f"  Credibility tier: {normalized.credibility_tier.value}")

    # Step 2: Pre-AI filter
    print("\nStep 2: Pre-AI generic filter...")
    generic_filter = GenericFilter(
        min_body_chars_paper=settings.min_body_chars_paper,
        min_body_chars_news=settings.min_body_chars_news,
    )
    pre_result = generic_filter.pre_ai_filter(normalized)
    debug_info["steps"]["pre_ai_filter"] = {
        "passed": pre_result.passed,
        "reason": pre_result.reason,
        "differentiators": pre_result.differentiators_found,
    }
    print(f"  Passed: {pre_result.passed}")
    print(f"  Reason: {pre_result.reason}")

    if not pre_result.passed:
        print("\n[STOPPED] Item rejected by pre-AI filter")
        return debug_info

    # Step 3: Narrative Extraction
    print("\nStep 3: Narrative Spine Extraction...")
    extractor = NarrativeExtractor()
    spine = extractor.extract(normalized)

    if spine is None:
        debug_info["steps"]["narrative_extraction"] = {"error": "Extraction failed"}
        print("  [FAILED] Could not extract narrative")
        return debug_info

    debug_info["steps"]["narrative_extraction"] = spine.to_dict()
    print(f"  Hook: {spine.hook[:80]}...")
    print(f"  Key numbers: {spine.key_numbers}")
    print(f"  Who: {spine.who_it_applies_to}")
    print(f"  Time: {spine.time_window}")
    print(f"  Consequence: {spine.real_world_consequence[:80]}...")
    print(f"  Clarity score: {spine.standalone_clarity_score}")
    print(f"  Emotional hook: {spine.emotional_hook}")
    print(f"  Archetype: {spine.content_archetype}")

    # Step 4: Narrative Quality Gate
    print("\nStep 4: Narrative Quality Gate...")
    quality_result = check_narrative_quality(
        spine,
        min_clarity_score=settings.min_clarity_score,
    )
    debug_info["steps"]["narrative_quality"] = {
        "passed": quality_result.passed,
        "reason": quality_result.reason,
    }
    print(f"  Passed: {quality_result.passed}")
    if not quality_result.passed:
        print(f"  Reason: {quality_result.reason}")
        print("\n[STOPPED] Item rejected by narrative quality gate")
        return debug_info

    # Step 5: Story Compression
    print("\nStep 5: Story Compression...")
    compressor = StoryCompressor()
    result = compressor.compress(normalized, spine)

    if result is None:
        debug_info["steps"]["story_compression"] = {"error": "Compression failed"}
        print("  [FAILED] Could not compress story")
        return debug_info

    debug_info["steps"]["story_compression"] = result.to_dict()
    print(f"  Headline: {result.headline}")
    print(f"  Highlight words: {result.highlight_words}")
    print(f"  Image suggestion: {result.image_suggestion}")

    # Step 6: Headline Quality Gate
    print("\nStep 6: Headline Quality Gate...")
    headline_quality = check_headline_quality(result.headline)
    debug_info["steps"]["headline_quality"] = {
        "passed": headline_quality.passed,
        "reason": headline_quality.reason,
        "has_number": headline_quality.has_number,
    }
    print(f"  Passed: {headline_quality.passed}")
    if not headline_quality.passed:
        print(f"  Reason: {headline_quality.reason}")

    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)

    return debug_info
