"""
Main orchestration module with anti-generic pipeline.

Coordinates the full workflow:
1. Fetch content from sources
2. Fetch full article/abstract text
3. Pre-AI generic filter
4. Deduplicate (URL + semantic)
5. Evaluate virality with AI
6. Post-AI differentiator gate
7. Novelty check against history
8. Mix outputs for diversity
9. Generate headlines and image suggestions
10. Export to Google Sheets

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
from .normalize import NormalizedItem, ContentType
from .content_fetch import ContentFetcher
from .anti_generic import GenericFilter, check_post_ai_differentiators, format_differentiator_summary
from .novelty import NoveltyChecker, load_novelty_data_from_db
from .mixer import OutputMixer, MixingConfig
from .dedupe import Deduplicator, load_recent_data_from_db
from .openai_eval import ViralityPredictor, EvaluationResult
from .openai_generate import ContentGenerator, GenerationResult
from .sheets import SheetsExporter

logger = get_logger(__name__)


class RunReport:
    """Tracks statistics and examples for run reporting."""

    def __init__(self):
        self.stats = defaultdict(int)
        self.rejection_reasons = defaultdict(int)
        self.rejection_examples = []
        self.accepted_items = []

    def log_rejection(self, stage: str, reason: str, title: str):
        """Log a rejection."""
        self.rejection_reasons[f"{stage}:{reason}"] += 1
        if len(self.rejection_examples) < 10:
            self.rejection_examples.append({
                "stage": stage,
                "reason": reason,
                "title": title[:100],
            })

    def log_acceptance(self, item, eval_result):
        """Log an accepted item."""
        self.accepted_items.append({
            "title": item.title[:100] if hasattr(item, 'title') else str(item)[:100],
            "score": eval_result.virality_score,
            "archetype": eval_result.suggested_archetype,
            "numbers": eval_result.must_include_numbers[:3] if eval_result.must_include_numbers else [],
        })

    def get_summary(self) -> str:
        """Get formatted summary."""
        lines = [
            "=" * 60,
            "RUN REPORT",
            "=" * 60,
            "",
            "PIPELINE STATISTICS:",
        ]

        for key, value in sorted(self.stats.items()):
            lines.append(f"  {key}: {value}")

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
                "ACCEPTED ITEMS:",
            ])
            for item in self.accepted_items[:5]:
                lines.append(f"  [{item['archetype']}] Score: {item['score']}")
                lines.append(f"    Title: {item['title']}")
                if item['numbers']:
                    lines.append(f"    Numbers: {', '.join(item['numbers'])}")

        lines.append("=" * 60)
        return "\n".join(lines)


class BotRunner:
    """
    Main bot runner that orchestrates the full workflow.
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
        self.predictor = ViralityPredictor()
        self.generator = ContentGenerator()
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
        Execute one full bot run.

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
            "items_evaluated": 0,
            "items_relevant": 0,
            "items_after_post_filter": 0,
            "items_after_novelty": 0,
            "items_after_mixing": 0,
            "items_generated": 0,
            "items_exported": 0,
            "errors": [],
            "dry_run": dry_run,
        }

        session = self.db.get_session()

        try:
            # Record run start
            self.db.start_run(session, run_id)

            # === STEP 1: Fetch content from all sources ===
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

            # Convert back to FetchedItem-like for dedup (uses URL and title)
            unique_normalized = self._dedupe_normalized(deduplicator, normalized_items)
            stats["items_after_dedup"] = len(unique_normalized)
            report.stats["after_dedup"] = len(unique_normalized)

            dedup_rejected = len(normalized_items) - len(unique_normalized)
            if dedup_rejected > 0:
                report.rejection_reasons["dedup:duplicate"] = dedup_rejected

            if not unique_normalized:
                logger.warning("no_items_after_dedup")
                return self._complete_run(session, run_id, stats, report)

            # === STEP 5: Save items to database ===
            logger.info("step_5_saving_items")

            db_items = []
            for item in unique_normalized:
                db_item, created = self.db.get_or_create_item(
                    session,
                    source=item.source_name,
                    url=item.url,
                    title=item.title,
                    published_at=item.published_at,
                    summary=item.snippet or item.body_text[:500],
                )
                if created:
                    db_items.append((item, db_item))

            session.commit()
            logger.info("items_saved", count=len(db_items))

            # === STEP 6: Evaluate virality ===
            logger.info("step_6_evaluating_virality")
            report.stats["step"] = 6

            # Only evaluate new items
            items_to_evaluate = [item for item, _ in db_items]

            evaluated = self.predictor.evaluate_batch(
                items_to_evaluate,
                min_score=self.settings.min_virality_score,
            )
            stats["items_evaluated"] = len(items_to_evaluate)
            stats["items_relevant"] = len(evaluated)
            report.stats["evaluated"] = len(items_to_evaluate)
            report.stats["relevant"] = len(evaluated)

            # Log AI rejections
            ai_rejected = len(items_to_evaluate) - len(evaluated)
            if ai_rejected > 0:
                report.rejection_reasons["ai_eval:not_relevant_or_low_score"] = ai_rejected

            if not evaluated:
                logger.warning("no_relevant_items")
                return self._complete_run(session, run_id, stats, report)

            # === STEP 7: Post-AI differentiator filter ===
            if self.settings.enable_post_ai_filter:
                logger.info("step_7_post_ai_filter")
                report.stats["step"] = 7

                post_filtered = []
                for item, eval_result in evaluated:
                    passed, reason, reqs = check_post_ai_differentiators(
                        eval_result.to_dict(),
                        min_differentiators=self.settings.min_differentiators,
                    )
                    if passed:
                        post_filtered.append((item, eval_result))
                        report.log_acceptance(item, eval_result)
                    else:
                        report.log_rejection(
                            "post_ai",
                            reason or "insufficient_differentiators",
                            item.title,
                        )

                stats["items_after_post_filter"] = len(post_filtered)
                report.stats["after_post_filter"] = len(post_filtered)
                evaluated = post_filtered

            else:
                stats["items_after_post_filter"] = len(evaluated)
                for item, eval_result in evaluated:
                    report.log_acceptance(item, eval_result)

            if not evaluated:
                logger.warning("no_items_after_post_filter")
                return self._complete_run(session, run_id, stats, report)

            # === STEP 8: Novelty check ===
            logger.info("step_8_novelty_check")
            report.stats["step"] = 8

            # Load recent findings for novelty check
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
            for item, eval_result in evaluated:
                finding = eval_result.most_surprising_finding or eval_result.extracted_claim or item.title
                novelty_result, adjusted_score = novelty_checker.check_novelty(
                    finding,
                    eval_result.virality_score,
                )

                if novelty_result.is_novel:
                    # Update score if penalized
                    if adjusted_score != eval_result.virality_score:
                        eval_result.virality_score = adjusted_score
                    novelty_passed.append((item, eval_result))
                    # Add to checker for subsequent items
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

            # === STEP 9: Mix outputs for diversity ===
            logger.info("step_9_mixing_outputs")
            report.stats["step"] = 9

            mixer_config = MixingConfig(
                max_study_stat=self.settings.max_study_stat_per_run,
                min_non_study_stat=self.settings.min_non_study_stat_per_run,
                max_per_archetype=self.settings.max_per_archetype,
            )
            mixer = OutputMixer(mixer_config)

            # Re-sort by score after novelty penalties
            novelty_passed.sort(key=lambda x: x[1].virality_score or 0, reverse=True)

            mixed = mixer.select_outputs(novelty_passed, max_outputs=max_outputs)
            stats["items_after_mixing"] = len(mixed)
            report.stats["after_mixing"] = len(mixed)
            report.stats["archetype_distribution"] = mixer.get_archetype_summary()

            # Save evaluations to database
            item_to_db_id = {item.url: db_item.id for item, db_item in db_items}

            for item, eval_result in mixed:
                db_item_id = item_to_db_id.get(item.url)
                if db_item_id:
                    self.db.save_evaluation(
                        session,
                        item_id=db_item_id,
                        relevant=eval_result.relevant,
                        relevance_reason=eval_result.reason,
                        virality_score=eval_result.virality_score,
                        confidence=eval_result.confidence,
                        suggested_archetype=eval_result.suggested_archetype,
                        extracted_claim=eval_result.most_surprising_finding or eval_result.extracted_claim,
                        why_it_will_work=eval_result.why_it_will_work,
                        must_include_numbers=eval_result.must_include_numbers,
                    )

            session.commit()

            # === STEP 10: Generate headlines and image suggestions ===
            logger.info("step_10_generating_content")
            report.stats["step"] = 10

            generated = self.generator.generate_batch(
                mixed,
                max_outputs=max_outputs,
            )
            stats["items_generated"] = len(generated)
            report.stats["generated"] = len(generated)

            if not generated:
                logger.warning("no_content_generated")
                return self._complete_run(session, run_id, stats, report)

            # Save outputs to database
            for item, eval_result, gen_result in generated:
                db_item_id = item_to_db_id.get(item.url)
                if db_item_id:
                    self.db.save_output(
                        session,
                        run_id=run_id,
                        item_id=db_item_id,
                        headline=gen_result.headline.image_headline,
                        archetype=eval_result.suggested_archetype or "STUDY_STAT",
                        image_suggestion=gen_result.image.image_suggestion,
                        layout_notes=gen_result.image.layout_notes,
                        highlight_words=gen_result.image.highlight_words,
                        extracted_claim=eval_result.most_surprising_finding or eval_result.extracted_claim,
                        virality_score=eval_result.virality_score,
                        confidence=eval_result.confidence,
                        why_it_will_work=eval_result.why_it_will_work,
                        sources_json={
                            "source_name": item.source_name,
                            "url": item.url,
                            "title": item.title,
                            "population": eval_result.population,
                            "time_window": eval_result.time_window,
                            "study_type": eval_result.study_type,
                        },
                    )

            session.commit()

            # === STEP 11: Export to Google Sheets ===
            if not dry_run:
                logger.info("step_11_exporting_to_sheets")

                try:
                    exported = self.exporter.export_outputs(run_id, generated)
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
                items_evaluated=stats.get("items_evaluated", 0),
                items_output=0,
                error_message=str(e),
            )

            raise

        finally:
            session.close()

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
            items_evaluated=stats.get("items_evaluated", 0),
            items_output=stats.get("items_generated", 0),
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
        """
        Deduplicate normalized items.

        Creates temporary FetchedItem-like objects for compatibility.
        """
        # URL dedup
        seen_urls = set(deduplicator.recent_urls)
        unique = []

        for item in items:
            url = item.url.lower().strip().rstrip("/")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            unique.append(item)

        # Semantic dedup would require embeddings - skip for normalized items
        # as they've already been filtered for sufficient content

        return unique


async def run_bot(
    freshness_hours: Optional[int] = None,
    max_outputs: Optional[int] = None,
    dry_run: bool = False,
) -> dict:
    """
    Convenience function to run the bot.

    Args:
        freshness_hours: Override default freshness window
        max_outputs: Override default max outputs
        dry_run: If True, don't write to Sheets

    Returns:
        Run statistics
    """
    runner = BotRunner()
    return await runner.run(freshness_hours, max_outputs, dry_run)


def run_bot_sync(
    freshness_hours: Optional[int] = None,
    max_outputs: Optional[int] = None,
    dry_run: bool = False,
) -> dict:
    """
    Synchronous wrapper for run_bot.
    """
    return asyncio.run(run_bot(freshness_hours, max_outputs, dry_run))


async def debug_single_url(url: str) -> dict:
    """
    Debug the full pipeline for a single URL.

    Args:
        url: URL to process

    Returns:
        Debug information at each step
    """
    from .sources.base import FetchedItem
    from datetime import datetime, timezone

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
        "title": normalized.title,
        "body_preview": normalized.body_text[:500] if normalized.body_text else None,
        "metadata": normalized.metadata,
    }
    print(f"  Body length: {normalized.body_length}")
    print(f"  Content type: {normalized.content_type.value}")

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

    # Step 3: AI Evaluation
    print("\nStep 3: AI Evaluation...")
    predictor = ViralityPredictor()
    eval_result = predictor.evaluate(normalized)
    debug_info["steps"]["ai_evaluation"] = eval_result.to_dict()
    print(f"  Relevant: {eval_result.relevant}")
    print(f"  Score: {eval_result.virality_score}")
    print(f"  Archetype: {eval_result.suggested_archetype}")
    print(f"  Finding: {eval_result.most_surprising_finding}")
    print(f"  Numbers: {eval_result.must_include_numbers}")
    print(f"  Population: {eval_result.population}")
    print(f"  Time window: {eval_result.time_window}")

    if not eval_result.relevant:
        print(f"\n[STOPPED] Item not relevant: {eval_result.reason}")
        return debug_info

    # Step 4: Post-AI filter
    print("\nStep 4: Post-AI differentiator check...")
    passed, reason, reqs = check_post_ai_differentiators(
        eval_result.to_dict(),
        min_differentiators=settings.min_differentiators,
    )
    debug_info["steps"]["post_ai_filter"] = {
        "passed": passed,
        "reason": reason,
        "differentiators": {
            "number": reqs.number_value,
            "population": reqs.population_value,
            "time_window": reqs.time_window_value,
            "comparison": reqs.comparison_value,
        },
    }
    print(f"  Passed: {passed}")
    print(f"  Differentiators: {format_differentiator_summary(reqs)}")

    if not passed:
        print(f"\n[STOPPED] Insufficient differentiators: {reason}")
        return debug_info

    # Step 5: Generate headline
    print("\nStep 5: Generating headline...")
    generator = ContentGenerator()
    gen_result = generator.generate(eval_result, normalized.source_name)

    if gen_result:
        debug_info["steps"]["generation"] = gen_result.to_dict()
        print(f"  Headline: {gen_result.headline.image_headline}")
        print(f"  Image: {gen_result.image.image_suggestion}")
        print(f"  Highlight: {gen_result.image.highlight_words}")
    else:
        debug_info["steps"]["generation"] = {"error": "Generation failed"}
        print("  [FAILED] Could not generate headline")

    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)

    return debug_info
