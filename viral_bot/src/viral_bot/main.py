"""
Main orchestration module.

Coordinates the full workflow:
1. Fetch content from sources
2. Deduplicate
3. Evaluate virality
4. Generate headlines and image suggestions
5. Export to Google Sheets
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional
import uuid

from .config import get_settings
from .logging_conf import get_logger, bind_context, setup_logging
from .db import get_database, ContentItem, Evaluation, Output
from .sources import get_source_registry, FetchedItem
from .dedupe import Deduplicator, load_recent_data_from_db
from .openai_eval import ViralityPredictor, EvaluationResult
from .openai_generate import ContentGenerator, GenerationResult
from .sheets import SheetsExporter

logger = get_logger(__name__)


class BotRunner:
    """
    Main bot runner that orchestrates the full workflow.
    """
    
    def __init__(self):
        """Initialize all components."""
        self.settings = get_settings()
        self.db = get_database()
        self.registry = get_source_registry()
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
    ) -> dict:
        """
        Execute one full bot run.
        
        Args:
            freshness_hours: Override default freshness window
            max_outputs: Override default max outputs
        
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
        )
        
        stats = {
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "items_fetched": 0,
            "items_after_dedup": 0,
            "items_evaluated": 0,
            "items_relevant": 0,
            "items_generated": 0,
            "items_exported": 0,
            "errors": [],
        }
        
        session = self.db.get_session()
        
        try:
            # Record run start
            self.db.start_run(session, run_id)
            
            # === STEP 1: Fetch content from all sources ===
            logger.info("step_1_fetching_content")
            
            items = await self.registry.fetch_all(
                freshness_hours=freshness_hours,
                max_concurrent=self.settings.max_concurrent_fetches,
            )
            stats["items_fetched"] = len(items)
            
            if not items:
                logger.warning("no_items_fetched")
                self.db.complete_run(
                    session, run_id, "SUCCESS",
                    items_fetched=0, items_output=0,
                )
                return stats
            
            # === STEP 2: Deduplicate ===
            logger.info("step_2_deduplicating")
            
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
            
            unique_items = deduplicator.deduplicate(items, semantic=True)
            stats["items_after_dedup"] = len(unique_items)
            
            if not unique_items:
                logger.warning("no_items_after_dedup")
                self.db.complete_run(
                    session, run_id, "SUCCESS",
                    items_fetched=stats["items_fetched"],
                    items_evaluated=0,
                    items_output=0,
                )
                return stats
            
            # === STEP 3: Save items to database ===
            logger.info("step_3_saving_items")
            
            db_items = []
            for item in unique_items:
                db_item, created = self.db.get_or_create_item(
                    session,
                    source=item.source_name,
                    url=item.url,
                    title=item.title,
                    published_at=item.published_at,
                    summary=item.summary,
                )
                if created:
                    db_items.append((item, db_item))
            
            session.commit()
            logger.info("items_saved", count=len(db_items))
            
            # === STEP 4: Evaluate virality ===
            logger.info("step_4_evaluating_virality")
            
            # Only evaluate new items
            items_to_evaluate = [item for item, _ in db_items]
            
            evaluated = self.predictor.evaluate_batch(
                items_to_evaluate,
                min_score=self.settings.min_virality_score,
            )
            stats["items_evaluated"] = len(items_to_evaluate)
            stats["items_relevant"] = len(evaluated)
            
            if not evaluated:
                logger.warning("no_relevant_items")
                self.db.complete_run(
                    session, run_id, "SUCCESS",
                    items_fetched=stats["items_fetched"],
                    items_evaluated=stats["items_evaluated"],
                    items_output=0,
                )
                return stats
            
            # Save evaluations to database
            item_to_db_id = {item.url: db_item.id for item, db_item in db_items}
            
            for item, eval_result in evaluated:
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
                        extracted_claim=eval_result.extracted_claim,
                        why_it_will_work=eval_result.why_it_will_work,
                        must_include_numbers=eval_result.must_include_numbers,
                    )
            
            session.commit()
            
            # === STEP 5: Generate headlines and image suggestions ===
            logger.info("step_5_generating_content")
            
            generated = self.generator.generate_batch(
                evaluated,
                max_outputs=max_outputs,
            )
            stats["items_generated"] = len(generated)
            
            if not generated:
                logger.warning("no_content_generated")
                self.db.complete_run(
                    session, run_id, "SUCCESS",
                    items_fetched=stats["items_fetched"],
                    items_evaluated=stats["items_evaluated"],
                    items_output=0,
                )
                return stats
            
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
                        extracted_claim=eval_result.extracted_claim,
                        virality_score=eval_result.virality_score,
                        confidence=eval_result.confidence,
                        why_it_will_work=eval_result.why_it_will_work,
                        sources_json={
                            "source_name": item.source_name,
                            "url": item.url,
                            "title": item.title,
                        },
                    )
            
            session.commit()
            
            # === STEP 6: Export to Google Sheets ===
            logger.info("step_6_exporting_to_sheets")
            
            try:
                exported = self.exporter.export_outputs(run_id, generated)
                stats["items_exported"] = exported
            except Exception as e:
                logger.error("sheets_export_failed", error=str(e))
                stats["errors"].append(f"Sheets export failed: {str(e)}")
            
            # === Complete run ===
            self.db.complete_run(
                session, run_id, "SUCCESS",
                items_fetched=stats["items_fetched"],
                items_evaluated=stats["items_evaluated"],
                items_output=stats["items_generated"],
            )
            
            stats["completed_at"] = datetime.now(timezone.utc).isoformat()
            stats["status"] = "SUCCESS"
            
            logger.info(
                "run_completed",
                **{k: v for k, v in stats.items() if k != "errors"},
            )
            
            return stats
            
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


async def run_bot(
    freshness_hours: Optional[int] = None,
    max_outputs: Optional[int] = None,
) -> dict:
    """
    Convenience function to run the bot.
    
    Args:
        freshness_hours: Override default freshness window
        max_outputs: Override default max outputs
    
    Returns:
        Run statistics
    """
    runner = BotRunner()
    return await runner.run(freshness_hours, max_outputs)


def run_bot_sync(
    freshness_hours: Optional[int] = None,
    max_outputs: Optional[int] = None,
) -> dict:
    """
    Synchronous wrapper for run_bot.
    """
    return asyncio.run(run_bot(freshness_hours, max_outputs))
