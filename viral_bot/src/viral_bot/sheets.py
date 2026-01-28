"""
Google Sheets integration for story-compressed output export.

Appends generated post ideas to a Google Sheet with extended columns for
narrative spine data, credibility tiers, and story compression metrics.
"""

from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING
import json

import gspread
from google.oauth2.service_account import Credentials

from .config import get_settings
from .logging_conf import get_logger

if TYPE_CHECKING:
    from .normalize import NormalizedItem
    from .narrative_extractor import NarrativeSpine
    from .story_generator import StoryCompressionResult

logger = get_logger(__name__)


# Required scopes for Google Sheets API
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# Column headers for the output sheet (V3 - Viral Likeness Pipeline)
SHEET_HEADERS_V3 = [
    # Core columns
    "run_timestamp_utc",
    "source_name",
    "source_url",
    "published_at",
    "credibility_tier",
    "content_type",

    # Slide output (main content)
    "image_headline",            # ALL CAPS Instagram slide text
    "caption",                   # Supporting caption with source

    # Viral scoring
    "primitive",                 # Viral primitive type
    "viral_likeness_score",      # 0-100 similarity to winners
    "primitive_score",           # 0-100 primitive match strength
    "final_score",               # Weighted combination

    # Narrative elements
    "hook",
    "action",
    "outcome",
    "numbers",                   # Comma-separated
    "time_window",
    "who_it_applies_to",

    # Quality metrics
    "standalone_clarity",        # 0-10 score
    "tone",                      # shock/awe/concern/warmth/humor
    "emotional_hook",
    "support_level",

    # Status and feedback
    "status",
    "feedback_likes",
    "feedback_shares",
    "feedback_saves",
    "feedback_notes",
]

# Column headers for the output sheet (V2 - Story Compression) - Legacy
SHEET_HEADERS_V2 = [
    # Core columns
    "run_timestamp_utc",
    "source_name",
    "source_url",
    "published_at",
    "credibility_tier",          # NEW: A/B/C tier
    "content_type",              # NEW: paper/news/press_release/etc

    # Narrative spine columns
    "hook",                      # NEW: The attention-grabbing element
    "who_it_applies_to",         # NEW: Population
    "time_window",               # NEW: Duration/time frame
    "key_numbers",               # NEW: List of specific numbers
    "mechanism_or_why",          # NEW: Causation explanation
    "real_world_consequence",    # NEW: Plain language impact

    # Story compression output
    "image_headline",            # The final compressed headline
    "archetype",                 # Content archetype

    # Quality metrics
    "standalone_clarity_score",  # NEW: 1-10 score
    "emotional_hook",            # NEW: fear/hope/surprise/etc
    "support_level",             # NEW: strong/moderate/emerging/preliminary

    # Image/design
    "image_suggestion",
    "layout_notes",              # NEW: List of layout notes
    "highlight_words",           # NEW: Words to highlight

    # Status and feedback
    "status",
    "feedback_likes",
    "feedback_shares",
    "feedback_saves",
    "feedback_notes",
]

# Legacy headers for backward compatibility
SHEET_HEADERS = [
    "run_timestamp_utc",
    "archetype",
    "source_name",
    "source_url",
    "published_at",
    "extracted_claim",
    "virality_score",
    "confidence",
    "image_headline",
    "image_suggestion",
    "why_it_will_work",
    "status",
    "population",
    "time_window",
    "study_type",
    "must_include_numbers",
    "feedback_likes",
    "feedback_shares",
    "feedback_saves",
    "feedback_notes",
]


class SheetsExporter:
    """
    Exports post ideas to Google Sheets.
    """

    def __init__(self):
        """Initialize with Google credentials."""
        settings = get_settings()

        # Get credentials from settings with error handling
        try:
            creds_dict = settings.google_credentials_dict
        except json.JSONDecodeError as e:
            logger.error(
                "google_credentials_invalid_json",
                error=str(e),
                hint="Check GOOGLE_SERVICE_ACCOUNT_JSON is valid JSON",
            )
            raise ValueError(
                "GOOGLE_SERVICE_ACCOUNT_JSON contains invalid JSON. "
                "Make sure you copied the entire service account JSON file contents."
            ) from e
        except Exception as e:
            logger.error("google_credentials_error", error=str(e))
            raise ValueError(
                f"Failed to parse Google credentials: {e}. "
                "Check GOOGLE_SERVICE_ACCOUNT_JSON and GOOGLE_SERVICE_ACCOUNT_BASE64 settings."
            ) from e

        try:
            self.credentials = Credentials.from_service_account_info(
                creds_dict,
                scopes=SCOPES,
            )
        except Exception as e:
            logger.error(
                "google_credentials_invalid",
                error=str(e),
                hint="Service account JSON may be malformed or missing required fields",
            )
            raise ValueError(
                f"Invalid Google service account credentials: {e}. "
                "Ensure you're using a valid service account JSON from Google Cloud Console."
            ) from e

        self.sheet_id = settings.google_sheet_id
        self.tab_name = settings.google_sheet_tab_name

        # Initialize gspread client
        try:
            self.client = gspread.authorize(self.credentials)
        except Exception as e:
            logger.error("gspread_auth_failed", error=str(e))
            raise ValueError(
                f"Failed to authorize with Google Sheets API: {e}"
            ) from e

        logger.info(
            "sheets_exporter_initialized",
            sheet_id=self.sheet_id[:20] + "...",
            tab_name=self.tab_name,
        )

    def _get_or_create_sheet(self, use_v2: bool = False) -> gspread.Worksheet:
        """Get or create the worksheet."""
        spreadsheet = self.client.open_by_key(self.sheet_id)
        headers = SHEET_HEADERS_V2 if use_v2 else SHEET_HEADERS

        # Try to get existing worksheet
        try:
            worksheet = spreadsheet.worksheet(self.tab_name)
            logger.debug("worksheet_found", tab=self.tab_name)
        except gspread.WorksheetNotFound:
            # Create new worksheet
            worksheet = spreadsheet.add_worksheet(
                title=self.tab_name,
                rows=1000,
                cols=len(headers),
            )
            # Add headers
            worksheet.append_row(headers)
            logger.info("worksheet_created", tab=self.tab_name)

        return worksheet

    def _ensure_headers(self, worksheet: gspread.Worksheet, use_v2: bool = False) -> None:
        """Ensure headers are present and updated in the worksheet."""
        headers = SHEET_HEADERS_V2 if use_v2 else SHEET_HEADERS

        try:
            first_row = worksheet.row_values(1)
            if not first_row:
                worksheet.insert_row(headers, 1)
                logger.info("headers_added")
            elif first_row != headers:
                if len(first_row) < len(headers):
                    for i, header in enumerate(headers):
                        if i >= len(first_row):
                            worksheet.update_cell(1, i + 1, header)
                    logger.info("headers_updated", new_columns=len(headers) - len(first_row))
        except Exception:
            worksheet.append_row(headers)

    def format_row_v3(
        self,
        run_id: str,
        item: "NormalizedItem",
        spine: "NarrativeSpine",
        image_headline: str,
        caption: str,
        viral_likeness_score: int,
        primitive_score: int,
        final_score: float,
    ) -> list:
        """
        Format a row for the V3 viral-likeness format.

        Args:
            run_id: Run identifier
            item: NormalizedItem
            spine: NarrativeSpine from extraction
            image_headline: Generated slide copy
            caption: Generated caption
            viral_likeness_score: Similarity to winners (0-100)
            primitive_score: Primitive match strength (0-100)
            final_score: Weighted final score

        Returns:
            List of cell values matching SHEET_HEADERS_V3
        """
        return [
            # Core columns
            datetime.now(timezone.utc).isoformat(),       # run_timestamp_utc
            item.source_name,                              # source_name
            item.url,                                      # source_url
            item.published_at.isoformat() if item.published_at else "",  # published_at
            item.credibility_tier.value,                   # credibility_tier
            item.content_type.value,                       # content_type

            # Slide output
            image_headline,                                # image_headline
            caption,                                       # caption

            # Viral scoring
            spine.primitive,                               # primitive
            viral_likeness_score,                          # viral_likeness_score
            primitive_score,                               # primitive_score
            round(final_score, 2),                         # final_score

            # Narrative elements
            spine.hook,                                    # hook
            spine.action or "",                            # action
            spine.outcome or "",                           # outcome
            ", ".join(spine.numbers),                      # numbers
            spine.time_window or "",                       # time_window
            spine.who_it_applies_to or "",                 # who_it_applies_to

            # Quality metrics
            spine.standalone_clarity,                      # standalone_clarity
            spine.tone,                                    # tone
            spine.emotional_hook,                          # emotional_hook
            spine.support_level,                           # support_level

            # Status and feedback
            "NEW",                                         # status
            "",                                            # feedback_likes
            "",                                            # feedback_shares
            "",                                            # feedback_saves
            "",                                            # feedback_notes
        ]

    def export_outputs_v3(
        self,
        run_id: str,
        outputs: list[dict],  # List of output dictionaries with all required fields
    ) -> int:
        """
        Export viral-likeness outputs to Google Sheets (V3 format).

        Args:
            run_id: Run identifier
            outputs: List of dicts with keys: item, spine, image_headline, caption,
                     viral_likeness_score, primitive_score, final_score

        Returns:
            Number of rows exported
        """
        logger.info("exporting_to_sheets_v3", count=len(outputs))

        rows = []
        for output in outputs:
            row = self.format_row_v3(
                run_id=run_id,
                item=output["item"],
                spine=output["spine"],
                image_headline=output["image_headline"],
                caption=output["caption"],
                viral_likeness_score=output["viral_likeness_score"],
                primitive_score=output["primitive_score"],
                final_score=output["final_score"],
            )
            rows.append(row)

        return self.append_rows_v3(rows)

    def append_rows_v3(self, rows: list[list]) -> int:
        """Append rows using V3 headers."""
        if not rows:
            return 0

        worksheet = self._get_or_create_sheet_v3()
        self._ensure_headers_v3(worksheet)

        worksheet.append_rows(rows)
        logger.info("rows_appended_v3", count=len(rows))
        return len(rows)

    def _get_or_create_sheet_v3(self) -> gspread.Worksheet:
        """Get or create worksheet with V3 headers."""
        spreadsheet = self.client.open_by_key(self.sheet_id)

        try:
            worksheet = spreadsheet.worksheet(self.tab_name)
        except gspread.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(
                title=self.tab_name,
                rows=1000,
                cols=len(SHEET_HEADERS_V3),
            )
            worksheet.append_row(SHEET_HEADERS_V3)
            logger.info("worksheet_created_v3", tab=self.tab_name)

        return worksheet

    def _ensure_headers_v3(self, worksheet: gspread.Worksheet) -> None:
        """Ensure V3 headers are present."""
        try:
            first_row = worksheet.row_values(1)
            if not first_row:
                worksheet.insert_row(SHEET_HEADERS_V3, 1)
            elif first_row != SHEET_HEADERS_V3:
                if len(first_row) < len(SHEET_HEADERS_V3):
                    for i, header in enumerate(SHEET_HEADERS_V3):
                        if i >= len(first_row):
                            worksheet.update_cell(1, i + 1, header)
        except Exception:
            worksheet.append_row(SHEET_HEADERS_V3)

    def format_row_v2(
        self,
        run_id: str,
        item: "NormalizedItem",
        spine: "NarrativeSpine",
        result: "StoryCompressionResult",
    ) -> list:
        """
        Format a row for the V2 story-compression format.

        Args:
            run_id: Run identifier
            item: NormalizedItem
            spine: NarrativeSpine from extraction
            result: StoryCompressionResult from compression

        Returns:
            List of cell values matching SHEET_HEADERS_V2
        """
        return [
            # Core columns
            datetime.now(timezone.utc).isoformat(),       # run_timestamp_utc
            item.source_name,                              # source_name
            item.url,                                      # source_url
            item.published_at.isoformat() if item.published_at else "",  # published_at
            item.credibility_tier.value,                   # credibility_tier
            item.content_type.value,                       # content_type

            # Narrative spine columns
            spine.hook,                                    # hook
            spine.who_it_applies_to,                       # who_it_applies_to
            spine.time_window,                             # time_window
            ", ".join(spine.key_numbers),                  # key_numbers
            spine.mechanism_or_why,                        # mechanism_or_why
            spine.real_world_consequence,                  # real_world_consequence

            # Story compression output
            result.headline,                               # image_headline
            spine.content_archetype,                       # archetype

            # Quality metrics
            spine.standalone_clarity_score,                # standalone_clarity_score
            spine.emotional_hook,                          # emotional_hook
            spine.support_level,                           # support_level

            # Image/design
            result.image_suggestion,                       # image_suggestion
            " | ".join(result.layout_notes),               # layout_notes
            ", ".join(result.highlight_words),             # highlight_words

            # Status and feedback
            "NEW",                                         # status
            "",                                            # feedback_likes
            "",                                            # feedback_shares
            "",                                            # feedback_saves
            "",                                            # feedback_notes
        ]

    def format_row(
        self,
        run_id: str,
        item,  # NormalizedItem or FetchedItem
        evaluation,  # EvaluationResult
        generation,  # GenerationResult
    ) -> list:
        """
        Format a single output row (legacy format).

        Args:
            run_id: Run identifier
            item: Source content item
            evaluation: Evaluation result
            generation: Generation result

        Returns:
            List of cell values matching SHEET_HEADERS
        """
        # Format why_it_will_work as bullet points
        why_bullets = " • ".join(evaluation.why_it_will_work) if evaluation.why_it_will_work else ""

        # Format must_include_numbers
        numbers_str = ", ".join(evaluation.must_include_numbers) if evaluation.must_include_numbers else ""

        return [
            datetime.now(timezone.utc).isoformat(),  # run_timestamp_utc
            evaluation.suggested_archetype or "",     # archetype
            item.source_name,                         # source_name
            item.url,                                 # source_url
            item.published_at.isoformat() if item.published_at else "",  # published_at
            evaluation.most_surprising_finding or evaluation.extracted_claim or "",  # extracted_claim
            evaluation.virality_score or 0,           # virality_score
            round(evaluation.confidence, 2) if evaluation.confidence else 0,  # confidence
            generation.headline.image_headline or "",  # image_headline
            generation.image.image_suggestion,        # image_suggestion
            why_bullets,                              # why_it_will_work
            "NEW",                                    # status
            # NEW COLUMNS
            evaluation.population or "",              # population
            evaluation.time_window or "",             # time_window
            evaluation.study_type or "",              # study_type
            numbers_str,                              # must_include_numbers
            # Feedback columns (empty by default)
            "",                                       # feedback_likes
            "",                                       # feedback_shares
            "",                                       # feedback_saves
            "",                                       # feedback_notes
        ]

    def append_rows(
        self,
        rows: list[list],
        use_v2: bool = False,
    ) -> int:
        """
        Append multiple rows to the sheet.

        Args:
            rows: List of row data
            use_v2: Use V2 headers

        Returns:
            Number of rows appended
        """
        if not rows:
            return 0

        worksheet = self._get_or_create_sheet(use_v2)
        self._ensure_headers(worksheet, use_v2)

        # Append rows
        worksheet.append_rows(rows)

        logger.info("rows_appended", count=len(rows))
        return len(rows)

    def export_outputs_v2(
        self,
        run_id: str,
        outputs: list[tuple],  # List of (NormalizedItem, NarrativeSpine, StoryCompressionResult) tuples
    ) -> int:
        """
        Export story-compressed outputs to Google Sheets (V2 format).

        Args:
            run_id: Run identifier
            outputs: List of (NormalizedItem, NarrativeSpine, StoryCompressionResult) tuples

        Returns:
            Number of rows exported
        """
        logger.info("exporting_to_sheets_v2", count=len(outputs))

        rows = []
        for item, spine, result in outputs:
            row = self.format_row_v2(run_id, item, spine, result)
            rows.append(row)

        return self.append_rows(rows, use_v2=True)

    def export_outputs(
        self,
        run_id: str,
        outputs: list[tuple],  # List of (item, evaluation, generation) tuples
    ) -> int:
        """
        Export generated outputs to Google Sheets (legacy format).

        Args:
            run_id: Run identifier
            outputs: List of (NormalizedItem, EvaluationResult, GenerationResult) tuples

        Returns:
            Number of rows exported
        """
        logger.info("exporting_to_sheets", count=len(outputs))

        rows = []
        for item, evaluation, generation in outputs:
            row = self.format_row(run_id, item, evaluation, generation)
            rows.append(row)

        return self.append_rows(rows, use_v2=False)

    def get_recent_outputs(self, limit: int = 50) -> list[dict]:
        """
        Get recent outputs from the sheet.

        Args:
            limit: Maximum number of rows to return

        Returns:
            List of output dictionaries
        """
        worksheet = self._get_or_create_sheet()

        # Get all values
        all_values = worksheet.get_all_values()

        if len(all_values) <= 1:  # Only headers or empty
            return []

        headers = all_values[0]
        rows = all_values[1:][-limit:]  # Get last 'limit' rows

        outputs = []
        for row in reversed(rows):  # Most recent first
            # Pad row to match headers length
            while len(row) < len(headers):
                row.append("")
            output = dict(zip(headers, row))
            outputs.append(output)

        return outputs

    def update_feedback(
        self,
        row_index: int,
        likes: Optional[int] = None,
        shares: Optional[int] = None,
        saves: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """
        Update feedback columns for a specific row.

        Args:
            row_index: Row number (1-indexed, excluding header)
            likes, shares, saves: Performance metrics
            notes: Additional notes

        Returns:
            True if successful
        """
        worksheet = self._get_or_create_sheet()

        # Row index is 1-indexed and we need to account for header
        actual_row = row_index + 1

        # Find column indices for feedback fields
        headers = worksheet.row_values(1)

        try:
            updates = []

            if likes is not None:
                col = headers.index("feedback_likes") + 1
                updates.append((actual_row, col, likes))

            if shares is not None:
                col = headers.index("feedback_shares") + 1
                updates.append((actual_row, col, shares))

            if saves is not None:
                col = headers.index("feedback_saves") + 1
                updates.append((actual_row, col, saves))

            if notes is not None:
                col = headers.index("feedback_notes") + 1
                updates.append((actual_row, col, notes))

            # Update status to indicate feedback received
            col = headers.index("status") + 1
            updates.append((actual_row, col, "FEEDBACK"))

            # Batch update
            for row, col, value in updates:
                worksheet.update_cell(row, col, value)

            logger.info("feedback_updated", row=row_index)
            return True

        except (ValueError, gspread.exceptions.APIError) as e:
            logger.error("feedback_update_failed", row=row_index, error=str(e))
            return False


def test_connection() -> bool:
    """Test Google Sheets connection."""
    try:
        exporter = SheetsExporter()
        worksheet = exporter._get_or_create_sheet()
        logger.info("sheets_connection_test_passed")
        return True
    except Exception as e:
        logger.error("sheets_connection_test_failed", error=str(e))
        return False
