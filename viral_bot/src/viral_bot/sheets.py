"""
Google Sheets integration for output export.

Appends generated post ideas to a Google Sheet for review and tracking.
"""

from datetime import datetime, timezone
from typing import Optional
import json

import gspread
from google.oauth2.service_account import Credentials

from .config import get_settings
from .logging_conf import get_logger

logger = get_logger(__name__)


# Required scopes for Google Sheets API
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# Column headers for the output sheet
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
        
        # Get credentials from settings
        creds_dict = settings.google_credentials_dict
        
        self.credentials = Credentials.from_service_account_info(
            creds_dict,
            scopes=SCOPES,
        )
        
        self.sheet_id = settings.google_sheet_id
        self.tab_name = settings.google_sheet_tab_name
        
        # Initialize gspread client
        self.client = gspread.authorize(self.credentials)
        
        logger.info(
            "sheets_exporter_initialized",
            sheet_id=self.sheet_id[:20] + "...",
            tab_name=self.tab_name,
        )
    
    def _get_or_create_sheet(self) -> gspread.Worksheet:
        """Get or create the worksheet."""
        spreadsheet = self.client.open_by_key(self.sheet_id)
        
        # Try to get existing worksheet
        try:
            worksheet = spreadsheet.worksheet(self.tab_name)
            logger.debug("worksheet_found", tab=self.tab_name)
        except gspread.WorksheetNotFound:
            # Create new worksheet
            worksheet = spreadsheet.add_worksheet(
                title=self.tab_name,
                rows=1000,
                cols=len(SHEET_HEADERS),
            )
            # Add headers
            worksheet.append_row(SHEET_HEADERS)
            logger.info("worksheet_created", tab=self.tab_name)
        
        return worksheet
    
    def _ensure_headers(self, worksheet: gspread.Worksheet) -> None:
        """Ensure headers are present in the worksheet."""
        # Check if first row has headers
        try:
            first_row = worksheet.row_values(1)
            if not first_row or first_row != SHEET_HEADERS:
                # Insert headers at top
                worksheet.insert_row(SHEET_HEADERS, 1)
                logger.info("headers_added")
        except Exception:
            worksheet.append_row(SHEET_HEADERS)
    
    def format_row(
        self,
        run_id: str,
        item,  # FetchedItem
        evaluation,  # EvaluationResult  
        generation,  # GenerationResult
    ) -> list:
        """
        Format a single output row.
        
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
        
        return [
            datetime.now(timezone.utc).isoformat(),  # run_timestamp_utc
            evaluation.suggested_archetype or "",     # archetype
            item.source_name,                         # source_name
            item.url,                                 # source_url
            item.published_at.isoformat() if item.published_at else "",  # published_at
            evaluation.extracted_claim or "",         # extracted_claim
            evaluation.virality_score or 0,           # virality_score
            round(evaluation.confidence, 2) if evaluation.confidence else 0,  # confidence
            generation.headline.image_headline,       # image_headline
            generation.image.image_suggestion,        # image_suggestion
            why_bullets,                              # why_it_will_work
            "NEW",                                    # status
            "",                                       # feedback_likes
            "",                                       # feedback_shares
            "",                                       # feedback_saves
            "",                                       # feedback_notes
        ]
    
    def append_rows(
        self,
        rows: list[list],
    ) -> int:
        """
        Append multiple rows to the sheet.
        
        Args:
            rows: List of row data
        
        Returns:
            Number of rows appended
        """
        if not rows:
            return 0
        
        worksheet = self._get_or_create_sheet()
        self._ensure_headers(worksheet)
        
        # Append rows
        worksheet.append_rows(rows)
        
        logger.info("rows_appended", count=len(rows))
        return len(rows)
    
    def export_outputs(
        self,
        run_id: str,
        outputs: list[tuple],  # List of (item, evaluation, generation) tuples
    ) -> int:
        """
        Export generated outputs to Google Sheets.
        
        Args:
            run_id: Run identifier
            outputs: List of (FetchedItem, EvaluationResult, GenerationResult) tuples
        
        Returns:
            Number of rows exported
        """
        logger.info("exporting_to_sheets", count=len(outputs))
        
        rows = []
        for item, evaluation, generation in outputs:
            row = self.format_row(run_id, item, evaluation, generation)
            rows.append(row)
        
        return self.append_rows(rows)
    
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
