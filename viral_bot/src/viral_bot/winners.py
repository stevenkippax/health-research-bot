"""
Winner corpus management for viral likeness scoring.

Loads, caches, and normalizes top-performing headlines from Google Sheets
to establish a "ground truth" style target for generated content.
"""

import csv
import re
import hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import gspread
from google.oauth2.service_account import Credentials

from .config import get_settings
from .logging_conf import get_logger
from .db import get_session, WinnerHeadline

logger = get_logger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]


@dataclass
class WinnerCorpus:
    """Container for winner headlines with metadata."""
    headlines: list[str] = field(default_factory=list)
    normalized: list[str] = field(default_factory=list)
    loaded_at: Optional[datetime] = None
    source: str = "unknown"  # "sheets" or "csv" or "cache"

    @property
    def count(self) -> int:
        return len(self.headlines)

    @property
    def is_empty(self) -> bool:
        return self.count == 0

    @property
    def is_stale(self) -> bool:
        """Check if corpus needs refresh."""
        if self.loaded_at is None:
            return True
        settings = get_settings()
        age = datetime.now(timezone.utc) - self.loaded_at
        return age > timedelta(hours=settings.winners_refresh_hours)


def normalize_headline(text: str) -> str:
    """
    Normalize headline for template mining and comparison.

    - Convert to uppercase
    - Strip extra whitespace
    - Remove punctuation except periods and ellipses
    - Standardize ellipses
    """
    if not text:
        return ""

    # Uppercase
    text = text.upper()

    # Standardize ellipses (... or …)
    text = re.sub(r'\.{2,}|…', '...', text)

    # Remove most punctuation but keep periods and ellipses
    text = re.sub(r'[^\w\s\.\-]', '', text)

    # Normalize whitespace
    text = ' '.join(text.split())

    return text.strip()


def extract_template_patterns(headlines: list[str]) -> dict[str, int]:
    """
    Extract common template patterns from winner headlines.

    Returns dict of pattern -> frequency.
    """
    patterns = {
        "STUDY_SAYS": r'^STUDY\s+SAYS',
        "STUDY_REVEALS": r'^STUDY\s+REVEALS',
        "STUDY_FINDS": r'^STUDY\s+FINDS',
        "ONE_X_CAN": r'^ONE\s+\w+\s+CAN',
        "THIS_IS_THE": r'^THIS\s+IS\s+THE',
        "IF_YOU": r'^IF\s+YOU',
        "X_LINKED_TO": r'\bLINKED\s+TO\b',
        "X_PERCENT": r'\d+%',
        "X_TIMES": r'\d+X?\s+TIMES',
        "X_YEARS": r'\d+\s+YEARS?',
        "X_MINUTES": r'\d+\s+MINUTES?',
        "X_HOURS": r'\d+\s+HOURS?',
        "EQUIVALENT_TO": r'EQUIVALENT\s+TO',
        "CAN_REDUCE": r'CAN\s+REDUCE',
        "MAY_HELP": r'MAY\s+HELP',
        "HELPS_WITH": r'HELPS?\s+WITH',
    }

    counts = {name: 0 for name in patterns}

    for headline in headlines:
        normalized = headline.upper()
        for name, pattern in patterns.items():
            if re.search(pattern, normalized):
                counts[name] += 1

    # Sort by frequency
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


class WinnerLoader:
    """Loads and caches winner headlines from Google Sheets or local CSV."""

    def __init__(self):
        self.settings = get_settings()
        self._corpus: Optional[WinnerCorpus] = None
        self._client: Optional[gspread.Client] = None

    def _get_sheets_client(self) -> gspread.Client:
        """Get authenticated gspread client."""
        if self._client is None:
            creds = Credentials.from_service_account_info(
                self.settings.google_credentials_dict,
                scopes=SCOPES,
            )
            self._client = gspread.authorize(creds)
        return self._client

    def load_from_sheets(self) -> WinnerCorpus:
        """Load winner headlines from Google Sheets."""
        try:
            client = self._get_sheets_client()
            sheet = client.open_by_key(self.settings.winners_sheet_id)
            worksheet = sheet.sheet1  # First sheet

            # Get all values from first column
            values = worksheet.col_values(1)

            # Filter out empty and header rows
            headlines = []
            for i, val in enumerate(values):
                val = val.strip()
                if not val:
                    continue
                # Skip likely headers
                if i == 0 and val.lower() in ("headline", "headlines", "title", "text"):
                    continue
                headlines.append(val)

            corpus = WinnerCorpus(
                headlines=headlines,
                normalized=[normalize_headline(h) for h in headlines],
                loaded_at=datetime.now(timezone.utc),
                source="sheets",
            )

            logger.info(
                "winners_loaded_from_sheets",
                count=corpus.count,
                sheet_id=self.settings.winners_sheet_id,
            )

            return corpus

        except Exception as e:
            logger.error("winners_sheets_load_failed", error=str(e))
            raise

    def load_from_csv(self, path: Optional[str] = None) -> WinnerCorpus:
        """Load winner headlines from local CSV file."""
        csv_path = Path(path or self.settings.winners_csv_path)

        if not csv_path.exists():
            logger.warning("winners_csv_not_found", path=str(csv_path))
            return WinnerCorpus(source="csv_missing")

        headlines = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if not row:
                        continue
                    val = row[0].strip()
                    if not val:
                        continue
                    # Skip header
                    if i == 0 and val.lower() in ("headline", "headlines", "title", "text"):
                        continue
                    headlines.append(val)

            corpus = WinnerCorpus(
                headlines=headlines,
                normalized=[normalize_headline(h) for h in headlines],
                loaded_at=datetime.now(timezone.utc),
                source="csv",
            )

            logger.info("winners_loaded_from_csv", count=corpus.count, path=str(csv_path))
            return corpus

        except Exception as e:
            logger.error("winners_csv_load_failed", error=str(e), path=str(csv_path))
            return WinnerCorpus(source="csv_error")

    def load_from_cache(self) -> Optional[WinnerCorpus]:
        """Load winner headlines from database cache."""
        try:
            with get_session() as session:
                cached = session.query(WinnerHeadline).all()

                if not cached:
                    return None

                # Check freshness
                latest = max(w.cached_at for w in cached)
                age = datetime.now(timezone.utc) - latest.replace(tzinfo=timezone.utc)

                if age > timedelta(hours=self.settings.winners_refresh_hours):
                    logger.info("winners_cache_stale", age_hours=age.total_seconds()/3600)
                    return None

                headlines = [w.headline for w in cached]

                corpus = WinnerCorpus(
                    headlines=headlines,
                    normalized=[normalize_headline(h) for h in headlines],
                    loaded_at=latest.replace(tzinfo=timezone.utc),
                    source="cache",
                )

                logger.info("winners_loaded_from_cache", count=corpus.count)
                return corpus

        except Exception as e:
            logger.warning("winners_cache_load_failed", error=str(e))
            return None

    def save_to_cache(self, corpus: WinnerCorpus) -> None:
        """Save winner headlines to database cache."""
        try:
            with get_session() as session:
                # Clear existing
                session.query(WinnerHeadline).delete()

                # Insert new
                now = datetime.now(timezone.utc)
                for headline in corpus.headlines:
                    headline_hash = hashlib.sha256(headline.encode()).hexdigest()[:32]
                    session.add(WinnerHeadline(
                        headline=headline,
                        headline_hash=headline_hash,
                        cached_at=now,
                    ))

                session.commit()
                logger.info("winners_saved_to_cache", count=corpus.count)

        except Exception as e:
            logger.error("winners_cache_save_failed", error=str(e))

    def load(self, force_refresh: bool = False) -> WinnerCorpus:
        """
        Load winner corpus with caching strategy.

        Order: cache -> sheets -> csv fallback
        """
        # Return cached if available and fresh
        if self._corpus and not self._corpus.is_stale and not force_refresh:
            return self._corpus

        # Try database cache first
        if not force_refresh:
            cached = self.load_from_cache()
            if cached and not cached.is_empty:
                self._corpus = cached
                return cached

        # Try Google Sheets
        try:
            corpus = self.load_from_sheets()
            if not corpus.is_empty:
                self.save_to_cache(corpus)
                self._corpus = corpus
                return corpus
        except Exception as e:
            logger.warning("winners_sheets_failed_trying_csv", error=str(e))

        # Fallback to local CSV
        corpus = self.load_from_csv()
        if not corpus.is_empty:
            self._corpus = corpus
            return corpus

        # Return empty corpus as last resort
        logger.warning("winners_all_sources_failed")
        return WinnerCorpus(source="none")

    def get_templates(self) -> dict[str, int]:
        """Get template pattern frequencies from corpus."""
        corpus = self.load()
        if corpus.is_empty:
            return {}
        return extract_template_patterns(corpus.normalized)


# Singleton loader
_loader: Optional[WinnerLoader] = None


def get_winner_loader() -> WinnerLoader:
    """Get singleton winner loader instance."""
    global _loader
    if _loader is None:
        _loader = WinnerLoader()
    return _loader


def get_winner_corpus(force_refresh: bool = False) -> WinnerCorpus:
    """Convenience function to get winner corpus."""
    return get_winner_loader().load(force_refresh)
