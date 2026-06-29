"""
Data quality checks for ingested raw price data.

Runs lightweight, explainable checks on the canonical tidy frame before it is
loaded into the warehouse. The goal is honest visibility (log what is wrong)
rather than silently dropping data, which matches a research pipeline where
data correctness and lineage matter more than convenience.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from ingest.base import RAW_PRICE_COLUMNS

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Summary of data quality findings for one ingested batch."""

    row_count: int = 0
    ticker_count: int = 0
    date_min: str | None = None
    date_max: str | None = None
    null_counts: dict[str, int] = field(default_factory=dict)
    duplicate_rows: int = 0
    non_positive_close: int = 0
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no blocking issues were found."""
        return not self.issues

    def log(self) -> None:
        logger.info("=" * 50)
        logger.info("DATA QUALITY REPORT")
        logger.info("=" * 50)
        logger.info("Rows: %d | Tickers: %d", self.row_count, self.ticker_count)
        logger.info("Date range: %s -> %s", self.date_min, self.date_max)
        if any(v > 0 for v in self.null_counts.values()):
            logger.warning("Null values: %s", {k: v for k, v in self.null_counts.items() if v})
        else:
            logger.info("Null values: none")
        logger.info("Duplicate (ticker,date) rows: %d", self.duplicate_rows)
        logger.info("Non-positive close prices: %d", self.non_positive_close)
        if self.issues:
            logger.warning("Issues: %s", self.issues)
        else:
            logger.info("Status: PASS")
        logger.info("=" * 50)


def run_quality_checks(df: pd.DataFrame) -> QualityReport:
    """Compute a :class:`QualityReport` for a canonical tidy price frame."""
    report = QualityReport()

    missing = [c for c in RAW_PRICE_COLUMNS if c not in df.columns]
    if missing:
        report.issues.append(f"missing columns: {missing}")
        return report

    report.row_count = len(df)
    report.ticker_count = int(df["ticker"].nunique())
    if not df.empty:
        report.date_min = str(pd.to_datetime(df["date"]).min().date())
        report.date_max = str(pd.to_datetime(df["date"]).max().date())

    report.null_counts = {c: int(df[c].isna().sum()) for c in RAW_PRICE_COLUMNS}

    dup_mask = df.duplicated(subset=["ticker", "date"], keep=False)
    report.duplicate_rows = int(dup_mask.sum())
    if report.duplicate_rows:
        report.issues.append(f"{report.duplicate_rows} duplicate (ticker,date) rows")

    close = pd.to_numeric(df["close"], errors="coerce")
    report.non_positive_close = int((close <= 0).sum())
    if report.non_positive_close:
        report.issues.append(
            f"{report.non_positive_close} non-positive close prices"
        )

    if report.row_count == 0:
        report.issues.append("empty dataset")

    return report
