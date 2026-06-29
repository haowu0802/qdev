"""
Source-agnostic market data provider interface.

Every concrete data source (yfinance, Tiingo, FRED, Stooq, ...) implements
``MarketDataProvider`` so the rest of the pipeline never depends on a specific
vendor. This keeps ingestion pluggable, which is a core data-engineering
quality signal and lets us add or swap sources without touching downstream code.
"""

from __future__ import annotations

import abc
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Canonical tidy schema every provider must return.
# One row per (ticker, date); columns below in this exact order.
RAW_PRICE_COLUMNS = [
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]


class MarketDataProvider(abc.ABC):
    """Abstract base class for all market data sources.

    Implementations must return a tidy (long) DataFrame conforming to
    ``RAW_PRICE_COLUMNS``: one row per ticker per trading day. Returning a
    single canonical shape (instead of each vendor's native format) is what
    makes the ingestion layer source-agnostic.
    """

    #: Short, stable identifier stored alongside ingested rows for lineage.
    name: str = "base"

    @abc.abstractmethod
    def fetch(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch historical daily OHLCV data for the given tickers.

        Args:
            tickers: Ticker symbols, e.g. ``["SPY", "GLD"]``.
            start_date: Inclusive start date, ``YYYY-MM-DD``.
            end_date: Inclusive end date, ``YYYY-MM-DD``.

        Returns:
            Tidy DataFrame with columns ``RAW_PRICE_COLUMNS`` (one row per
            ticker per day), sorted by ``(ticker, date)``.
        """
        raise NotImplementedError

    def _finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize a provider's output to the canonical schema.

        Concrete providers should call this on their assembled frame before
        returning, so schema enforcement lives in one place.
        """
        missing = [c for c in RAW_PRICE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Provider '{self.name}' output missing columns: {missing}"
            )

        df = df[RAW_PRICE_COLUMNS].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        logger.info(
            "Provider '%s' returned %d rows for %d tickers (%s to %s)",
            self.name,
            len(df),
            df["ticker"].nunique(),
            df["date"].min().date() if not df.empty else "n/a",
            df["date"].max().date() if not df.empty else "n/a",
        )
        return df
