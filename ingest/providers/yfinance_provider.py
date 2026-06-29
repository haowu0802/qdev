"""
yfinance-backed market data provider.

yfinance is the primary source for the showcase because it requires no API key,
so anyone who clones the repo can run the pipeline end to end. The provider
normalizes yfinance's wide, multi-index output into the canonical tidy schema
defined in ``ingest.base``.
"""

from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from ingest.base import MarketDataProvider

logger = logging.getLogger(__name__)


class YFinanceProvider(MarketDataProvider):
    """Fetch daily OHLCV data from Yahoo Finance via the ``yfinance`` library."""

    name = "yfinance"

    def fetch(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        if not tickers:
            raise ValueError("tickers must be a non-empty list")

        logger.info(
            "Fetching %s from yfinance (%s to %s)", tickers, start_date, end_date
        )

        # auto_adjust=False keeps both 'Close' and 'Adj Close' so we can store
        # the raw close and the dividend/split-adjusted close separately.
        raw = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
        )

        if raw is None or raw.empty:
            raise ValueError(f"yfinance returned no data for {tickers}")

        tidy = self._to_tidy(raw, tickers)
        return self._finalize(tidy)

    @staticmethod
    def _to_tidy(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
        """Convert yfinance output into a tidy long frame.

        yfinance returns a single-level column frame for one ticker and a
        two-level (ticker, field) frame for multiple tickers. Normalize both
        into one row per (ticker, date).
        """
        field_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }

        frames: list[pd.DataFrame] = []

        if isinstance(raw.columns, pd.MultiIndex):
            for ticker in tickers:
                if ticker not in raw.columns.get_level_values(0):
                    logger.warning("No data for ticker %s; skipping", ticker)
                    continue
                sub = raw[ticker].rename(columns=field_map).copy()
                sub.insert(0, "ticker", ticker)
                sub = sub.reset_index().rename(columns={"Date": "date"})
                frames.append(sub)
        else:
            sub = raw.rename(columns=field_map).copy()
            sub.insert(0, "ticker", tickers[0])
            sub = sub.reset_index().rename(columns={"Date": "date"})
            frames.append(sub)

        if not frames:
            raise ValueError("No usable ticker data after parsing yfinance output")

        out = pd.concat(frames, ignore_index=True)

        for col in ["open", "high", "low", "close", "adj_close", "volume"]:
            if col not in out.columns:
                out[col] = pd.NA

        return out
