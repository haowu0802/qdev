"""
Ingestion entrypoint: fetch market data -> quality check -> load into DuckDB.

This is the first orchestrated step of the pipeline
(ingest -> resolve -> score). Run it directly:

    python -m ingest.run_ingest --tickers SPY,GLD,QQQ --start 2023-01-01 --end 2023-12-31

With no arguments it ingests a small default basket so the project runs
out of the box with no API key (yfinance).
"""

from __future__ import annotations

import argparse
import logging

from ingest.providers.yfinance_provider import YFinanceProvider
from ingest.quality import run_quality_checks
from warehouse.duckdb_io import (
    DEFAULT_DB_PATH,
    get_connection,
    summarize_raw_prices,
    write_raw_prices,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ingest")

DEFAULT_TICKERS = ["SPY", "GLD", "QQQ"]
DEFAULT_START = "2023-01-01"
DEFAULT_END = "2023-12-31"


def run(
    tickers: list[str],
    start_date: str,
    end_date: str,
    db_path: str = str(DEFAULT_DB_PATH),
) -> int:
    """Run one ingestion pass. Returns the number of rows loaded."""
    provider = YFinanceProvider()
    df = provider.fetch(tickers, start_date, end_date)

    report = run_quality_checks(df)
    report.log()
    if not report.ok:
        logger.warning("Quality issues detected; loading anyway for visibility.")

    con = get_connection(db_path)
    try:
        rows = write_raw_prices(df, con, source=provider.name)
        logger.info("\nWarehouse summary (raw.prices):")
        logger.info("\n%s", summarize_raw_prices(con).to_string(index=False))
    finally:
        con.close()

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest daily market data into the DuckDB warehouse."
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=",".join(DEFAULT_TICKERS),
        help="Comma-separated ticker symbols (default: SPY,GLD,QQQ)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=DEFAULT_START,
        help="Start date YYYY-MM-DD (default: 2023-01-01)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=DEFAULT_END,
        help="End date YYYY-MM-DD (default: 2023-12-31)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=str(DEFAULT_DB_PATH),
        help="Path to the DuckDB file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    run(tickers, args.start, args.end, args.db)


if __name__ == "__main__":
    main()
