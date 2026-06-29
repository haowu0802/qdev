"""
DuckDB input/output helpers.

DuckDB is an in-process columnar analytics engine ("the SQLite of analytics").
It needs no server, so a reviewer can clone the repo and run the whole pipeline
against a single local file. This mirrors a warehouse (dim/fact, window
functions, CTEs) while staying fully reproducible. In production the same models
can target Postgres; see the project direction notes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

# Default warehouse file lives next to this module, under warehouse/.
DEFAULT_DB_PATH = Path(__file__).resolve().parent / "qdev.duckdb"

# Schema that holds untransformed source data, before staging/marts.
RAW_SCHEMA = "raw"
RAW_PRICES_TABLE = f"{RAW_SCHEMA}.prices"


def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    """Open (creating if needed) a DuckDB connection to ``db_path``."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Opening DuckDB at %s", path)
    return duckdb.connect(str(path))


def write_raw_prices(
    df: pd.DataFrame,
    con: duckdb.DuckDBPyConnection,
    source: str,
) -> int:
    """Write tidy price rows into ``raw.prices`` (idempotent upsert by key).

    A ``source`` and ``ingested_at`` column are added for lineage. Existing rows
    with the same (ticker, date, source) are replaced so re-running ingestion is
    safe and does not create duplicates.

    Returns:
        Number of rows written in this batch.
    """
    if df.empty:
        logger.warning("No rows to write to %s", RAW_PRICES_TABLE)
        return 0

    staged = df.copy()
    staged["source"] = source
    staged["ingested_at"] = pd.Timestamp.utcnow().tz_localize(None)

    con.execute(f"CREATE SCHEMA IF NOT EXISTS {RAW_SCHEMA};")
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {RAW_PRICES_TABLE} (
            ticker      VARCHAR,
            date        DATE,
            open        DOUBLE,
            high        DOUBLE,
            low         DOUBLE,
            close       DOUBLE,
            adj_close   DOUBLE,
            volume      DOUBLE,
            source      VARCHAR,
            ingested_at TIMESTAMP
        );
        """
    )

    # Idempotent: delete the keys we are about to insert, then insert fresh.
    con.register("staged_prices", staged)
    con.execute(
        f"""
        DELETE FROM {RAW_PRICES_TABLE} t
        USING staged_prices s
        WHERE t.ticker = s.ticker
          AND t.date = s.date
          AND t.source = s.source;
        """
    )
    con.execute(
        f"""
        INSERT INTO {RAW_PRICES_TABLE}
        SELECT ticker, date, open, high, low, close, adj_close, volume,
               source, ingested_at
        FROM staged_prices;
        """
    )
    con.unregister("staged_prices")

    logger.info("Wrote %d rows to %s (source=%s)", len(staged), RAW_PRICES_TABLE, source)
    return len(staged)


def summarize_raw_prices(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Return a small per-ticker summary of what is currently in raw.prices."""
    return con.execute(
        f"""
        SELECT
            ticker,
            COUNT(*)        AS n_rows,
            MIN(date)       AS first_date,
            MAX(date)       AS last_date,
            ROUND(AVG(adj_close), 2) AS avg_adj_close
        FROM {RAW_PRICES_TABLE}
        GROUP BY ticker
        ORDER BY ticker;
        """
    ).df()
