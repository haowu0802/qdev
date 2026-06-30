# qdev — Market Forecast Evaluation Pipeline

A reproducible research pipeline that evaluates **algorithmic forecasting strategies** on resolvable market trend questions. The core object model is strategy × question × forecast × resolution → proper scoring, applied to market data.

> **Not a trading tool.** This project measures *how accurately strategies predict market outcomes*, scored with proper scoring rules (Brier score, calibration).

## What it does

```text
ingest (yfinance) → raw.prices (DuckDB)
                  → dbt staging (stg_prices)
                  → [planned] questions → strategies → resolve → score → leaderboard
```

| Layer | Status | Description |
|---|---|---|
| **ingest/** | ✅ | Source-agnostic market data providers (yfinance, no API key) |
| **warehouse/** | ✅ | DuckDB local analytics engine + dbt transformations |
| **questions/** | 🔜 | Resolvable market trend questions (`dim_question`) |
| **strategies/** | 🔜 | Forecasting strategies (`dim_strategy`, `fct_forecast`) |
| **resolve/** | 🔜 | Outcome resolution from market data (`fct_resolution`) |
| **score/** | 🔜 | Brier score, calibration, accuracy marts |

## Quick start

### 1. Clone and set up a virtual environment

```bash
git clone https://github.com/haowu0802/qdev.git
cd qdev
python -m venv .venv
```

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Ingest market data

No API key required — uses yfinance:

```bash
python -m ingest.run_ingest
```

Default: `SPY`, `GLD`, `QQQ` for calendar year 2023. Custom tickers:

```bash
python -m ingest.run_ingest --tickers SPY,QQQ --start 2023-01-01 --end 2023-12-31
```

Data lands in `warehouse/qdev.duckdb` → schema `raw`, table `prices`.

### 3. Run dbt transformations

```bash
cd dbt
dbt run --profiles-dir .
dbt test --profiles-dir .
```

This builds `main_staging.stg_prices` (a cleaned view over `raw.prices`) and runs data quality tests.

> **Note:** Close any open Python/DuckDB sessions before running dbt. DuckDB uses an exclusive file lock on Windows — a lingering `python` REPL that connected to `qdev.duckdb` will block dbt.

## Project structure

```text
qdev/
├── ingest/                 # Data ingestion (source-agnostic providers)
│   ├── base.py             # MarketDataProvider ABC
│   ├── quality.py          # Data quality checks
│   ├── run_ingest.py       # CLI entrypoint
│   └── providers/
│       └── yfinance_provider.py
├── warehouse/
│   └── duckdb_io.py        # DuckDB read/write helpers
├── dbt/                    # dbt-duckdb project
│   ├── profiles.yml        # Points at warehouse/qdev.duckdb
│   └── models/staging/
│       ├── stg_prices.sql
│       └── schema.yml      # Column tests
├── questions/              # [planned] Question definitions
├── strategies/             # [planned] Forecasting strategies
├── resolve/                # [planned] Outcome resolution
├── score/                  # [planned] Brier / calibration
└── check_cointegration.py  # Legacy v1 script (cointegration analysis; to be
                            # refactored into mean_reversion strategy)
```

## Design principles

- **Source-agnostic ingest** — swap data providers without touching downstream code
- **Idempotent loads** — safe to re-run ingestion; no duplicate rows
- **Visible data quality** — log issues instead of silently dropping data
- **Reproducible locally** — DuckDB file + dbt; no cloud account needed to run
- **Proper scoring** — forecasts evaluated with Brier score and calibration (planned)

## Legacy: cointegration analysis (v1)

The original `check_cointegration.py` (GLD/GDX pair trading analysis) is preserved for reference. In v2, cointegration signals will be refactored into a `mean_reversion` **strategy** whose probabilistic forecasts are scored with Brier — turning a trading signal into a falsifiable forecaster.

## Requirements

- Python 3.10+
- See `requirements.txt` for dependencies (pandas, yfinance, duckdb, dbt-duckdb, statsmodels)

## License

MIT — see [LICENSE](LICENSE).
