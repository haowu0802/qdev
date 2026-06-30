-- Staging layer: clean, typed view over raw market prices.
-- Grain: one row per (ticker, date, source).

select
    ticker,
    cast(date as date) as date,
    open,
    high,
    low,
    close,
    adj_close,
    volume,
    source,
    ingested_at
from {{ source('raw', 'prices') }}
