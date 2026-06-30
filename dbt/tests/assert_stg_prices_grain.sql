-- Fail if grain (ticker, date, source) is violated.
select
    ticker,
    date,
    source,
    count(*) as n
from {{ ref('stg_prices') }}
group by ticker, date, source
having count(*) > 1
