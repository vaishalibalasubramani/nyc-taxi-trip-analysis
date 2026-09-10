-- Base cleaned view of green taxi trips.
-- Reads directly from the TLC's HTTPS parquet files via DuckDB httpfs
-- (no local download). Run build_features.py to execute this against
-- your chosen months -- this file documents the exact query used there.

-- INSTALL httpfs; LOAD httpfs;   -- (done automatically by build_features.py)

CREATE OR REPLACE VIEW raw_trips AS
SELECT
    lpep_pickup_datetime  AS pickup_datetime,
    lpep_dropoff_datetime AS dropoff_datetime,
    PULocationID,
    DOLocationID,
    passenger_count,
    trip_distance,
    trip_type,
    fare_amount,
    total_amount,
    payment_type,
    date_diff('second', lpep_pickup_datetime, lpep_dropoff_datetime) AS trip_duration_s
FROM read_parquet(
    ['https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2025-01.parquet' /* , ... more months */],
    union_by_name=True
)
WHERE lpep_pickup_datetime IS NOT NULL
  AND lpep_dropoff_datetime IS NOT NULL
  -- Sanity filters: drop bad/impossible records
  AND date_diff('second', lpep_pickup_datetime, lpep_dropoff_datetime) BETWEEN 60 AND 10800  -- 1 min .. 3 hr
  AND trip_distance > 0 AND trip_distance < 100
  AND passenger_count > 0
  AND PULocationID IS NOT NULL
  AND DOLocationID IS NOT NULL
  -- Drop corrupted timestamps (a small number of TLC records have garbage
  -- dates years outside the actual reporting month)
  AND lpep_pickup_datetime >= DATE '2025-01-01'
  AND lpep_pickup_datetime <  DATE '2026-01-01';
