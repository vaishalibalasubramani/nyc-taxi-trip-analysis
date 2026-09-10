-- Task 2: citywide hourly demand (pickup counts per hour).
-- Depends on raw_trips (see 01_raw_trips_view.sql).

CREATE OR REPLACE TABLE hourly_demand AS
SELECT
    date_trunc('hour', pickup_datetime) AS pickup_hour_ts,
    count(*)                            AS trip_count
FROM raw_trips
GROUP BY 1
ORDER BY 1;

-- COPY hourly_demand TO 'data/processed/hourly_demand.parquet' (FORMAT PARQUET);
