-- =============================================================================
-- HVFHV Pipeline — DuckDB SQL
-- Source: src/build_features_hvfhv.py
-- Run via DuckDB with the httpfs extension loaded:
--     INSTALL httpfs; LOAD httpfs;
-- Each block below runs once per month; replace the month in the URL
-- (e.g. fhvhv_tripdata_2025-01.parquet) to process a different month.
-- =============================================================================

INSTALL httpfs;
LOAD httpfs;

-- -----------------------------------------------------------------------------
-- 1. Raw trips (filtered), per month
-- Source for all three tables below.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW raw_trips_month AS
SELECT
    hvfhs_license_num, pickup_datetime, dropoff_datetime,
    PULocationID, DOLocationID, trip_miles,
    trip_time AS trip_duration_s,
    shared_request_flag, wav_request_flag
FROM read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2025-01.parquet')
WHERE pickup_datetime IS NOT NULL AND dropoff_datetime IS NOT NULL
  AND trip_time BETWEEN 60 AND 10800        -- 1 min to 3 hours
  AND trip_miles > 0 AND trip_miles < 100
  AND PULocationID IS NOT NULL AND DOLocationID IS NOT NULL
  AND pickup_datetime >= DATE '2025-01-01'
  AND pickup_datetime <  DATE '2026-01-01';


-- -----------------------------------------------------------------------------
-- 2. Task 1 — Trip features (duration model input)
-- Reservoir-sampled subset, per month (166,666 ~= 2,000,000 / 12 months).
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TABLE trip_features_month AS
SELECT
    trip_duration_s, trip_miles, PULocationID, DOLocationID, hvfhs_license_num,
    CAST(shared_request_flag = 'Y' AS INTEGER) AS is_shared_request,
    CAST(wav_request_flag   = 'Y' AS INTEGER)  AS is_wav_request,
    hour(pickup_datetime)                       AS pickup_hour,
    dayofweek(pickup_datetime)                  AS pickup_dow,
    CASE WHEN dayofweek(pickup_datetime) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,
    month(pickup_datetime)                       AS pickup_month
FROM (SELECT * FROM raw_trips_month USING SAMPLE 166666 (reservoir));


-- -----------------------------------------------------------------------------
-- 3. Task 2 — Citywide hourly demand
-- Full month, no sampling.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TABLE hourly_demand_month AS
SELECT date_trunc('hour', pickup_datetime) AS pickup_hour_ts, count(*) AS trip_count
FROM raw_trips_month
GROUP BY 1;


-- -----------------------------------------------------------------------------
-- 4. Task 3 — Zone-hourly demand
-- Full month, joined to the zone lookup CSV (also streamed over HTTPS).
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TABLE zone_hourly_demand_month AS
SELECT
    date_trunc('hour', r.pickup_datetime) AS pickup_hour_ts,
    r.PULocationID                        AS zone_id,
    z.Zone                                AS zone_name,
    z.Borough                             AS borough,
    count(*)                              AS trip_count
FROM raw_trips_month r
LEFT JOIN read_csv_auto('https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv') z
    ON r.PULocationID = z.LocationID
GROUP BY 1, 2, 3, 4;


-- =============================================================================
-- 5. Cross-month finalization
-- Run once, after all months have been unioned into hourly_demand_all /
-- zone_hourly_demand_all (see build_features_hvfhv.py's append logic —
-- first month uses CREATE TABLE AS, subsequent months use INSERT INTO).
-- =============================================================================
CREATE OR REPLACE TABLE hourly_demand_final AS
SELECT pickup_hour_ts, sum(trip_count) AS trip_count
FROM hourly_demand_all
GROUP BY 1
ORDER BY 1;

CREATE OR REPLACE TABLE zone_hourly_demand_final AS
SELECT pickup_hour_ts, zone_id, zone_name, borough, sum(trip_count) AS trip_count
FROM zone_hourly_demand_all
GROUP BY 1, 2, 3, 4
ORDER BY 1, 2;

-- trip_features_all needs no re-aggregation: row-level sampled data is
-- simply concatenated across months (INSERT INTO ... SELECT *), never grouped.


-- -----------------------------------------------------------------------------
-- 6. Final export (what build_features_hvfhv.py writes to data/processed/)
-- -----------------------------------------------------------------------------
COPY trip_features_all       TO 'data/processed/hvfhv_trip_features.parquet'       (FORMAT PARQUET);
COPY hourly_demand_final     TO 'data/processed/hvfhv_hourly_demand.parquet'       (FORMAT PARQUET);
COPY zone_hourly_demand_final TO 'data/processed/hvfhv_zone_hourly_demand.parquet' (FORMAT PARQUET);
