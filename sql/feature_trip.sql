-- Task 1: trip-level features for duration prediction.
-- One row per trip. Depends on raw_trips (see 01_raw_trips_view.sql).

CREATE OR REPLACE TABLE trip_features AS
SELECT
    trip_duration_s,
    trip_distance,
    passenger_count,
    PULocationID,
    DOLocationID,
    trip_type,
    payment_type,
    hour(pickup_datetime)                AS pickup_hour,
    dayofweek(pickup_datetime)           AS pickup_dow,
    CASE WHEN dayofweek(pickup_datetime) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,
    month(pickup_datetime)               AS pickup_month
FROM raw_trips;

-- COPY trip_features TO 'data/processed/trip_features.parquet' (FORMAT PARQUET);
