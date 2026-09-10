-- Task 3: per-zone hourly demand, joined with zone names/boroughs.
-- Depends on raw_trips (see 01_raw_trips_view.sql).

CREATE OR REPLACE TABLE zone_hourly_demand AS
SELECT
    date_trunc('hour', r.pickup_datetime) AS pickup_hour_ts,
    r.PULocationID                        AS zone_id,
    z.Zone                                AS zone_name,
    z.Borough                             AS borough,
    count(*)                              AS trip_count
FROM raw_trips r
LEFT JOIN read_csv_auto('https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv') z
    ON r.PULocationID = z.LocationID
GROUP BY 1, 2, 3, 4
ORDER BY 1, 2;

-- COPY zone_hourly_demand TO 'data/processed/zone_hourly_demand.parquet' (FORMAT PARQUET);
