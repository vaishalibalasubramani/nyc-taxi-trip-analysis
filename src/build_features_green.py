"""
Build feature tables for all three modeling tasks by querying green taxi
parquet files DIRECTLY over HTTPS with DuckDB's httpfs extension.

No manual download step: DuckDB uses HTTP range requests to read only the
row groups/columns it needs, straight from the TLC's CloudFront URLs.

Usage:
    python src/build_features.py --months 2025-01 2025-02 2025-03

Produces (in data/processed/):
    trip_features.parquet        -> row-level, for Task 1 (duration prediction)
    hourly_demand.parquet        -> citywide hourly counts, for Task 2
    zone_hourly_demand.parquet   -> per-zone hourly counts, for Task 3
"""

import argparse
from pathlib import Path

import duckdb

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def main():
    parser = argparse.ArgumentParser(description="Build feature tables from remote green taxi data")
    parser.add_argument(
        "--months",
        nargs="+",
        required=True,
        help="One or more YYYY-MM values, e.g. 2025-01 2025-02 2025-03",
    )
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    urls = [f"{BASE_URL}/green_tripdata_{m}.parquet" for m in args.months]
    print("Querying remote files (no local download):")
    for u in urls:
        print(" -", u)

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")

    # union_by_name=True protects against schema drift between months
    # (e.g. the cbd_congestion_fee column that 2025 files added).
    url_list_sql = "[" + ", ".join(f"'{u}'" for u in urls) + "]"

    con.execute(
        f"""
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
        FROM read_parquet({url_list_sql}, union_by_name=True)
        WHERE lpep_pickup_datetime IS NOT NULL
          AND lpep_dropoff_datetime IS NOT NULL
          AND date_diff('second', lpep_pickup_datetime, lpep_dropoff_datetime) BETWEEN 60 AND 10800
          AND trip_distance > 0 AND trip_distance < 100
          AND passenger_count > 0
          AND PULocationID IS NOT NULL
          AND DOLocationID IS NOT NULL
          AND lpep_pickup_datetime >= DATE '2025-01-01'
          AND lpep_pickup_datetime < DATE '2026-01-01'
        """
    )

    n_raw = con.execute("SELECT count(*) FROM raw_trips").fetchone()[0]
    print(f"\nRows after sanity filtering: {n_raw:,}")

    # ---------------------------------------------------------------
    # Task 1: trip-level features for duration prediction
    # ---------------------------------------------------------------
    print("\nBuilding trip_features.parquet (Task 1)...")
    con.execute(
        """
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
            dayofweek(pickup_datetime)            AS pickup_dow,
            CASE WHEN dayofweek(pickup_datetime) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,
            month(pickup_datetime)                AS pickup_month
        FROM raw_trips
        """
    )
    con.execute(
        f"COPY trip_features TO '{PROCESSED_DIR / 'trip_features.parquet'}' (FORMAT PARQUET)"
    )
    print(f"  -> {con.execute('SELECT count(*) FROM trip_features').fetchone()[0]:,} rows")

    # ---------------------------------------------------------------
    # Task 2: citywide hourly demand (pickup counts per hour)
    # ---------------------------------------------------------------
    print("\nBuilding hourly_demand.parquet (Task 2)...")
    con.execute(
        """
        CREATE OR REPLACE TABLE hourly_demand AS
        SELECT
            date_trunc('hour', pickup_datetime) AS pickup_hour_ts,
            count(*)                            AS trip_count
        FROM raw_trips
        GROUP BY 1
        ORDER BY 1
        """
    )
    con.execute(
        f"COPY hourly_demand TO '{PROCESSED_DIR / 'hourly_demand.parquet'}' (FORMAT PARQUET)"
    )
    print(f"  -> {con.execute('SELECT count(*) FROM hourly_demand').fetchone()[0]:,} hourly rows")

    # ---------------------------------------------------------------
    # Task 3: per-zone hourly demand, joined with zone names/boroughs
    # ---------------------------------------------------------------
    print("\nBuilding zone_hourly_demand.parquet (Task 3)...")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE zone_hourly_demand AS
        SELECT
            date_trunc('hour', r.pickup_datetime) AS pickup_hour_ts,
            r.PULocationID                        AS zone_id,
            z.Zone                                AS zone_name,
            z.Borough                             AS borough,
            count(*)                              AS trip_count
        FROM raw_trips r
        LEFT JOIN read_csv_auto('{ZONE_LOOKUP_URL}') z
            ON r.PULocationID = z.LocationID
        GROUP BY 1, 2, 3, 4
        ORDER BY 1, 2
        """
    )
    con.execute(
        f"COPY zone_hourly_demand TO '{PROCESSED_DIR / 'zone_hourly_demand.parquet'}' (FORMAT PARQUET)"
    )
    print(f"  -> {con.execute('SELECT count(*) FROM zone_hourly_demand').fetchone()[0]:,} zone-hour rows")

    print("\nDone. Processed files in", PROCESSED_DIR)


if __name__ == "__main__":
    main()
