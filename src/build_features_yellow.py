"""
Build feature tables for all three modeling tasks by querying yellow taxi
parquet files DIRECTLY over HTTPS with DuckDB's httpfs extension.

No manual download step: DuckDB reads the TLC Parquet files directly
from the TLC CloudFront URLs.

Usage:
    python src/build_features_yellow.py --months 2025-01 2025-02 2025-03

Produces (in data/processed/):
    yellow_trip_features.parquet
        -> row-level, for Task 1 (duration prediction)

    yellow_hourly_demand.parquet
        -> citywide hourly counts, for Task 2

    yellow_zone_hourly_demand.parquet
        -> per-zone hourly counts, for Task 3
"""

import argparse
from pathlib import Path

import duckdb


BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"

ZONE_LOOKUP_URL = (
    "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
)

PROCESSED_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
)


def main():

    parser = argparse.ArgumentParser(
        description="Build feature tables from remote yellow taxi data"
    )

    parser.add_argument(
        "--months",
        nargs="+",
        required=True,
        help="One or more YYYY-MM values, e.g. 2025-01 2025-02 2025-03",
    )

    args = parser.parse_args()

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    urls = [
        f"{BASE_URL}/yellow_tripdata_{month}.parquet"
        for month in args.months
    ]

    print(
        "Querying remote Yellow Taxi files "
        "(no local download):"
    )

    for url in urls:
        print(" -", url)

    con = duckdb.connect()

    con.execute(
        "INSTALL httpfs; LOAD httpfs;"
    )

    # ---------------------------------------------------------------
    # BUILD SQL LIST OF REMOTE PARQUET URLS
    # ---------------------------------------------------------------

    url_list_sql = "[" + ", ".join(
        f"'{url}'"
        for url in urls
    ) + "]"

    # ---------------------------------------------------------------
    # RAW / CLEANED SOURCE
    # ---------------------------------------------------------------

    con.execute(
        f"""
        CREATE OR REPLACE VIEW raw_trips AS

        SELECT
            tpep_pickup_datetime  AS pickup_datetime,
            tpep_dropoff_datetime AS dropoff_datetime,

            PULocationID,
            DOLocationID,

            passenger_count,
            trip_distance,

            RatecodeID,
            payment_type,

            fare_amount,
            total_amount,

            date_diff(
                'second',
                tpep_pickup_datetime,
                tpep_dropoff_datetime
            ) AS trip_duration_s

        FROM read_parquet(
            {url_list_sql},
            union_by_name=True
        )

        WHERE
            tpep_pickup_datetime IS NOT NULL

            AND tpep_dropoff_datetime IS NOT NULL

            -- Duration: 1 minute to 3 hours
            AND date_diff(
                'second',
                tpep_pickup_datetime,
                tpep_dropoff_datetime
            ) BETWEEN 60 AND 10800

            -- Reasonable trip distance
            AND trip_distance > 0
            AND trip_distance < 100

            -- Valid passenger count
            AND passenger_count > 0

            -- Valid pickup/dropoff zones
            AND PULocationID IS NOT NULL
            AND DOLocationID IS NOT NULL

            -- Only 2025
            AND tpep_pickup_datetime >= DATE '2025-01-01'
            AND tpep_pickup_datetime < DATE '2026-01-01'
        """
    )

    n_raw = con.execute(
        "SELECT count(*) FROM raw_trips"
    ).fetchone()[0]

    print(
        f"\nRows after sanity filtering: "
        f"{n_raw:,}"
    )

    # ---------------------------------------------------------------
    # TASK 1
    # TRIP DURATION PREDICTION
    # ---------------------------------------------------------------

    print(
        "\nBuilding yellow_trip_features.parquet "
        "(Task 1)..."
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE yellow_trip_features AS

        SELECT
            -- Keep the actual pickup timestamp so that
            -- duration training can perform a proper
            -- time-based train/test split.
            pickup_datetime,

            trip_duration_s,

            trip_distance,

            passenger_count,

            PULocationID,
            DOLocationID,

            RatecodeID,
            payment_type,

            hour(
                pickup_datetime
            ) AS pickup_hour,

            dayofweek(
                pickup_datetime
            ) AS pickup_dow,

            CASE
                WHEN dayofweek(
                    pickup_datetime
                ) IN (0, 6)
                THEN 1
                ELSE 0
            END AS is_weekend,

            month(
                pickup_datetime
            ) AS pickup_month

        FROM raw_trips
        """
    )

    con.execute(
        f"""
        COPY yellow_trip_features

        TO '{PROCESSED_DIR / "yellow_trip_features.parquet"}'

        (FORMAT PARQUET)
        """
    )

    print(
        "  -> "
        f"{con.execute('SELECT count(*) FROM yellow_trip_features').fetchone()[0]:,}"
        " rows"
    )

    # ---------------------------------------------------------------
    # TASK 2
    # CITYWIDE HOURLY DEMAND
    # ---------------------------------------------------------------

    print(
        "\nBuilding yellow_hourly_demand.parquet "
        "(Task 2)..."
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE yellow_hourly_demand AS

        SELECT
            date_trunc(
                'hour',
                pickup_datetime
            ) AS pickup_hour_ts,

            count(*) AS trip_count

        FROM raw_trips

        GROUP BY 1

        ORDER BY 1
        """
    )

    con.execute(
        f"""
        COPY yellow_hourly_demand

        TO '{PROCESSED_DIR / "yellow_hourly_demand.parquet"}'

        (FORMAT PARQUET)
        """
    )

    print(
        "  -> "
        f"{con.execute('SELECT count(*) FROM yellow_hourly_demand').fetchone()[0]:,}"
        " hourly rows"
    )

    # ---------------------------------------------------------------
    # TASK 3
    # ZONE-LEVEL HOURLY DEMAND
    # ---------------------------------------------------------------

    print(
        "\nBuilding yellow_zone_hourly_demand.parquet "
        "(Task 3)..."
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE yellow_zone_hourly_demand AS

        SELECT
            date_trunc(
                'hour',
                r.pickup_datetime
            ) AS pickup_hour_ts,

            r.PULocationID AS zone_id,

            z.Zone AS zone_name,

            z.Borough AS borough,

            count(*) AS trip_count

        FROM raw_trips r

        LEFT JOIN read_csv_auto(
            '{ZONE_LOOKUP_URL}'
        ) z

            ON r.PULocationID = z.LocationID

        GROUP BY
            1,
            2,
            3,
            4

        ORDER BY
            1,
            2
        """
    )

    con.execute(
        f"""
        COPY yellow_zone_hourly_demand

        TO '{PROCESSED_DIR / "yellow_zone_hourly_demand.parquet"}'

        (FORMAT PARQUET)
        """
    )

    print(
        "  -> "
        f"{con.execute('SELECT count(*) FROM yellow_zone_hourly_demand').fetchone()[0]:,}"
        " zone-hour rows"
    )

    # ---------------------------------------------------------------
    # FINISHED
    # ---------------------------------------------------------------

    print(
        "\nDone. Yellow Taxi processed files are in:",
        PROCESSED_DIR
    )

    con.close()


if __name__ == "__main__":
    main()