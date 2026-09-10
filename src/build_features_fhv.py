"""
Build FHV (For-Hire Vehicle) features for 2025.

Reads official NYC TLC FHV monthly Parquet files directly from
CloudFront using DuckDB HTTPFS.

Outputs:
    data/processed/fhv_trip_features.parquet
    data/processed/fhv_hourly_demand.parquet
    data/processed/fhv_zone_hourly_demand.parquet

Usage:
    python src/build_features_fhv.py --months 2025-01 2025-02 ... 2025-12
"""

from pathlib import Path
import argparse

import duckdb


# ---------------------------------------------------------------------
# PATHS / URLS
# ---------------------------------------------------------------------

BASE_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data"
)

ZONE_LOOKUP_URL = (
    "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
)

PROCESSED_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
)


# ---------------------------------------------------------------------
# BUILD REMOTE URLS
# ---------------------------------------------------------------------

def build_remote_urls(months):
    return [
        f"{BASE_URL}/fhv_tripdata_{month}.parquet"
        for month in months
    ]


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--months",
        nargs="+",
        required=True,
        help="Months to process, e.g. 2025-01 2025-02",
    )

    args = parser.parse_args()

    months = args.months

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    urls = build_remote_urls(months)

    print("=" * 70)
    print("FHV FEATURE BUILDER")
    print("=" * 70)

    print("\nMonths:")
    for month in months:
        print(f"  {month}")

    print("\nConnecting to DuckDB...")

    con = duckdb.connect()

    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")

    # ---------------------------------------------------------------
    # READ FHV DATA
    # ---------------------------------------------------------------

    print("\nReading FHV Parquet files directly from TLC...")

    url_list = "[" + ", ".join(
        "'" + url + "'"
        for url in urls
    ) + "]"

    # ---------------------------------------------------------------
    # COMMON CLEANED FHV DATA
    # ---------------------------------------------------------------

    # IMPORTANT:
    # All three outputs use the SAME cleaned trip population.
    #
    # This guarantees:
    #
    # citywide hourly demand
    #       =
    # sum of zone hourly demand
    #
    # ---------------------------------------------------------------

    common_cte = f"""
        WITH raw AS (
            SELECT
                pickup_datetime,
                dropOff_datetime,
                PUlocationID,
                DOlocationID,
                SR_Flag,
                dispatching_base_num,
                Affiliated_base_number

            FROM read_parquet(
                {url_list},
                union_by_name = true
            )
        ),

        cleaned AS (
            SELECT
                pickup_datetime,
                dropOff_datetime,
                PUlocationID,
                DOlocationID,

                COALESCE(
                    SR_Flag,
                    0
                ) AS SR_Flag,

                dispatching_base_num,
                Affiliated_base_number,

                EXTRACT(
                    EPOCH FROM
                    (
                        dropOff_datetime
                        - pickup_datetime
                    )
                ) AS trip_duration_s

            FROM raw

            WHERE
                pickup_datetime IS NOT NULL
                AND dropOff_datetime IS NOT NULL

                AND PUlocationID IS NOT NULL
                AND DOlocationID IS NOT NULL

                AND dropOff_datetime > pickup_datetime

                AND EXTRACT(
                    EPOCH FROM
                    (
                        dropOff_datetime
                        - pickup_datetime
                    )
                ) BETWEEN 60 AND 10800

                AND pickup_datetime >= TIMESTAMP '2025-01-01'
                AND pickup_datetime < TIMESTAMP '2026-01-01'
        )
    """

    # ---------------------------------------------------------------
    # TRIP-LEVEL FEATURES
    # ---------------------------------------------------------------

    print("\nBuilding FHV trip features...")

    trip_query = f"""
        COPY (
            {common_cte}

            SELECT
                trip_duration_s,

                PUlocationID,
                DOlocationID,

                SR_Flag,

                pickup_datetime,

                dispatching_base_num,
                Affiliated_base_number,

                EXTRACT(
                    HOUR FROM pickup_datetime
                ) AS pickup_hour,

                /*
                 * DuckDB DAYOFWEEK:
                 *   0 = Sunday
                 *   1 = Monday
                 *   2 = Tuesday
                 *   3 = Wednesday
                 *   4 = Thursday
                 *   5 = Friday
                 *   6 = Saturday
                 */
                EXTRACT(
                    DAYOFWEEK FROM pickup_datetime
                ) AS pickup_dow,

                /*
                 * Correct weekend definition:
                 *   Sunday = 0
                 *   Saturday = 6
                 */
                CASE
                    WHEN EXTRACT(
                        DAYOFWEEK FROM pickup_datetime
                    ) IN (0, 6)
                    THEN 1
                    ELSE 0
                END AS is_weekend,

                EXTRACT(
                    MONTH FROM pickup_datetime
                ) AS pickup_month

            FROM cleaned
        )
        TO '{PROCESSED_DIR / "fhv_trip_features.parquet"}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
    """

    con.execute(trip_query)

    trip_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet(
            '{PROCESSED_DIR / "fhv_trip_features.parquet"}'
        )
        """
    ).fetchone()[0]

    print(
        f"FHV trip features: {trip_count:,} rows"
    )

    # ---------------------------------------------------------------
    # CITYWIDE HOURLY DEMAND
    # ---------------------------------------------------------------

    print("\nBuilding FHV citywide hourly demand...")

    hourly_query = f"""
        COPY (
            {common_cte}

            SELECT
                DATE_TRUNC(
                    'hour',
                    pickup_datetime
                ) AS pickup_hour_ts,

                COUNT(*) AS trip_count

            FROM cleaned

            GROUP BY 1

            ORDER BY 1
        )
        TO '{PROCESSED_DIR / "fhv_hourly_demand.parquet"}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
    """

    con.execute(hourly_query)

    hourly_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet(
            '{PROCESSED_DIR / "fhv_hourly_demand.parquet"}'
        )
        """
    ).fetchone()[0]

    print(
        f"FHV hourly demand: {hourly_count:,} rows"
    )

    # ---------------------------------------------------------------
    # ZONE-LEVEL HOURLY DEMAND
    # ---------------------------------------------------------------

    print("\nBuilding FHV zone-level hourly demand...")

    zone_query = f"""
        COPY (
            {common_cte},

            demand AS (
                SELECT
                    DATE_TRUNC(
                        'hour',
                        pickup_datetime
                    ) AS pickup_hour_ts,

                    PUlocationID AS zone_id,

                    COUNT(*) AS trip_count

                FROM cleaned

                GROUP BY 1, 2
            )

            SELECT
                d.pickup_hour_ts,

                d.zone_id,

                z.Zone AS zone_name,

                z.Borough AS borough,

                d.trip_count

            FROM demand d

            LEFT JOIN read_csv_auto(
                '{ZONE_LOOKUP_URL}'
            ) z

            ON d.zone_id = z.LocationID

            ORDER BY
                d.pickup_hour_ts,
                d.zone_id
        )
        TO '{PROCESSED_DIR / "fhv_zone_hourly_demand.parquet"}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
    """

    con.execute(zone_query)

    zone_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet(
            '{PROCESSED_DIR / "fhv_zone_hourly_demand.parquet"}'
        )
        """
    ).fetchone()[0]

    print(
        f"FHV zone-hour demand: {zone_count:,} rows"
    )

    # ---------------------------------------------------------------
    # CONSISTENCY CHECK
    # ---------------------------------------------------------------

    print("\nChecking citywide vs zone demand consistency...")

    consistency = con.execute(
        f"""
        WITH city AS (
            SELECT
                pickup_hour_ts,
                trip_count AS citywide_count

            FROM read_parquet(
                '{PROCESSED_DIR / "fhv_hourly_demand.parquet"}'
            )
        ),

        zones AS (
            SELECT
                pickup_hour_ts,
                SUM(trip_count) AS zone_count

            FROM read_parquet(
                '{PROCESSED_DIR / "fhv_zone_hourly_demand.parquet"}'
            )

            GROUP BY pickup_hour_ts
        )

        SELECT
            MAX(
                ABS(
                    city.citywide_count
                    - COALESCE(zones.zone_count, 0)
                )
            ) AS max_difference,

            SUM(city.citywide_count)
                AS citywide_total,

            SUM(
                COALESCE(zones.zone_count, 0)
            ) AS zone_total

        FROM city

        LEFT JOIN zones
            ON city.pickup_hour_ts = zones.pickup_hour_ts
        """
    ).fetchone()

    max_difference = consistency[0]
    citywide_total = consistency[1]
    zone_total = consistency[2]

    print(
        f"\nCitywide total: {citywide_total:,}"
    )

    print(
        f"Zone total:     {zone_total:,}"
    )

    print(
        f"Maximum hourly difference: {max_difference:,}"
    )

    if max_difference == 0:
        print(
            "\n✓ CITYWIDE AND ZONE DEMAND ARE CONSISTENT"
        )
    else:
        print(
            "\nWARNING: CITYWIDE AND ZONE DEMAND DO NOT MATCH"
        )

    # ---------------------------------------------------------------
    # FINISHED
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FHV FEATURE BUILD COMPLETE")
    print("=" * 70)

    print(
        f"\nTrip features:"
        f"\n  {PROCESSED_DIR / 'fhv_trip_features.parquet'}"
    )

    print(
        f"\nHourly demand:"
        f"\n  {PROCESSED_DIR / 'fhv_hourly_demand.parquet'}"
    )

    print(
        f"\nZone hourly demand:"
        f"\n  {PROCESSED_DIR / 'fhv_zone_hourly_demand.parquet'}"
    )

    con.close()


if __name__ == "__main__":
    main()