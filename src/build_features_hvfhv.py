"""
Build feature tables for High Volume For-Hire Vehicle (HVFHV -- Uber, Lyft,
Via, Juno) trip data, streamed directly over HTTPS via DuckDB httpfs.

Processes ONE MONTH AT A TIME with retries, AND if a month still fails after
retries, it's skipped (not fatal) and tried again in a final pass after all
other months finish -- CloudFront's transient errors often clear given more
time, and this way one stubborn file doesn't lose an otherwise-successful run.

Usage:
    python src/build_features_hvfhv.py --months 2025-01 2025-02 ... 2025-12
    python src/build_features_hvfhv.py --months 2025-01 2025-02 --sample-rows 1000000
"""

import argparse
import time
from pathlib import Path

import duckdb

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 15  # doubles each retry: 15, 30, 60, 120


def run_with_retries(con, sql, description):
    """Execute a DuckDB statement, retrying on transient HTTP errors.
    Raises the final exception if all retries are exhausted."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return con.execute(sql)
        except duckdb.Error as e:
            if attempt == MAX_RETRIES:
                print(f"    still failing after {MAX_RETRIES} attempts: {description}")
                raise
            wait = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
            print(f"    HTTP error on attempt {attempt}/{MAX_RETRIES}: {e}")
            print(f"    Retrying in {wait}s...")
            time.sleep(wait)


def process_month(con, month, rows_per_month):
    """Build this month's chunk of all three feature tables. Raises on failure."""
    url = f"{BASE_URL}/fhvhv_tripdata_{month}.parquet"

    run_with_retries(
        con,
        f"""
        CREATE OR REPLACE VIEW raw_trips_month AS
        SELECT
            hvfhs_license_num, pickup_datetime, dropoff_datetime,
            PULocationID, DOLocationID, trip_miles,
            trip_time AS trip_duration_s,
            shared_request_flag, wav_request_flag
        FROM read_parquet('{url}')
        WHERE pickup_datetime IS NOT NULL AND dropoff_datetime IS NOT NULL
          AND trip_time BETWEEN 60 AND 10800
          AND trip_miles > 0 AND trip_miles < 100
          AND PULocationID IS NOT NULL AND DOLocationID IS NOT NULL
          AND pickup_datetime >= DATE '2025-01-01'
          AND pickup_datetime <  DATE '2026-01-01'
        """,
        f"raw_trips_month view ({month})",
    )

    run_with_retries(
        con,
        f"""
        CREATE OR REPLACE TABLE trip_features_month AS
        SELECT
            trip_duration_s, trip_miles, PULocationID, DOLocationID, hvfhs_license_num,
            CAST(shared_request_flag = 'Y' AS INTEGER) AS is_shared_request,
            CAST(wav_request_flag = 'Y' AS INTEGER)    AS is_wav_request,
            hour(pickup_datetime)                       AS pickup_hour,
            dayofweek(pickup_datetime)                  AS pickup_dow,
            CASE WHEN dayofweek(pickup_datetime) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,
            month(pickup_datetime)                       AS pickup_month
        FROM (SELECT * FROM raw_trips_month USING SAMPLE {rows_per_month} (reservoir))
        """,
        f"trip_features_month ({month})",
    )

    run_with_retries(
        con,
        """
        CREATE OR REPLACE TABLE hourly_demand_month AS
        SELECT date_trunc('hour', pickup_datetime) AS pickup_hour_ts, count(*) AS trip_count
        FROM raw_trips_month
        GROUP BY 1
        """,
        f"hourly_demand_month ({month})",
    )

    run_with_retries(
        con,
        f"""
        CREATE OR REPLACE TABLE zone_hourly_demand_month AS
        SELECT
            date_trunc('hour', r.pickup_datetime) AS pickup_hour_ts,
            r.PULocationID                        AS zone_id,
            z.Zone                                AS zone_name,
            z.Borough                             AS borough,
            count(*)                              AS trip_count
        FROM raw_trips_month r
        LEFT JOIN read_csv_auto('{ZONE_LOOKUP_URL}') z
            ON r.PULocationID = z.LocationID
        GROUP BY 1, 2, 3, 4
        """,
        f"zone_hourly_demand_month ({month})",
    )


def append_month_results(con, first_chunk):
    if first_chunk:
        con.execute("CREATE OR REPLACE TABLE trip_features_all AS SELECT * FROM trip_features_month")
        con.execute("CREATE OR REPLACE TABLE hourly_demand_all AS SELECT * FROM hourly_demand_month")
        con.execute("CREATE OR REPLACE TABLE zone_hourly_demand_all AS SELECT * FROM zone_hourly_demand_month")
    else:
        con.execute("INSERT INTO trip_features_all SELECT * FROM trip_features_month")
        con.execute("INSERT INTO hourly_demand_all SELECT * FROM hourly_demand_month")
        con.execute("INSERT INTO zone_hourly_demand_all SELECT * FROM zone_hourly_demand_month")


def main():
    parser = argparse.ArgumentParser(description="Build feature tables from remote HVFHV data")
    parser.add_argument(
        "--months", nargs="+", required=True,
        help="One or more YYYY-MM values, e.g. 2025-01 2025-02 ... 2025-12",
    )
    parser.add_argument(
        "--sample-rows", type=int, default=2_000_000,
        help="Total row-level sample size for Task 1 across all months combined "
             "(default 2,000,000, split evenly per month). Tasks 2/3 always use full data.",
    )
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")

    rows_per_month = max(1, args.sample_rows // len(args.months))
    print(f"Processing {len(args.months)} months sequentially, "
          f"~{rows_per_month:,} sampled rows/month for Task 1.\n")

    trip_features_path = PROCESSED_DIR / "hvfhv_trip_features.parquet"
    hourly_demand_path = PROCESSED_DIR / "hvfhv_hourly_demand.parquet"
    zone_hourly_demand_path = PROCESSED_DIR / "hvfhv_zone_hourly_demand.parquet"

    first_chunk = True
    failed_months = []
    succeeded_months = []

    def try_month(month, idx, total, label):
        nonlocal first_chunk
        print(f"[{label}] {month}")
        try:
            process_month(con, month, rows_per_month)
            append_month_results(con, first_chunk)
            first_chunk = False
            succeeded_months.append(month)
            n = con.execute("SELECT count(*) FROM trip_features_all").fetchone()[0]
            print(f"    done. Running total trip_features rows: {n:,}\n")
            return True
        except duckdb.Error:
            print(f"    SKIPPING {month} for now -- will retry after the other months.\n")
            failed_months.append(month)
            return False

    # --- First pass: all requested months, in order ---
    for i, month in enumerate(args.months, 1):
        try_month(month, i, len(args.months), f"{i}/{len(args.months)}")

    # --- Second pass: retry any months that failed, once each ---
    if failed_months:
        print(f"\n=== Retrying {len(failed_months)} month(s) that failed on the first pass: {failed_months} ===\n")
        still_failed = []
        for i, month in enumerate(list(failed_months), 1):
            ok = try_month(month, i, len(failed_months), f"retry {i}/{len(failed_months)}")
            if not ok:
                still_failed.append(month)
        failed_months = still_failed

    if not succeeded_months:
        raise SystemExit("No months succeeded -- nothing to write. Check your connection and try again.")

    # ---------------------------------------------------------------
    # Finalize: aggregate hourly/zone tables (in case of any duplicate
    # month insertion) and write everything out
    # ---------------------------------------------------------------
    print("Finalizing aggregates...")
    con.execute(
        """
        CREATE OR REPLACE TABLE hourly_demand_final AS
        SELECT pickup_hour_ts, sum(trip_count) AS trip_count
        FROM hourly_demand_all
        GROUP BY 1
        ORDER BY 1
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE zone_hourly_demand_final AS
        SELECT pickup_hour_ts, zone_id, zone_name, borough, sum(trip_count) AS trip_count
        FROM zone_hourly_demand_all
        GROUP BY 1, 2, 3, 4
        ORDER BY 1, 2
        """
    )

    con.execute(f"COPY trip_features_all TO '{trip_features_path}' (FORMAT PARQUET)")
    con.execute(f"COPY hourly_demand_final TO '{hourly_demand_path}' (FORMAT PARQUET)")
    con.execute(f"COPY zone_hourly_demand_final TO '{zone_hourly_demand_path}' (FORMAT PARQUET)")

    print(f"\nDone.")
    print(f"  Succeeded months: {succeeded_months}")
    if failed_months:
        print(f"  STILL FAILED (excluded from output, rerun these specifically later): {failed_months}")
        print(f"  e.g. python src/build_features_hvfhv.py --months {' '.join(failed_months)}")
    print(f"\n  trip_features:      {con.execute('SELECT count(*) FROM trip_features_all').fetchone()[0]:,} rows -> {trip_features_path}")
    print(f"  hourly_demand:      {con.execute('SELECT count(*) FROM hourly_demand_final').fetchone()[0]:,} rows -> {hourly_demand_path}")
    print(f"  zone_hourly_demand: {con.execute('SELECT count(*) FROM zone_hourly_demand_final').fetchone()[0]:,} rows -> {zone_hourly_demand_path}")


if __name__ == "__main__":
    main()