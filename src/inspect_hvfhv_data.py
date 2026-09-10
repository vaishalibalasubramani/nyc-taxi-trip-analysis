"""
Quick inspection of HVFHV (Uber/Lyft) trip data BEFORE committing to a full
12-month build. Reads only the schema and a handful of sample rows over
HTTPS -- fast, cheap, and catches schema surprises early (HVFHV's columns
have changed across years, e.g. congestion_surcharge/airport_fee were added
after the format's initial release).

Usage:
    python src/inspect_hvfhv_data.py --month 2025-01
"""

import argparse

import duckdb

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"


def main():
    parser = argparse.ArgumentParser(description="Inspect one month of HVFHV data")
    parser.add_argument("--month", default="2025-01", help="YYYY-MM to inspect")
    args = parser.parse_args()

    url = f"{BASE_URL}/fhvhv_tripdata_{args.month}.parquet"
    print(f"Inspecting: {url}\n")

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")

    print("--- Schema ---")
    schema = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{url}')").fetchdf()
    print(schema.to_string(index=False))

    expected = {
        "hvfhs_license_num", "pickup_datetime", "dropoff_datetime",
        "PULocationID", "DOLocationID", "trip_miles", "trip_time",
        "shared_request_flag", "wav_request_flag",
    }
    actual = set(schema["column_name"])
    missing = expected - actual
    if missing:
        print(f"\n*** WARNING: expected columns not found: {missing} ***")
        print("The build script's column references will need updating.")
    else:
        print("\nAll columns build_features_hvfhv.py depends on are present. Good to go.")

    print("\n--- Row count (this month) ---")
    n = con.execute(f"SELECT count(*) FROM read_parquet('{url}')").fetchone()[0]
    print(f"{n:,} trips")

    print("\n--- Sample rows ---")
    sample = con.execute(
        f"""
        SELECT hvfhs_license_num, pickup_datetime, dropoff_datetime,
               PULocationID, DOLocationID, trip_miles, trip_time
        FROM read_parquet('{url}')
        LIMIT 5
        """
    ).fetchdf()
    print(sample.to_string(index=False))

    print("\n--- hvfhs_license_num breakdown (which companies) ---")
    breakdown = con.execute(
        f"""
        SELECT hvfhs_license_num, count(*) AS trips
        FROM read_parquet('{url}')
        GROUP BY 1 ORDER BY 2 DESC
        """
    ).fetchdf()
    print(breakdown.to_string(index=False))
    print(
        "\n(HV0002=Juno, HV0003=Uber, HV0004=Via, HV0005=Lyft -- codes per TLC's data dictionary)"
    )


if __name__ == "__main__":
    main()
