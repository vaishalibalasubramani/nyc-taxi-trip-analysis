"""
Diagnostic: does the duration model perform worse specifically on
same-zone (pickup == dropoff) trips vs. cross-zone trips? Uses the REAL
held-out test-set predictions saved by train_duration_task1.py /
train_duration_hvfhv.py (actual, predicted, PULocationID, DOLocationID
columns) -- not a guess, actual measured error on data the model never
trained on.

Usage:
    python src/diagnose_same_zone_error.py
"""

from pathlib import Path

import duckdb

OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"

FILES = {
    "Green Taxi": (OUTPUTS_DIR / "duration_predictions.parquet", "trip_distance"),
    "HVFHV": (OUTPUTS_DIR / "hvfhv_duration_predictions.parquet", "trip_miles"),
}


def main():
    con = duckdb.connect()
    for label, (path, distance_col) in FILES.items():
        if not path.exists():
            print(f"[{label}] No predictions file found at {path} -- skipping.")
            continue

        print(f"\n=== {label} ===")
        result = con.execute(
            f"""
            SELECT
                CASE WHEN PULocationID = DOLocationID THEN 'Same zone' ELSE 'Different zones' END AS trip_kind,
                count(*) AS n_trips,
                avg(abs(actual - predicted)) AS mae,
                avg(abs(actual - predicted) / NULLIF(actual, 0)) * 100 AS mape_pct,
                avg(actual) AS avg_actual_duration_s,
                avg(actual / 60.0) AS avg_actual_duration_min
            FROM read_parquet('{path.as_posix()}')
            GROUP BY 1
            ORDER BY 1
            """
        ).fetchdf()
        print(result.to_string(index=False))

        # Also break down same-zone specifically by distance bucket, since
        # short same-zone trips are the exact scenario that looked odd
        # (very short distance, unusually long predicted duration).
        short_same_zone = con.execute(
            f"""
            SELECT
                count(*) AS n_trips,
                avg(abs(actual - predicted)) AS mae,
                avg(abs(actual - predicted) / NULLIF(actual, 0)) * 100 AS mape_pct,
                avg(actual / 60.0) AS avg_actual_duration_min,
                min(actual / 60.0) AS min_actual_duration_min,
                max(actual / 60.0) AS max_actual_duration_min
            FROM read_parquet('{path.as_posix()}')
            WHERE PULocationID = DOLocationID AND {distance_col} < 1.0
            """
        ).fetchdf()
        print(f"\nSame-zone AND {distance_col} < 1 mile specifically:")
        print(short_same_zone.to_string(index=False))


if __name__ == "__main__":
    main()