"""
Diagnostic for Yellow Taxi and FHV duration errors on SAME-ZONE trips.

A same-zone trip is:
    PULocationID == DOLocationID

This script uses held-out prediction files if they exist. It does NOT
retrain a model and does not use training predictions.

Run:
    python src/diagnose_same_zone_error.py
"""

from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"

CANDIDATES = {
    "Yellow Taxi": {
        "files": [
            OUTPUTS_DIR / "yellow_duration_predictions.parquet",
            OUTPUTS_DIR / "yellow_duration_predictions.csv",
            OUTPUTS_DIR / "duration" / "yellow_duration_predictions.parquet",
            OUTPUTS_DIR / "duration" / "yellow_duration_predictions.csv",
        ],
        "distance_cols": ["trip_distance", "distance"],
    },
    "FHV": {
        "files": [
            OUTPUTS_DIR / "fhv_duration_predictions.parquet",
            OUTPUTS_DIR / "fhv_duration_predictions.csv",
            OUTPUTS_DIR / "duration" / "fhv_duration_predictions.parquet",
            OUTPUTS_DIR / "duration" / "fhv_duration_predictions.csv",
        ],
        "distance_cols": ["trip_miles", "trip_distance", "distance"],
    },
}


def find_file(vehicle):
    for path in CANDIDATES[vehicle]["files"]:
        if path.exists():
            return path
    return None


def read_columns(path):
    if path.suffix.lower() == ".csv":
        return duckdb.sql(
            f"DESCRIBE SELECT * FROM read_csv_auto('{path.as_posix()}')"
        ).df()["column_name"].tolist()

    return duckdb.sql(
        f"DESCRIBE SELECT * FROM read_parquet('{path.as_posix()}')"
    ).df()["column_name"].tolist()


def source_sql(path):
    if path.suffix.lower() == ".csv":
        return f"read_csv_auto('{path.as_posix()}')"
    return f"read_parquet('{path.as_posix()}')"


def pick_column(columns, candidates):
    lower = {c.lower(): c for c in columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def main():
    con = duckdb.connect()
    all_found = False

    print("=" * 80)
    print("SAME-ZONE DURATION ERROR DIAGNOSTIC")
    print("YELLOW TAXI + FHV")
    print("=" * 80)

    for vehicle, cfg in CANDIDATES.items():
        print(f"\n{'-' * 80}")
        print(vehicle)
        print(f"{'-' * 80}")

        path = find_file(vehicle)

        if not path:
            print("  No held-out duration prediction file found.")
            print("  Expected one of the configured prediction-file locations.")
            continue

        all_found = True
        print(f"  Prediction file: {path}")

        columns = read_columns(path)

        actual = pick_column(columns, ["actual", "actual_duration_s", "y_true"])
        predicted = pick_column(
            columns, ["predicted", "predicted_duration_s", "prediction", "y_pred"]
        )
        pu = pick_column(columns, ["PULocationID", "PUlocationID"])
        do = pick_column(columns, ["DOLocationID", "DOlocationID"])
        distance = pick_column(columns, cfg["distance_cols"])

        required = {
            "actual": actual,
            "predicted": predicted,
            "pickup zone": pu,
            "dropoff zone": do,
        }

        missing = [name for name, col in required.items() if col is None]
        if missing:
            print(f"  Cannot run diagnostic; missing columns: {missing}")
            print(f"  Available columns: {columns}")
            continue

        source = source_sql(path)

        result = con.execute(
            f"""
            SELECT
                CASE
                    WHEN "{pu}" = "{do}" THEN 'Same zone'
                    ELSE 'Different zones'
                END AS trip_kind,
                count(*) AS n_trips,
                avg(abs("{actual}" - "{predicted}")) AS mae_s,
                avg(
                    abs("{actual}" - "{predicted}")
                    / NULLIF("{actual}", 0)
                ) * 100 AS mape_pct,
                avg("{actual}") / 60.0 AS avg_actual_duration_min,
                avg("{predicted}") / 60.0 AS avg_predicted_duration_min
            FROM {source}
            GROUP BY 1
            ORDER BY 1
            """
        ).fetchdf()

        print("\nOverall comparison:")
        print(result.to_string(index=False))

        distance_filter = ""
        distance_label = ""

        if distance:
            distance_filter = f'AND "{distance}" < 1.0'
            distance_label = f" AND {distance} < 1 mile"

        short_same_zone = con.execute(
            f"""
            SELECT
                count(*) AS n_trips,
                avg(abs("{actual}" - "{predicted}")) AS mae_s,
                avg(
                    abs("{actual}" - "{predicted}")
                    / NULLIF("{actual}", 0)
                ) * 100 AS mape_pct,
                avg("{actual}") / 60.0 AS avg_actual_duration_min,
                avg("{predicted}") / 60.0 AS avg_predicted_duration_min,
                min("{actual}") / 60.0 AS min_actual_duration_min,
                max("{actual}") / 60.0 AS max_actual_duration_min
            FROM {source}
            WHERE "{pu}" = "{do}"
              {distance_filter}
            """
        ).fetchdf()

        print(f"\nSame-zone{distance_label}:")
        print(short_same_zone.to_string(index=False))

        by_distance = ""
        if distance:
            by_distance = f"""
            SELECT
                CASE
                    WHEN "{distance}" < 0.5 THEN '<0.5 mi'
                    WHEN "{distance}" < 1.0 THEN '0.5-<1 mi'
                    WHEN "{distance}" < 2.0 THEN '1-<2 mi'
                    WHEN "{distance}" < 5.0 THEN '2-<5 mi'
                    ELSE '5+ mi'
                END AS distance_bucket,
                count(*) AS n_trips,
                avg(abs("{actual}" - "{predicted}")) AS mae_s,
                avg("{actual}") / 60.0 AS avg_actual_min,
                avg("{predicted}") / 60.0 AS avg_predicted_min
            FROM {source}
            WHERE "{pu}" = "{do}"
            GROUP BY 1
            ORDER BY
                CASE distance_bucket
                    WHEN '<0.5 mi' THEN 1
                    WHEN '0.5-<1 mi' THEN 2
                    WHEN '1-<2 mi' THEN 3
                    WHEN '2-<5 mi' THEN 4
                    ELSE 5
                END
            """

            print("\nSame-zone error by distance bucket:")
            print(con.execute(by_distance).fetchdf().to_string(index=False))

    print("\n" + "=" * 80)
    if not all_found:
        print("No usable held-out prediction file was found for at least one vehicle.")
        print("This diagnostic should only be interpreted when the real held-out")
        print("prediction files are available.")
    else:
        print("SAME-ZONE DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
