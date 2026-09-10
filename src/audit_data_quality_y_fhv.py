"""
Independent data-quality audit for the FINAL project scope:
    - Yellow Taxi
    - FHV

Important:
    This audit is defensive. Some FHV processed features do not contain a
    distance column, so the distance check is skipped with INFO instead of
    crashing. Also, zone referential integrity is checked against an official
    taxi-zone lookup when one is available; it is NOT checked against
    zone_hourly_demand because that file contains only zones with recorded
    activity and therefore is not a complete zone reference.

Run:
    python src/audit_data_quality.py
"""

from pathlib import Path
import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

MIN_DURATION_S = 60
MAX_DURATION_S = 10800
MIN_DISTANCE = 0
MAX_DISTANCE = 100
VALID_YEAR = 2025
TLC_ZONE_ID_RANGE = (1, 265)


CONFIGS = {
    "Yellow Taxi": {
        "trip_features": PROCESSED_DIR / "yellow_trip_features.parquet",
        "hourly_demand": PROCESSED_DIR / "hourly_demand.parquet",
        "zone_hourly_demand": PROCESSED_DIR / "zone_hourly_demand.parquet",
        "distance_candidates": ["trip_distance", "distance"],
        "duration_col": "trip_duration_s",
    },
    "FHV": {
        "trip_features": PROCESSED_DIR / "fhv_trip_features.parquet",
        "hourly_demand": PROCESSED_DIR / "fhv_hourly_demand.parquet",
        "zone_hourly_demand": PROCESSED_DIR / "fhv_zone_hourly_demand.parquet",
        # Current FHV engineered file does not contain trip_miles.
        "distance_candidates": ["trip_miles", "trip_distance", "distance"],
        "duration_col": "trip_duration_s",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title):
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def check(label, condition, detail=""):
    status = "PASS" if condition else "FLAG"
    print(f"  [{status}] {label}" + (f" -- {detail}" if detail else ""))
    return condition


def info(label, detail=""):
    print(f"  [INFO] {label}" + (f" -- {detail}" if detail else ""))


def get_columns(path):
    return duckdb.sql(
        f"DESCRIBE SELECT * FROM read_parquet('{path.as_posix()}')"
    ).df()["column_name"].tolist()


def find_column(columns, candidates):
    lookup = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def find_zone_lookup():
    """
    Try common project locations for the TLC taxi-zone lookup.

    The audit can still run without it. In that case zone IDs are checked
    against the plausible TLC range, and referential integrity is reported
    as INFO rather than incorrectly comparing against only active zones.
    """
    candidates = [
        ROOT / "data" / "taxi_zone_lookup.csv",
        ROOT / "data" / "taxi+_zone_lookup.csv",
        ROOT / "data" / "reference" / "taxi_zone_lookup.csv",
        ROOT / "data" / "raw" / "taxi_zone_lookup.csv",
        ROOT / "data" / "processed" / "taxi_zone_lookup.csv",
        ROOT / "taxi_zone_lookup.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def source_sql(path):
    return f"read_parquet('{path.as_posix()}')"


# ---------------------------------------------------------------------------
# Trip features
# ---------------------------------------------------------------------------

def audit_trip_features(vehicle, cfg, flags):
    path = cfg["trip_features"]

    section(f"[{vehicle}] trip_features.parquet")

    if not path.exists():
        print(f"  [SKIP] File not found: {path}")
        flags.append(f"{vehicle}: missing {path.name}")
        return

    columns = get_columns(path)

    n_total = duckdb.sql(
        f"SELECT count(*) FROM {source_sql(path)}"
    ).fetchone()[0]

    print(f"  Total rows: {n_total:,}")

    # ---------------------------------------------------------------
    # NULLs
    # ---------------------------------------------------------------

    null_counts = duckdb.sql(
        f"""
        SELECT
            {", ".join(
                f'sum(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END) AS "{c}"'
                for c in columns
            )}
        FROM {source_sql(path)}
        """
    ).df().iloc[0]

    null_columns = []
    for c, n in null_counts.items():
        if n and int(n) > 0:
            null_columns.append((c, int(n)))
            print(f"    NULL {c}: {int(n):,}")

    # Affiliated_base_number is an optional/source identifier in FHV.
    # Do not treat a small number of missing values in this field as a
    # model-data failure.
    optional_nulls = {"Affiliated_base_number"}

    critical_nulls = [
        (c, n) for c, n in null_columns if c not in optional_nulls
    ]

    if critical_nulls:
        if not check(
            "No NULLs in critical/model columns",
            False,
            f"{sum(n for _, n in critical_nulls):,} NULL values",
        ):
            flags.append(
                f"{vehicle}: NULLs in critical/model columns: "
                + ", ".join(f"{c}={n:,}" for c, n in critical_nulls)
            )
    else:
        check("No NULLs in critical/model columns", True)

    for c, n in null_columns:
        if c in optional_nulls:
            info(
                f"Optional field {c} contains {n:,} NULL values",
                "This field is not required by the current FHV duration/demand models.",
            )

    # ---------------------------------------------------------------
    # Duration
    # ---------------------------------------------------------------

    dur_col = cfg["duration_col"]

    if dur_col not in columns:
        check(
            f"Required duration column '{dur_col}' exists",
            False,
        )
        flags.append(f"{vehicle}: missing {dur_col}")
    else:
        dur_stats = duckdb.sql(
            f"""
            SELECT
                min("{dur_col}"),
                max("{dur_col}"),
                sum(CASE WHEN "{dur_col}" < {MIN_DURATION_S} THEN 1 ELSE 0 END),
                sum(CASE WHEN "{dur_col}" > {MAX_DURATION_S} THEN 1 ELSE 0 END)
            FROM {source_sql(path)}
            """
        ).fetchone()

        print(
            f"  Duration range: "
            f"{dur_stats[0]:.0f}s to {dur_stats[1]:.0f}s"
        )

        if not check(
            f"Duration within [{MIN_DURATION_S}, {MAX_DURATION_S}] seconds",
            dur_stats[2] == 0 and dur_stats[3] == 0,
            f"{dur_stats[2]} below min, {dur_stats[3]} above max",
        ):
            flags.append(f"{vehicle}: duration filter-bound violations")

    # ---------------------------------------------------------------
    # Distance -- optional because current FHV engineered data may not
    # contain a distance field.
    # ---------------------------------------------------------------

    dist_col = find_column(columns, cfg["distance_candidates"])

    if dist_col is None:
        info(
            "Distance check skipped",
            f"No distance column found. Available candidates: "
            f"{cfg['distance_candidates']}",
        )
    else:
        dist_stats = duckdb.sql(
            f"""
            SELECT
                min("{dist_col}"),
                max("{dist_col}"),
                sum(CASE WHEN "{dist_col}" <= {MIN_DISTANCE} THEN 1 ELSE 0 END),
                sum(CASE WHEN "{dist_col}" > {MAX_DISTANCE} THEN 1 ELSE 0 END)
            FROM {source_sql(path)}
            """
        ).fetchone()

        print(
            f"  Distance column: {dist_col}"
        )
        print(
            f"  Distance range: "
            f"{dist_stats[0]:.2f} to {dist_stats[1]:.2f} miles"
        )

        if not check(
            f"Distance within (0, {MAX_DISTANCE}] miles",
            dist_stats[2] == 0 and dist_stats[3] == 0,
            f"{dist_stats[2]} zero/negative, "
            f"{dist_stats[3]} above max",
        ):
            flags.append(f"{vehicle}: distance filter-bound violations")

    # ---------------------------------------------------------------
    # Calendar features
    # ---------------------------------------------------------------

    required_calendar = [
        "pickup_hour",
        "pickup_dow",
        "pickup_month",
        "is_weekend",
    ]

    missing_calendar = [
        c for c in required_calendar if c not in columns
    ]

    if missing_calendar:
        info(
            "Calendar range checks partially skipped",
            f"Missing columns: {missing_calendar}",
        )
    else:
        cal = duckdb.sql(
            f"""
            SELECT
                min(pickup_hour), max(pickup_hour),
                min(pickup_dow), max(pickup_dow),
                min(pickup_month), max(pickup_month),
                sum(CASE WHEN is_weekend NOT IN (0,1) THEN 1 ELSE 0 END)
            FROM {source_sql(path)}
            """
        ).fetchone()

        if not check(
            "pickup_hour within [0,23]",
            0 <= cal[0] and cal[1] <= 23,
            f"observed [{cal[0]}, {cal[1]}]",
        ):
            flags.append(f"{vehicle}: invalid pickup_hour")

        if not check(
            "pickup_dow within [0,6]",
            0 <= cal[2] and cal[3] <= 6,
            f"observed [{cal[2]}, {cal[3]}]",
        ):
            flags.append(f"{vehicle}: invalid pickup_dow")

        if not check(
            "pickup_month within [1,12]",
            1 <= cal[4] and cal[5] <= 12,
            f"observed [{cal[4]}, {cal[5]}]",
        ):
            flags.append(f"{vehicle}: invalid pickup_month")

        if not check(
            "is_weekend only 0 or 1",
            cal[6] == 0,
            f"{cal[6]} invalid values" if cal[6] else "",
        ):
            flags.append(f"{vehicle}: invalid is_weekend values")

    # ---------------------------------------------------------------
    # Zone IDs
    # ---------------------------------------------------------------

    if "PULocationID" in columns and "DOLocationID" in columns:
        zone_range = duckdb.sql(
            f"""
            SELECT
                min(PULocationID), max(PULocationID),
                min(DOLocationID), max(DOLocationID)
            FROM {source_sql(path)}
            """
        ).fetchone()

        if not check(
            f"Zone IDs within plausible TLC range {TLC_ZONE_ID_RANGE}",
            TLC_ZONE_ID_RANGE[0] <= zone_range[0]
            and zone_range[1] <= TLC_ZONE_ID_RANGE[1]
            and TLC_ZONE_ID_RANGE[0] <= zone_range[2]
            and zone_range[3] <= TLC_ZONE_ID_RANGE[1],
            f"PU[{zone_range[0]},{zone_range[1]}] "
            f"DO[{zone_range[2]},{zone_range[3]}]",
        ):
            flags.append(f"{vehicle}: zone IDs outside plausible TLC range")

        lookup_path = find_zone_lookup()

        if lookup_path:
            info(
                "Official zone lookup found",
                str(lookup_path),
            )

            try:
                lookup_cols = duckdb.sql(
                    f"DESCRIBE SELECT * FROM read_csv_auto('{lookup_path.as_posix()}')"
                ).df()["column_name"].tolist()

                location_col = find_column(
                    lookup_cols,
                    ["LocationID", "location_id", "zone_id"],
                )

                if location_col:
                    orphans = duckdb.sql(
                        f"""
                        WITH known_zones AS (
                            SELECT DISTINCT "{location_col}" AS zone_id
                            FROM read_csv_auto('{lookup_path.as_posix()}')
                        )
                        SELECT count(*)
                        FROM {source_sql(path)} t
                        WHERE t.PULocationID NOT IN (
                            SELECT zone_id FROM known_zones
                        )
                        OR t.DOLocationID NOT IN (
                            SELECT zone_id FROM known_zones
                        )
                        """
                    ).fetchone()[0]

                    if not check(
                        "All pickup/dropoff zone IDs exist in official TLC lookup",
                        orphans == 0,
                        f"{orphans:,} rows reference an unknown zone",
                    ):
                        flags.append(
                            f"{vehicle}: {orphans:,} rows have zone IDs "
                            "not present in official TLC lookup"
                        )
                else:
                    info(
                        "Official zone referential-integrity check skipped",
                        "Could not identify LocationID column in lookup.",
                    )

            except Exception as exc:
                info(
                    "Official zone referential-integrity check skipped",
                    f"Lookup could not be read: {exc}",
                )
        else:
            info(
                "Official zone referential-integrity check skipped",
                "No taxi_zone_lookup.csv found in the common project locations.",
            )

    # ---------------------------------------------------------------
    # Exact duplicate engineered rows
    # ---------------------------------------------------------------

    n_distinct = duckdb.sql(
        f"""
        SELECT count(*)
        FROM (
            SELECT DISTINCT *
            FROM {source_sql(path)}
        )
        """
    ).fetchone()[0]

    n_dupes = n_total - n_distinct

    info(
        f"Exact duplicate rows: {n_dupes:,} "
        f"({n_dupes / n_total * 100:.2f}%)",
        "Informational only: identical engineered feature rows can legitimately "
        "occur when different trips share the same engineered values.",
    )


# ---------------------------------------------------------------------------
# Citywide hourly demand
# ---------------------------------------------------------------------------

def audit_hourly_demand(vehicle, cfg, flags):
    path = cfg["hourly_demand"]

    section(f"[{vehicle}] hourly_demand.parquet")

    if not path.exists():
        print(f"  [SKIP] File not found: {path}")
        flags.append(f"{vehicle}: missing {path.name}")
        return

    stats = duckdb.sql(
        f"""
        SELECT
            count(*),
            min(pickup_hour_ts),
            max(pickup_hour_ts),
            sum(CASE WHEN trip_count IS NULL THEN 1 ELSE 0 END),
            sum(CASE WHEN trip_count < 0 THEN 1 ELSE 0 END),
            count(*) - count(DISTINCT pickup_hour_ts)
        FROM {source_sql(path)}
        """
    ).fetchone()

    n, min_ts, max_ts, n_null, n_neg, n_dupe = stats

    print(f"  Total rows: {n:,}")
    print(f"  Date range: {min_ts} to {max_ts}")

    if not check(
        "No NULL trip_count",
        n_null == 0,
        f"{n_null} null rows" if n_null else "",
    ):
        flags.append(f"{vehicle}: NULL citywide trip_count")

    if not check(
        "No negative trip_count",
        n_neg == 0,
        f"{n_neg} negative rows" if n_neg else "",
    ):
        flags.append(f"{vehicle}: negative citywide trip_count")

    if not check(
        "No duplicate pickup_hour_ts",
        n_dupe == 0,
        f"{n_dupe} duplicate timestamps" if n_dupe else "",
    ):
        flags.append(f"{vehicle}: duplicate citywide timestamps")

    if not check(
        f"Data year is {VALID_YEAR}",
        pd.Timestamp(min_ts).year == VALID_YEAR
        and pd.Timestamp(max_ts).year == VALID_YEAR,
        f"observed {min_ts} to {max_ts}",
    ):
        flags.append(f"{vehicle}: citywide demand outside {VALID_YEAR}")

    expected_hours = (
        int(
            (
                pd.Timestamp(max_ts) - pd.Timestamp(min_ts)
            ).total_seconds()
            / 3600
        )
        + 1
    )

    missing_hours = expected_hours - n

    info(
        f"Hours with no stored row: {missing_hours:,} "
        f"of {expected_hours:,}",
        "Expected for GROUP BY output when an hour has zero trips; "
        "the forecasting feature builder reindexes and fills these with 0.",
    )


# ---------------------------------------------------------------------------
# Zone hourly demand
# ---------------------------------------------------------------------------

def audit_zone_hourly_demand(vehicle, cfg, flags):
    path = cfg["zone_hourly_demand"]

    section(f"[{vehicle}] zone_hourly_demand.parquet")

    if not path.exists():
        print(f"  [SKIP] File not found: {path}")
        flags.append(f"{vehicle}: missing {path.name}")
        return

    stats = duckdb.sql(
        f"""
        SELECT
            count(*),
            min(pickup_hour_ts),
            max(pickup_hour_ts),
            count(DISTINCT zone_id),
            sum(CASE WHEN zone_name IS NULL THEN 1 ELSE 0 END),
            sum(CASE WHEN borough IS NULL THEN 1 ELSE 0 END),
            sum(CASE WHEN trip_count IS NULL THEN 1 ELSE 0 END),
            sum(CASE WHEN trip_count < 0 THEN 1 ELSE 0 END),
            count(*) - count(DISTINCT (pickup_hour_ts, zone_id))
        FROM {source_sql(path)}
        """
    ).fetchone()

    (
        n,
        min_ts,
        max_ts,
        n_zones,
        n_null_name,
        n_null_borough,
        n_null_count,
        n_neg,
        n_dupe,
    ) = stats

    print(f"  Total rows: {n:,}")
    print(f"  Date range: {min_ts} to {max_ts}")
    print(f"  Distinct zones covered: {n_zones}")

    if not check(
        "No unmatched zone_name",
        n_null_name == 0,
        f"{n_null_name:,} rows" if n_null_name else "",
    ):
        flags.append(f"{vehicle}: unmatched zone_name values")

    if not check(
        "No unmatched borough",
        n_null_borough == 0,
        f"{n_null_borough:,} rows" if n_null_borough else "",
    ):
        flags.append(f"{vehicle}: unmatched borough values")

    if not check(
        "No NULL trip_count",
        n_null_count == 0,
        f"{n_null_count:,} rows" if n_null_count else "",
    ):
        flags.append(f"{vehicle}: NULL zone trip_count")

    if not check(
        "No negative trip_count",
        n_neg == 0,
        f"{n_neg:,} rows" if n_neg else "",
    ):
        flags.append(f"{vehicle}: negative zone trip_count")

    if not check(
        "No duplicate (pickup_hour_ts, zone_id) pairs",
        n_dupe == 0,
        f"{n_dupe:,} duplicate pairs" if n_dupe else "",
    ):
        flags.append(f"{vehicle}: duplicate hour-zone pairs")

    zone_range = duckdb.sql(
        f"""
        SELECT min(zone_id), max(zone_id)
        FROM {source_sql(path)}
        """
    ).fetchone()

    if not check(
        f"Zone IDs within plausible TLC range {TLC_ZONE_ID_RANGE}",
        TLC_ZONE_ID_RANGE[0] <= zone_range[0]
        and zone_range[1] <= TLC_ZONE_ID_RANGE[1],
        f"observed [{zone_range[0]}, {zone_range[1]}]",
    ):
        flags.append(f"{vehicle}: zone demand IDs outside TLC range")

    if not check(
        f"Data year is {VALID_YEAR}",
        pd.Timestamp(min_ts).year == VALID_YEAR
        and pd.Timestamp(max_ts).year == VALID_YEAR,
    ):
        flags.append(f"{vehicle}: zone demand outside {VALID_YEAR}")

    if n_zones < 200:
        info(
            f"Only {n_zones} distinct zones have recorded trips",
            "This can be legitimate because the file contains active "
            "zone-hour observations rather than every possible TLC zone.",
        )
    else:
        info(
            f"{n_zones} distinct zones have recorded trips",
            "Coverage is based on zones with recorded activity.",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    flags = []

    section("NYC TLC DATA QUALITY AUDIT")
    print("FINAL SCOPE: Yellow Taxi + FHV")
    print("Checks are performed directly on processed Parquet outputs.")

    for vehicle, cfg in CONFIGS.items():
        audit_trip_features(vehicle, cfg, flags)
        audit_hourly_demand(vehicle, cfg, flags)
        audit_zone_hourly_demand(vehicle, cfg, flags)

    section("SUMMARY")

    if flags:
        print(f"{len(flags)} item(s) flagged for review:\n")
        for flag in flags:
            print(f"  - {flag}")
    else:
        print("NO FLAGS RAISED.")
        print("All checked data-quality invariants passed.")

    report_path = ROOT / "outputs" / "data_quality_yellow_fhv_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("NYC TLC DATA QUALITY AUDIT\n")
        f.write("=" * 70 + "\n")
        f.write("Scope: Yellow Taxi + FHV\n\n")

        if flags:
            f.write(f"{len(flags)} item(s) flagged for review:\n")
            for flag in flags:
                f.write(f"  - {flag}\n")
        else:
            f.write("No flags raised. All checked invariants passed.\n")

    print(f"\nSaved -> {report_path}")


if __name__ == "__main__":
    main()
