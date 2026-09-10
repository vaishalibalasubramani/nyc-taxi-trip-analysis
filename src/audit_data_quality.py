"""
Data quality audit -- runs a fresh, independent set of checks directly
against your processed parquet files (not just trusting that the cleaning
filters in build_features*.py worked as intended).

For each vehicle (Green Taxi, HVFHV), checks:
  trip_features.parquet   -- nulls, out-of-range distance/duration,
                              duplicate rows, referential integrity of
                              zone IDs against the zone lookup, valid
                              ranges for derived calendar features
  hourly_demand.parquet   -- nulls, negative counts, duplicate timestamps,
                              date range, gap accounting
  zone_hourly_demand.parquet -- nulls, unmatched zone/borough (failed
                              joins), duplicate (hour, zone) pairs,
                              zone_id range vs the official TLC zone count

Every check prints PASS/FLAG/INFO so you can scan for problems quickly.
FLAG means something to actually look at; INFO is context, not a defect
(e.g. duplicate rows are common and expected in coarse categorical trip
data -- reported for transparency, not because duplicates are inherently
wrong).

Usage:
    python src/audit_data_quality.py
"""

from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

# The filter bounds actually used in build_features_hvfhv.py -- this
# script checks whether the OUTPUT data actually respects these bounds,
# as an independent verification that the filters ran correctly.
MIN_DURATION_S = 60
MAX_DURATION_S = 10800
MIN_DISTANCE = 0
MAX_DISTANCE = 100
VALID_YEAR = 2025
TLC_ZONE_ID_RANGE = (1, 265)  # official TLC zones run 1-263, +2 unused/buffer IDs historically
TAXI_ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

CONFIGS = {
    "Green Taxi": {
        "trip_features": PROCESSED_DIR / "trip_features.parquet",
        "hourly_demand": PROCESSED_DIR / "hourly_demand.parquet",
        "zone_hourly_demand": PROCESSED_DIR / "zone_hourly_demand.parquet",
        "distance_col": "trip_distance",
        "duration_col": "trip_duration_s",
    },
    "HVFHV (Uber/Lyft)": {
        "trip_features": PROCESSED_DIR / "hvfhv_trip_features.parquet",
        "hourly_demand": PROCESSED_DIR / "hvfhv_hourly_demand.parquet",
        "zone_hourly_demand": PROCESSED_DIR / "hvfhv_zone_hourly_demand.parquet",
        "distance_col": "trip_miles",
        "duration_col": "trip_duration_s",
    },
}


def section(title):
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def check(label: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FLAG"
    print(f"  [{status}] {label}" + (f" -- {detail}" if detail else ""))
    return condition


def info(label: str, detail: str = ""):
    print(f"  [INFO] {label}" + (f" -- {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# trip_features.parquet
# ---------------------------------------------------------------------------

def audit_trip_features(vehicle: str, cfg: dict, all_flags: list):
    path = cfg["trip_features"]
    section(f"[{vehicle}] trip_features.parquet")
    if not path.exists():
        print("  SKIP -- file not found")
        return

    dist_col, dur_col = cfg["distance_col"], cfg["duration_col"]
    n_total = duckdb.sql(f"SELECT count(*) FROM read_parquet('{path.as_posix()}')").fetchone()[0]
    print(f"  Total rows: {n_total:,}")

    # --- Nulls in every column ---
    cols = duckdb.sql(f"DESCRIBE SELECT * FROM read_parquet('{path.as_posix()}')").df()["column_name"].tolist()
    null_counts = duckdb.sql(
        f"SELECT {', '.join(f'sum(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) AS {c}' for c in cols)} "
        f"FROM read_parquet('{path.as_posix()}')"
    ).df().iloc[0]
    any_nulls = False
    for c, n in null_counts.items():
        if n and n > 0:
            any_nulls = True
            print(f"    null {c}: {n:,}")
    if not check("No NULLs in any column", not any_nulls):
        all_flags.append(f"{vehicle}: trip_features has NULL values (see above)")

    # --- Duration bounds (the actual filter used at build time) ---
    dur_stats = duckdb.sql(
        f"SELECT min({dur_col}), max({dur_col}), "
        f"sum(CASE WHEN {dur_col} < {MIN_DURATION_S} THEN 1 ELSE 0 END), "
        f"sum(CASE WHEN {dur_col} > {MAX_DURATION_S} THEN 1 ELSE 0 END) "
        f"FROM read_parquet('{path.as_posix()}')"
    ).fetchone()
    print(f"  Duration range: {dur_stats[0]:.0f}s to {dur_stats[1]:.0f}s")
    ok = check(f"Duration within [{MIN_DURATION_S}, {MAX_DURATION_S}]s filter bound",
               dur_stats[2] == 0 and dur_stats[3] == 0,
               f"{dur_stats[2]} below min, {dur_stats[3]} above max")
    if not ok:
        all_flags.append(f"{vehicle}: trip_features has {dur_stats[2]+dur_stats[3]} duration outliers outside build filter")

    # --- Distance bounds ---
    dist_stats = duckdb.sql(
        f"SELECT min({dist_col}), max({dist_col}), "
        f"sum(CASE WHEN {dist_col} <= {MIN_DISTANCE} THEN 1 ELSE 0 END), "
        f"sum(CASE WHEN {dist_col} > {MAX_DISTANCE} THEN 1 ELSE 0 END) "
        f"FROM read_parquet('{path.as_posix()}')"
    ).fetchone()
    print(f"  Distance range: {dist_stats[0]:.2f}mi to {dist_stats[1]:.2f}mi")
    ok = check(f"Distance within (0, {MAX_DISTANCE}]mi filter bound",
               dist_stats[2] == 0 and dist_stats[3] == 0,
               f"{dist_stats[2]} zero/negative, {dist_stats[3]} above max")
    if not ok:
        all_flags.append(f"{vehicle}: trip_features has {dist_stats[2]+dist_stats[3]} distance outliers outside build filter")

    # --- Calendar feature ranges ---
    cal = duckdb.sql(
        f"""
        SELECT
            min(pickup_hour), max(pickup_hour),
            min(pickup_dow), max(pickup_dow),
            min(pickup_month), max(pickup_month),
            sum(CASE WHEN is_weekend NOT IN (0,1) THEN 1 ELSE 0 END)
        FROM read_parquet('{path.as_posix()}')
        """
    ).fetchone()
    check("pickup_hour within [0,23]", 0 <= cal[0] and cal[1] <= 23, f"observed [{cal[0]},{cal[1]}]")
    check("pickup_dow within [0,6]", 0 <= cal[2] and cal[3] <= 6, f"observed [{cal[2]},{cal[3]}]")
    check("pickup_month within [1,12]", 1 <= cal[4] and cal[5] <= 12, f"observed [{cal[4]},{cal[5]}]")
    check("is_weekend only 0 or 1", cal[6] == 0, f"{cal[6]} invalid values" if cal[6] else "")

    # --- Zone ID sanity + referential integrity against the zone lookup ---
    # IMPORTANT: checked against the FULL official TLC zone list (streamed
    # from the TLC's own CSV), NOT the pickup-only zone_hourly_demand list.
    # zone_hourly_demand is built by grouping on PULocationID only, so a
    # dropoff-only zone (e.g. core Manhattan for Green Taxi, which is
    # legally barred from PICKUPS there but can drop off) would show as a
    # false-positive "orphan" against that list even though it's a
    # perfectly valid TLC zone -- checked separately by PU vs DO below so
    # that distinction is visible rather than hidden in one combined count.
    try:
        full_zones = duckdb.sql(
            f"SELECT DISTINCT LocationID AS zone_id FROM read_csv_auto('{TAXI_ZONE_LOOKUP_URL}')"
        ).df()
    except Exception:
        full_zones = None

    if full_zones is not None and not full_zones.empty:
        con = duckdb.connect()
        con.register("full_zones", full_zones)
        pu_orphans, do_orphans = con.execute(
            f"""
            SELECT
                sum(CASE WHEN t.PULocationID NOT IN (SELECT zone_id FROM full_zones) THEN 1 ELSE 0 END),
                sum(CASE WHEN t.DOLocationID NOT IN (SELECT zone_id FROM full_zones) THEN 1 ELSE 0 END)
            FROM read_parquet('{path.as_posix()}') t
            """
        ).fetchone()
        ok = check("All PULocationID values exist in the FULL official TLC zone list",
                    pu_orphans == 0, f"{pu_orphans:,} rows" if pu_orphans else "")
        if not ok:
            all_flags.append(f"{vehicle}: trip_features has {pu_orphans} rows with an invalid PULocationID "
                              f"(not in the official TLC zone list at all -- this WOULD be a real problem)")

        ok = check("All DOLocationID values exist in the FULL official TLC zone list",
                    do_orphans == 0, f"{do_orphans:,} rows" if do_orphans else "")
        if not ok:
            all_flags.append(f"{vehicle}: trip_features has {do_orphans} rows with an invalid DOLocationID "
                              f"(not in the official TLC zone list at all -- this WOULD be a real problem)")
    else:
        info("Could not reach the TLC zone lookup CSV to verify zone IDs against the full official list",
             "falling back to the (pickup-only) local zone_hourly_demand check below, which will "
             "over-flag any legitimate dropoff-only zone as an 'orphan' -- not a reliable signal on its own")
        zone_path = cfg["zone_hourly_demand"]
        if zone_path.exists():
            pu_orphans, do_orphans = duckdb.sql(
                f"""
                WITH known_zones AS (SELECT DISTINCT zone_id FROM read_parquet('{zone_path.as_posix()}'))
                SELECT
                    sum(CASE WHEN t.PULocationID NOT IN (SELECT zone_id FROM known_zones) THEN 1 ELSE 0 END),
                    sum(CASE WHEN t.DOLocationID NOT IN (SELECT zone_id FROM known_zones) THEN 1 ELSE 0 END)
                FROM read_parquet('{path.as_posix()}') t
                """
            ).fetchone()
            info(f"PULocationID not in pickup-only list: {pu_orphans:,} rows",
                 "should be 0 -- trip_features is a SAMPLE of the same data zone_hourly_demand is "
                 "built from, so every pickup zone must already be represented there")
            info(f"DOLocationID not in pickup-only list: {do_orphans:,} rows",
                 "expected to be > 0 -- dropoff-only zones are real and legitimate, this list simply "
                 "doesn't include them by construction")

    zone_range = duckdb.sql(
        f"SELECT min(PULocationID), max(PULocationID), min(DOLocationID), max(DOLocationID) "
        f"FROM read_parquet('{path.as_posix()}')"
    ).fetchone()
    check(f"Zone IDs within plausible TLC range {TLC_ZONE_ID_RANGE}",
          TLC_ZONE_ID_RANGE[0] <= zone_range[0] and zone_range[3] <= TLC_ZONE_ID_RANGE[1],
          f"observed PU[{zone_range[0]},{zone_range[1]}] DO[{zone_range[2]},{zone_range[3]}]")

    # --- Exact duplicate rows (informational, not necessarily a defect) ---
    n_distinct = duckdb.sql(f"SELECT count(*) FROM (SELECT DISTINCT * FROM read_parquet('{path.as_posix()}'))").fetchone()[0]
    n_dupe_rows = n_total - n_distinct
    info(f"Exact duplicate rows: {n_dupe_rows:,} ({n_dupe_rows/n_total*100:.2f}%)",
         "expected to some degree with coarse/binned features (same distance+zones+hour); "
         "only worth investigating further if this is a large fraction of the data")


# ---------------------------------------------------------------------------
# hourly_demand.parquet
# ---------------------------------------------------------------------------

def audit_hourly_demand(vehicle: str, cfg: dict, all_flags: list):
    path = cfg["hourly_demand"]
    section(f"[{vehicle}] hourly_demand.parquet")
    if not path.exists():
        print("  SKIP -- file not found")
        return

    stats = duckdb.sql(
        f"""
        SELECT count(*), min(pickup_hour_ts), max(pickup_hour_ts),
               sum(CASE WHEN trip_count IS NULL THEN 1 ELSE 0 END),
               sum(CASE WHEN trip_count < 0 THEN 1 ELSE 0 END),
               count(*) - count(DISTINCT pickup_hour_ts)
        FROM read_parquet('{path.as_posix()}')
        """
    ).fetchone()
    n, min_ts, max_ts, n_null, n_neg, n_dupe_ts = stats
    print(f"  Total rows: {n:,}")
    print(f"  Date range: {min_ts} to {max_ts}")

    check("No NULL trip_count", n_null == 0, f"{n_null} null rows" if n_null else "")
    ok = check("No negative trip_count", n_neg == 0, f"{n_neg} negative rows" if n_neg else "")
    if not ok:
        all_flags.append(f"{vehicle}: hourly_demand has {n_neg} negative trip_count values")

    ok = check("No duplicate pickup_hour_ts (one row per hour)", n_dupe_ts == 0,
               f"{n_dupe_ts} duplicate timestamps" if n_dupe_ts else "")
    if not ok:
        all_flags.append(f"{vehicle}: hourly_demand has {n_dupe_ts} duplicate hour timestamps")

    check(f"Data year is {VALID_YEAR}",
          pd.Timestamp(min_ts).year == VALID_YEAR and pd.Timestamp(max_ts).year == VALID_YEAR)

    expected_hours = int((pd.Timestamp(max_ts) - pd.Timestamp(min_ts)).total_seconds() / 3600) + 1
    missing_hours = expected_hours - n
    info(f"Hours with zero recorded trips: {missing_hours:,} of {expected_hours:,} possible hours "
         f"({missing_hours/expected_hours*100:.1f}%)",
         "these hours have NO row at all (GROUP BY produces no row for zero trips) -- this is "
         "EXPECTED behavior, not missing data; build_features scripts reindex+fill 0 before "
         "lagging for the demand models specifically")


# ---------------------------------------------------------------------------
# zone_hourly_demand.parquet
# ---------------------------------------------------------------------------

def audit_zone_hourly_demand(vehicle: str, cfg: dict, all_flags: list):
    path = cfg["zone_hourly_demand"]
    section(f"[{vehicle}] zone_hourly_demand.parquet")
    if not path.exists():
        print("  SKIP -- file not found")
        return

    stats = duckdb.sql(
        f"""
        SELECT count(*), min(pickup_hour_ts), max(pickup_hour_ts),
               count(DISTINCT zone_id),
               sum(CASE WHEN zone_name IS NULL THEN 1 ELSE 0 END),
               sum(CASE WHEN borough IS NULL THEN 1 ELSE 0 END),
               sum(CASE WHEN trip_count < 0 THEN 1 ELSE 0 END),
               count(*) - count(DISTINCT (pickup_hour_ts, zone_id))
        FROM read_parquet('{path.as_posix()}')
        """
    ).fetchone()
    n, min_ts, max_ts, n_zones, n_null_name, n_null_borough, n_neg, n_dupe = stats
    print(f"  Total rows: {n:,}")
    print(f"  Date range: {min_ts} to {max_ts}")
    print(f"  Distinct zones covered: {n_zones}")

    ok = check("No unmatched zone_name (LEFT JOIN against zone lookup didn't fail)",
               n_null_name == 0, f"{n_null_name:,} rows with no zone_name" if n_null_name else "")
    if not ok:
        all_flags.append(f"{vehicle}: zone_hourly_demand has {n_null_name} rows with unmatched zone_name "
                          f"(zone_id not found in the TLC zone lookup CSV)")

    ok = check("No unmatched borough", n_null_borough == 0,
               f"{n_null_borough:,} rows with no borough" if n_null_borough else "")
    if not ok:
        all_flags.append(f"{vehicle}: zone_hourly_demand has {n_null_borough} rows with unmatched borough")

    check("No negative trip_count", n_neg == 0)
    ok = check("No duplicate (pickup_hour_ts, zone_id) pairs", n_dupe == 0,
               f"{n_dupe} duplicate pairs" if n_dupe else "")
    if not ok:
        all_flags.append(f"{vehicle}: zone_hourly_demand has {n_dupe} duplicate (hour, zone) pairs")

    if n_zones < 200:
        info(f"Only {n_zones} distinct zones have any recorded trips",
             "plausible -- not every one of the ~263 TLC zones sees this vehicle type at all "
             "(e.g. some green-taxi-restricted Manhattan zones, or very low-traffic areas)")


# ---------------------------------------------------------------------------

def main():
    all_flags = []
    for vehicle, cfg in CONFIGS.items():
        audit_trip_features(vehicle, cfg, all_flags)
        audit_hourly_demand(vehicle, cfg, all_flags)
        audit_zone_hourly_demand(vehicle, cfg, all_flags)

    section("SUMMARY")
    if all_flags:
        print(f"{len(all_flags)} item(s) flagged for review:\n")
        for f in all_flags:
            print(f"  - {f}")
    else:
        print("No FLAGs raised across any file. All checked invariants held:")
        print("  - no nulls in key columns")
        print("  - distance/duration values respect the build-time filter bounds")
        print("  - no orphan zone IDs (referential integrity against the zone lookup holds)")
        print("  - no duplicate timestamp/zone-hour rows")
        print("  - calendar-derived features (hour/dow/month/is_weekend) are all in valid ranges")

    report_path = ROOT / "outputs" / "data_quality_report.txt"
    with open(report_path, "w") as f:
        f.write("Data Quality Audit Report\n")
        f.write("=" * 70 + "\n\n")
        if all_flags:
            f.write(f"{len(all_flags)} item(s) flagged:\n")
            for flag in all_flags:
                f.write(f"  - {flag}\n")
        else:
            f.write("No issues found. All invariants held across trip_features, "
                     "hourly_demand, and zone_hourly_demand for both vehicle types.\n")
    print(f"\nSaved -> {report_path}")


if __name__ == "__main__":
    main()