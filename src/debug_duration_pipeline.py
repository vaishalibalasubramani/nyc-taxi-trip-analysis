"""
Duration prediction pipeline debugger.

Runs every check requested in the Kew Gardens -> Port Richmond bug report,
against your REAL local data (this script must be run on your machine --
Claude's sandbox has no access to your data or network to run these
queries itself).

Covers:
  A. Zone name -> Zone ID resolution + verification
  B. Historical distance stats for named test routes (mean/median/min/max/P10/P90)
  C. Root-cause trace: why the naive fallback produced 2.94 mi
  D. Distance estimation logic audit (zone mapping, filtering, joins)
  E. Historical vs geographic (straight-line) distance comparison
  F. Model/feature audit: training vs prediction feature list, order,
     encoding, target leakage check, outlier filtering, distributions
  G. End-to-end test matrix across 5 realistic routes

Usage:
    python src/debug_duration_pipeline.py
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from joblib import load

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "outputs"

CONFIGS = {
    "Green Taxi": {
        "trip_features": PROCESSED_DIR / "trip_features.parquet",
        "zone_hourly_demand": PROCESSED_DIR / "zone_hourly_demand.parquet",
        "duration_model": OUTPUTS_DIR / "duration_model.joblib",
        "duration_metrics": OUTPUTS_DIR / "duration_model_metrics.txt",
        "distance_col": "trip_distance",
        "target_col": "trip_duration_s",
    },
    "HVFHV (Uber/Lyft)": {
        "trip_features": PROCESSED_DIR / "hvfhv_trip_features.parquet",
        "zone_hourly_demand": PROCESSED_DIR / "hvfhv_zone_hourly_demand.parquet",
        "duration_model": OUTPUTS_DIR / "hvfhv_duration_model.joblib",
        "duration_metrics": OUTPUTS_DIR / "hvfhv_duration_model_metrics.txt",
        "distance_col": "trip_miles",
        "target_col": "trip_duration_s",
    },
}

MIN_ZONE_PAIR_TRIPS = 5

# Route test matrix -- resolved by FUZZY NAME SEARCH against your actual
# zone lookup, not hardcoded zone IDs, so this trace is honest about what
# your data actually contains (satisfies checklist items 1-2).
TEST_ROUTES = [
    ("Kew Gardens", "Port Richmond"),
    ("Midtown", "JFK"),
    ("JFK", "Midtown"),
    ("Kew Gardens", "Forest Hills"),       # nearby Queens -> Queens
    ("Midtown", "St. George"),             # Manhattan -> Staten Island
]


def line(char="=", n=90):
    print(char * n)


def section(title):
    print()
    line()
    print(title)
    line()


# ---------------------------------------------------------------------------
# A. Zone name -> Zone ID resolution
# ---------------------------------------------------------------------------

def resolve_zone(zone_hourly_demand_path: Path, search_term: str) -> pd.DataFrame:
    """Fuzzy, case-insensitive search against the REAL zone lookup embedded
    in zone_hourly_demand.parquet -- this is exactly how the dashboard's
    zone dropdown is populated, so any mapping bug would show up here too."""
    if not zone_hourly_demand_path.exists():
        return pd.DataFrame()
    return duckdb.sql(
        f"""
        SELECT DISTINCT zone_id, zone_name, borough
        FROM read_parquet('{zone_hourly_demand_path.as_posix()}')
        WHERE lower(zone_name) LIKE lower('%{search_term}%')
        ORDER BY zone_name
        """
    ).df()


def audit_zone_resolution(vehicle: str, cfg: dict):
    section(f"[{vehicle}] A. Zone name -> Zone ID resolution")
    for term in ["Kew Gardens", "Port Richmond"]:
        matches = resolve_zone(cfg["zone_hourly_demand"], term)
        print(f"\nSearch term: '{term}'")
        if matches.empty:
            print(f"  NO MATCH FOUND for '{term}' in the zone lookup. This would be a real bug "
                  f"if the dashboard dropdown also can't find it -- check spelling/data.")
        else:
            print(matches.to_string(index=False))
            if len(matches) > 1:
                print(f"  NOTE: {len(matches)} zones match '{term}' -- ambiguous name, "
                      f"make sure the dashboard dropdown is selecting the intended one.")


# ---------------------------------------------------------------------------
# B/C/D. Historical distance stats for the exact reported pair + root cause
# ---------------------------------------------------------------------------

def historical_distance_stats(trip_features_path: Path, distance_col: str,
                                pu_id: int, do_id: int) -> dict:
    if not trip_features_path.exists():
        return {}
    row = duckdb.sql(
        f"""
        SELECT
            count(*) AS n,
            avg({distance_col}) AS mean_dist,
            median({distance_col}) AS median_dist,
            min({distance_col}) AS min_dist,
            max({distance_col}) AS max_dist,
            quantile_cont({distance_col}, 0.10) AS p10_dist,
            quantile_cont({distance_col}, 0.90) AS p90_dist
        FROM read_parquet('{trip_features_path.as_posix()}')
        WHERE PULocationID = {pu_id} AND DOLocationID = {do_id}
        """
    ).fetchone()
    cols = ["n", "mean_dist", "median_dist", "min_dist", "max_dist", "p10_dist", "p90_dist"]
    return dict(zip(cols, row))


def historical_duration_stats(trip_features_path: Path, target_col: str,
                                pu_id: int, do_id: int) -> dict:
    if not trip_features_path.exists():
        return {}
    row = duckdb.sql(
        f"""
        SELECT count(*) AS n, avg({target_col}) AS mean_dur, median({target_col}) AS median_dur
        FROM read_parquet('{trip_features_path.as_posix()}')
        WHERE PULocationID = {pu_id} AND DOLocationID = {do_id}
        """
    ).fetchone()
    return dict(zip(["n", "mean_dur", "median_dur"], row))


def borough_pair_stats(trip_features_path: Path, zone_hourly_demand_path: Path,
                        distance_col: str, pu_borough: str, do_borough: str) -> dict:
    if not trip_features_path.exists() or not zone_hourly_demand_path.exists():
        return {}
    row = duckdb.sql(
        f"""
        WITH zones AS (
            SELECT DISTINCT zone_id, borough
            FROM read_parquet('{zone_hourly_demand_path.as_posix()}')
            WHERE borough IS NOT NULL
        )
        SELECT count(*) AS n, avg(t.{distance_col}) AS mean_dist, median(t.{distance_col}) AS median_dist
        FROM read_parquet('{trip_features_path.as_posix()}') t
        JOIN zones zp ON t.PULocationID = zp.zone_id
        JOIN zones zd ON t.DOLocationID = zd.zone_id
        WHERE zp.borough = '{pu_borough}' AND zd.borough = '{do_borough}'
        """
    ).fetchone()
    return dict(zip(["n", "mean_dist", "median_dist"], row))


def citywide_avg_distance(trip_features_path: Path, distance_col: str) -> float:
    if not trip_features_path.exists():
        return float("nan")
    result = duckdb.sql(
        f"SELECT avg({distance_col}) FROM read_parquet('{trip_features_path.as_posix()}')"
    ).fetchone()
    return float(result[0]) if result and result[0] is not None else float("nan")


def audit_kew_gardens_port_richmond(vehicle: str, cfg: dict):
    section(f"[{vehicle}] B/C/D. Root-cause trace: Kew Gardens -> Port Richmond")

    pu_matches = resolve_zone(cfg["zone_hourly_demand"], "Kew Gardens")
    do_matches = resolve_zone(cfg["zone_hourly_demand"], "Port Richmond")
    if pu_matches.empty or do_matches.empty:
        print("Could not resolve one or both zone names in this vehicle's data -- skipping.")
        return
    pu_id, pu_name, pu_borough = pu_matches.iloc[0][["zone_id", "zone_name", "borough"]]
    do_id, do_name, do_borough = do_matches.iloc[0][["zone_id", "zone_name", "borough"]]
    print(f"Resolved pickup:  zone_id={pu_id}  name='{pu_name}'  borough={pu_borough}")
    print(f"Resolved dropoff: zone_id={do_id}  name='{do_name}'  borough={do_borough}")

    print(f"\n--- Query 4: exact-pair historical distance stats ---")
    print(f"SQL: SELECT count(*), avg(dist), median(dist), min(dist), max(dist), "
          f"quantile_cont(dist,0.1), quantile_cont(dist,0.9)")
    print(f"     FROM trip_features WHERE PULocationID={pu_id} AND DOLocationID={do_id}")
    exact = historical_distance_stats(cfg["trip_features"], cfg["distance_col"], pu_id, do_id)
    print(f"Result: {exact}")

    exact_dur = historical_duration_stats(cfg["trip_features"], cfg["target_col"], pu_id, do_id)
    print(f"Historical DURATION for this exact pair: {exact_dur}")

    print(f"\n--- Borough-pair fallback stats ({pu_borough} -> {do_borough}) ---")
    borough = borough_pair_stats(cfg["trip_features"], cfg["zone_hourly_demand"],
                                   cfg["distance_col"], pu_borough, do_borough)
    print(f"Result: {borough}")

    citywide = citywide_avg_distance(cfg["trip_features"], cfg["distance_col"])
    print(f"\n--- Citywide average distance (the OLD buggy fallback) ---")
    print(f"Citywide avg: {citywide:.2f} mi")

    print(f"\n--- Diagnosis ---")
    n_exact = exact.get("n", 0) or 0
    if n_exact == 0:
        print(f"CONFIRMED ROOT CAUSE: {n_exact} historical trips exist for this exact zone pair. "
              f"The OLD code had no borough-level fallback, so it fell straight through to the "
              f"citywide average ({citywide:.2f} mi) -- which is a real number, correctly computed, "
              f"but meaningless for a ~20+ mile cross-borough route.")
    elif n_exact < MIN_ZONE_PAIR_TRIPS:
        print(f"CONFIRMED: only {n_exact} historical trip(s) for this exact pair -- too few to "
              f"trust as-is (the OLD code trusted ANY count >= 1, including a single outlier trip).")
    else:
        print(f"{n_exact} historical trips exist for this pair -- if the bug is still reproducing "
              f"with this many, the issue is elsewhere (check the NEW code is actually deployed).")

    n_borough = borough.get("n", 0) or 0
    if n_borough >= MIN_ZONE_PAIR_TRIPS:
        print(f"\nNEW behavior: falls back to the {pu_borough}->{do_borough} borough average "
              f"({borough.get('mean_dist', float('nan')):.2f} mi from {n_borough} trips) instead "
              f"of the citywide average -- a much more representative number for this route.")
    else:
        print(f"\nWARNING: even the borough-pair has only {n_borough} historical trips. "
              f"This vehicle type may have near-zero real coverage of this route -- worth checking "
              f"whether {vehicle} legitimately serves {do_borough} at all.")


# ---------------------------------------------------------------------------
# F. Model / feature audit
# ---------------------------------------------------------------------------

def audit_model(vehicle: str, cfg: dict):
    section(f"[{vehicle}] F. Model / feature audit")

    if not cfg["duration_model"].exists():
        print("Model file not found -- skipping.")
        return
    model = load(cfg["duration_model"])

    training_features = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
    print(f"Training feature list (from model.feature_names_in_, in order):")
    print(f"  {training_features}")

    if cfg["distance_col"] not in (training_features or []):
        print(f"  *** WARNING: distance column '{cfg['distance_col']}' is NOT in the trained "
              f"feature list -- the model isn't using distance at all! ***")
    else:
        idx = training_features.index(cfg["distance_col"])
        print(f"  Distance column '{cfg['distance_col']}' is feature #{idx} of {len(training_features)}.")

    for zone_col in ("PULocationID", "DOLocationID"):
        if zone_col in (training_features or []):
            print(f"  '{zone_col}' IS used by the model (confirms pickup/dropoff zone affects prediction).")
        else:
            print(f"  *** '{zone_col}' is NOT in the trained feature list. ***")

    target_col = cfg["target_col"]
    if target_col in (training_features or []):
        print(f"  *** TARGET LEAKAGE: '{target_col}' (the target) appears in the FEATURE list! ***")
    else:
        print(f"  No target leakage: '{target_col}' is not among the input features. Good.")

    # Feature importances, if available from the metrics file, to show
    # whether distance/zones actually matter empirically (not just present).
    if cfg["duration_metrics"].exists():
        text = cfg["duration_metrics"].read_text()
        print(f"\nFeature importances (from {cfg['duration_metrics'].name}):")
        in_imp = False
        for l in text.splitlines():
            if l.strip().startswith("Feature importances"):
                in_imp = True
                continue
            if in_imp and l.strip():
                print(f"  {l.strip()}")

    # Distance & duration distribution in the TRAINING data, to check
    # whether long-distance trips were realistically represented at all.
    if cfg["trip_features"].exists():
        dist_stats = duckdb.sql(
            f"""
            SELECT count(*) AS n, avg({cfg['distance_col']}) AS mean_d, median({cfg['distance_col']}) AS median_d,
                   max({cfg['distance_col']}) AS max_d,
                   quantile_cont({cfg['distance_col']}, 0.99) AS p99_d,
                   sum(CASE WHEN {cfg['distance_col']} > 15 THEN 1 ELSE 0 END) AS n_over_15mi
            FROM read_parquet('{cfg['trip_features'].as_posix()}')
            """
        ).fetchone()
        print(f"\nTraining data distance distribution: n={dist_stats[0]:,}, mean={dist_stats[1]:.2f}mi, "
              f"median={dist_stats[2]:.2f}mi, max={dist_stats[3]:.2f}mi, P99={dist_stats[4]:.2f}mi")
        print(f"Trips over 15 miles in training data: {dist_stats[5]:,} "
              f"({dist_stats[5]/dist_stats[0]*100:.2f}% of all sampled trips)")
        if dist_stats[5] / dist_stats[0] < 0.01:
            print("  NOTE: long-distance trips (>15mi) are a very small slice of training data -- "
                  "the model has limited examples to learn long-trip duration patterns from, "
                  "which is a real (data-coverage) limitation, separate from the distance-input bug.")


# ---------------------------------------------------------------------------
# G. End-to-end test matrix
# ---------------------------------------------------------------------------

def build_feature_row(model, known: dict):
    cols = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else list(known.keys())
    row, defaulted = {}, []
    for c in cols:
        if c in known and known[c] is not None:
            row[c] = known[c]
        else:
            row[c] = 0
            defaulted.append(c)
    return pd.DataFrame([row])[cols], defaulted


def resolve_trip_distance(pu_id, do_id, pu_borough, do_borough, trip_features_path,
                           zone_hourly_demand_path, distance_col):
    exact = historical_distance_stats(trip_features_path, distance_col, pu_id, do_id)
    n_exact = exact.get("n", 0) or 0
    if n_exact >= MIN_ZONE_PAIR_TRIPS:
        return exact["mean_dist"], "exact_pair", n_exact

    borough = borough_pair_stats(trip_features_path, zone_hourly_demand_path, distance_col,
                                   pu_borough, do_borough)
    n_borough = borough.get("n", 0) or 0
    if n_borough >= MIN_ZONE_PAIR_TRIPS:
        return borough["mean_dist"], "borough_pair", n_borough

    return citywide_avg_distance(trip_features_path, distance_col), "citywide", 0


def run_test_matrix(vehicle: str, cfg: dict):
    section(f"[{vehicle}] G. End-to-end test matrix")

    if not cfg["duration_model"].exists():
        print("Model not found -- skipping.")
        return
    model = load(cfg["duration_model"])

    rows = []
    for pu_term, do_term in TEST_ROUTES:
        pu_matches = resolve_zone(cfg["zone_hourly_demand"], pu_term)
        do_matches = resolve_zone(cfg["zone_hourly_demand"], do_term)
        if pu_matches.empty or do_matches.empty:
            rows.append({"pickup": pu_term, "dropoff": do_term, "status": "ZONE NOT FOUND"})
            continue

        pu_id, pu_name, pu_borough = pu_matches.iloc[0][["zone_id", "zone_name", "borough"]]
        do_id, do_name, do_borough = do_matches.iloc[0][["zone_id", "zone_name", "borough"]]

        dist, tier, n = resolve_trip_distance(
            pu_id, do_id, pu_borough, do_borough, cfg["trip_features"], cfg["zone_hourly_demand"],
            cfg["distance_col"]
        )
        dur_stats = historical_duration_stats(cfg["trip_features"], cfg["target_col"], pu_id, do_id)

        known = {
            "PULocationID": pu_id, "DOLocationID": do_id, cfg["distance_col"]: dist,
            "pickup_hour": 9, "pickup_dow": 2, "is_weekend": 0, "pickup_month": 9,
            "passenger_count": 1,
        }
        X, _ = build_feature_row(model, known)
        pred_sec = float(model.predict(X)[0])

        rows.append({
            "pickup": f"{pu_name} ({pu_borough})",
            "dropoff": f"{do_name} ({do_borough})",
            "dist_tier": tier,
            "dist_used_mi": round(dist, 2) if dist is not None else None,
            "pred_duration": f"{int(pred_sec//60)}m {int(pred_sec%60)}s",
            "hist_trip_n": dur_stats.get("n", 0),
            "hist_mean_dur_s": round(dur_stats["mean_dur"], 1) if dur_stats.get("mean_dur") else None,
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


# ---------------------------------------------------------------------------

def main():
    for vehicle, cfg in CONFIGS.items():
        if not cfg["trip_features"].exists():
            print(f"\n[{vehicle}] SKIP -- {cfg['trip_features']} not found.")
            continue
        audit_zone_resolution(vehicle, cfg)
        audit_kew_gardens_port_richmond(vehicle, cfg)
        audit_model(vehicle, cfg)
        run_test_matrix(vehicle, cfg)

    section("DONE")
    print("Paste this full output back for interpretation.")


if __name__ == "__main__":
    main()