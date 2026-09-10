"""
NYC Taxi & HVFHV Analytics Dashboard.

Two pages, selectable from the sidebar:
  1. Predict -- three tabs, one per trained task:
       - Trip Duration: pick a vehicle type, pickup zone, dropoff zone,
         date, time, passenger count -> predicted trip duration.
       - Citywide Demand: pick a vehicle type, date, hour -> predicted
         citywide trips for that hour, using actual recent trip counts
         (1/2/3 hrs ago, same hour last week, etc.) pulled from your
         processed data, since this model forecasts from recent history
         rather than a standalone scenario.
       - Zone Demand: same idea as Citywide Demand, but for one specific
         pickup zone at a time.
  2. Model Performance Dashboard -- manager-facing view of all 6 trained
     models (2 vehicle types x 3 tasks), metrics computed live via SQL
     (DuckDB) against the saved prediction/metrics files, with plain-
     language explanations under every chart.

Run with:
    streamlit run src/dashboard_app.py

Requires (add to requirements.txt if not already present):
    streamlit
    duckdb
    plotly
"""

from pathlib import Path
from datetime import date, time as dtime

import duckdb
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths -- matches the layout used by build_features.py / build_features_hvfhv.py
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "outputs"

PATHS = {
    "Green Taxi": {
        "trip_features": PROCESSED_DIR / "trip_features.parquet",
        "hourly_demand": PROCESSED_DIR / "hourly_demand.parquet",
        "zone_hourly_demand": PROCESSED_DIR / "zone_hourly_demand.parquet",
        "duration_model": OUTPUTS_DIR / "duration_model.joblib",
        "duration_metrics": OUTPUTS_DIR / "duration_model_metrics.txt",
        "demand_model": OUTPUTS_DIR / "demand_model.joblib",
        "demand_predictions": OUTPUTS_DIR / "demand_predictions.parquet",
        "zone_demand_model": OUTPUTS_DIR / "zone_demand_model.joblib",
        "zone_demand_predictions": OUTPUTS_DIR / "zone_demand_predictions.parquet",
        "pickup_dt_col": "lpep_pickup_datetime",
        "distance_col": "trip_distance",
    },
    "HVFHV (Uber/Lyft)": {
        "trip_features": PROCESSED_DIR / "hvfhv_trip_features.parquet",
        "hourly_demand": PROCESSED_DIR / "hvfhv_hourly_demand.parquet",
        "zone_hourly_demand": PROCESSED_DIR / "hvfhv_zone_hourly_demand.parquet",
        "duration_model": OUTPUTS_DIR / "hvfhv_duration_model.joblib",
        "duration_metrics": OUTPUTS_DIR / "hvfhv_duration_model_metrics.txt",
        "demand_model": OUTPUTS_DIR / "hvfhv_demand_model.joblib",
        "demand_predictions": OUTPUTS_DIR / "hvfhv_demand_predictions.parquet",
        "zone_demand_model": OUTPUTS_DIR / "hvfhv_zone_demand_model.joblib",
        "zone_demand_predictions": OUTPUTS_DIR / "hvfhv_zone_demand_predictions.parquet",
        "pickup_dt_col": "pickup_datetime",
        "distance_col": "trip_miles",
    },
}

# HVFHV license codes, per TLC documentation. astype("category").cat.codes
# assigns integers in *sorted* order of the categories seen during training,
# so as long as all four appeared in the training sample this mapping holds.
# If your training data didn't include all four, adjust this after checking
# `dict(enumerate(sorted(df['hvfhs_license_num'].unique())))` on your own data.
HVFHS_CODES = {"Juno (HV0002)": 0, "Uber (HV0003)": 1, "Via (HV0004)": 2, "Lyft (HV0005)": 3}


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_zone_list(zone_hourly_demand_path: Path) -> pd.DataFrame:
    """Distinct zone_id / zone_name / borough, pulled via SQL from the
    already-built zone_hourly_demand parquet (no network needed). NOTE:
    this is PICKUP-ZONE-ONLY -- zone_hourly_demand is built by grouping on
    PULocationID, so a zone that only ever appears as a DROPOFF (e.g. core
    Manhattan for Green Taxi, which is legally barred from street-hail
    PICKUPS there but can drop off) will be missing from this list. Correct
    to use for the Zone Demand task (forecasting pickup demand for a zone
    that never has pickups isn't meaningful), but NOT for Trip Duration's
    dropoff dropdown -- see load_full_zone_lookup() for that."""
    if not zone_hourly_demand_path.exists():
        return pd.DataFrame(columns=["zone_id", "zone_name", "borough"])
    return duckdb.sql(
        f"""
        SELECT DISTINCT zone_id, zone_name, borough
        FROM read_parquet('{zone_hourly_demand_path.as_posix()}')
        WHERE zone_name IS NOT NULL
        ORDER BY borough, zone_name
        """
    ).df()


TAXI_ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"


@st.cache_data(show_spinner=False)
def load_full_zone_lookup(zone_hourly_demand_path: Path) -> pd.DataFrame:
    """ALL ~263 official TLC zones (pickup AND dropoff-only), streamed
    directly from the TLC's own lookup CSV -- same DuckDB-httpfs technique
    already used throughout build_features*.py, not a new dependency or a
    downloaded file. This is what Trip Duration's pickup/dropoff dropdowns
    should use: a zone that's a legitimate dropoff (e.g. Midtown for Green
    Taxi) but never a pickup would otherwise be silently missing, as
    confirmed by the ~13.6% of trip_features rows whose DOLocationID
    doesn't appear in the pickup-only zone_hourly_demand list.
    Falls back to the pickup-only list (load_zone_list) if the remote CSV
    can't be reached, so the app still works, just with the old limitation."""
    try:
        return duckdb.sql(
            f"""
            SELECT LocationID AS zone_id, Zone AS zone_name, Borough AS borough
            FROM read_csv_auto('{TAXI_ZONE_LOOKUP_URL}')
            WHERE Zone IS NOT NULL
            ORDER BY Borough, Zone
            """
        ).df()
    except Exception:
        return load_zone_list(zone_hourly_demand_path)


MIN_ZONE_PAIR_TRIPS = 5  # below this, an "exact pair" average is trusted too little
                          # to use on its own -- a pair with 1-2 historical trips can
                          # give a wildly unrepresentative average, which is exactly
                          # what happened for rare cross-borough pairs like
                          # Kew Gardens -> Port Richmond.


@st.cache_data(show_spinner=False)
def load_zone_pair_distance_stats(trip_features_path: Path, distance_col: str) -> pd.DataFrame:
    """Average distance AND trip count for each (pickup, dropoff) zone pair.
    The count matters as much as the average: an average built from 1-2
    historical trips is not trustworthy, and the prediction logic needs to
    know that to decide whether to fall back to a broader estimate."""
    if not trip_features_path.exists():
        return pd.DataFrame(columns=["PULocationID", "DOLocationID", "avg_trip_distance", "trip_count"])
    return duckdb.sql(
        f"""
        SELECT PULocationID, DOLocationID,
               avg({distance_col}) AS avg_trip_distance,
               count(*) AS trip_count
        FROM read_parquet('{trip_features_path.as_posix()}')
        GROUP BY 1, 2
        """
    ).df()


@st.cache_data(show_spinner=False)
def load_data_date_range(hourly_demand_path: Path) -> tuple:
    """Min/max pickup date actually present in this vehicle's data --
    used to constrain every date picker in the app so the calendar can't
    show years with no data behind them at all (e.g. 2017 or 2026)."""
    if not hourly_demand_path.exists():
        return date(2025, 1, 1), date(2025, 12, 31)
    row = duckdb.sql(
        f"SELECT min(pickup_hour_ts), max(pickup_hour_ts) FROM read_parquet('{hourly_demand_path.as_posix()}')"
    ).fetchone()
    if row and row[0] is not None:
        return pd.Timestamp(row[0]).date(), pd.Timestamp(row[1]).date()
    return date(2025, 1, 1), date(2025, 12, 31)


@st.cache_data(show_spinner=False)
def load_borough_pair_distance_stats(trip_features_path: Path, zone_hourly_demand_path: Path,
                                      distance_col: str) -> pd.DataFrame:
    """Average distance AND trip count per (pickup_borough, dropoff_borough) --
    the fallback tier BETWEEN an exact zone pair and the citywide average.
    For a rare pair like Queens->Staten Island, this is far more
    representative than a flat citywide number dominated by short
    in-borough trips. Built entirely from data already in the pipeline
    (trip_features joined against the zone lookup already embedded in
    zone_hourly_demand) -- no new external fetch needed."""
    if not trip_features_path.exists() or not zone_hourly_demand_path.exists():
        return pd.DataFrame(columns=["pickup_borough", "dropoff_borough", "avg_trip_distance", "trip_count"])
    return duckdb.sql(
        f"""
        WITH zones AS (
            SELECT DISTINCT zone_id, borough
            FROM read_parquet('{zone_hourly_demand_path.as_posix()}')
            WHERE borough IS NOT NULL
        )
        SELECT
            zp.borough AS pickup_borough,
            zd.borough AS dropoff_borough,
            avg(t.{distance_col}) AS avg_trip_distance,
            count(*) AS trip_count
        FROM read_parquet('{trip_features_path.as_posix()}') t
        JOIN zones zp ON t.PULocationID = zp.zone_id
        JOIN zones zd ON t.DOLocationID = zd.zone_id
        GROUP BY 1, 2
        """
    ).df()


@st.cache_data(show_spinner=False)
def load_overall_avg_trip_distance(trip_features_path: Path, distance_col: str) -> float:
    if not trip_features_path.exists():
        return 3.0
    result = duckdb.sql(
        f"SELECT avg({distance_col}) FROM read_parquet('{trip_features_path.as_posix()}')"
    ).fetchone()
    return float(result[0]) if result and result[0] is not None else 3.0


def resolve_trip_distance(pu_id: int, do_id: int, pu_borough: str, do_borough: str,
                           zone_pair_stats: pd.DataFrame, borough_pair_stats: pd.DataFrame,
                           overall_avg: float) -> dict:
    """Three-tier distance resolution, most-specific-available tier wins:
      1. Exact zone pair, IF it has at least MIN_ZONE_PAIR_TRIPS historical trips.
      2. Borough pair (e.g. Queens -> Staten Island), IF that has enough trips.
      3. Citywide average, as an explicit last resort.
    Returns everything needed to show the user exactly which tier was used
    and why, rather than silently picking a number."""
    exact = zone_pair_stats[
        (zone_pair_stats["PULocationID"] == pu_id) & (zone_pair_stats["DOLocationID"] == do_id)
    ]
    exact_count = int(exact["trip_count"].iloc[0]) if not exact.empty else 0
    exact_avg = float(exact["avg_trip_distance"].iloc[0]) if not exact.empty else None

    if exact_count >= MIN_ZONE_PAIR_TRIPS:
        return {
            "distance": exact_avg, "tier": "exact_pair",
            "note": f"based on {exact_count:,} historical trips between these exact zones",
            "exact_count": exact_count, "exact_avg": exact_avg,
        }

    borough_match = borough_pair_stats[
        (borough_pair_stats["pickup_borough"] == pu_borough)
        & (borough_pair_stats["dropoff_borough"] == do_borough)
    ]
    borough_count = int(borough_match["trip_count"].iloc[0]) if not borough_match.empty else 0
    borough_avg = float(borough_match["avg_trip_distance"].iloc[0]) if not borough_match.empty else None

    if borough_count >= MIN_ZONE_PAIR_TRIPS:
        exact_note = f"only {exact_count} historical trip(s) between these exact zones (too few to trust)" \
            if exact_count else "no historical trips between these exact zones"
        return {
            "distance": borough_avg, "tier": "borough_pair",
            "note": f"{exact_note} -- using the {pu_borough}\u2192{do_borough} borough-pair average "
                    f"from {borough_count:,} trips instead",
            "exact_count": exact_count, "exact_avg": exact_avg,
            "borough_count": borough_count, "borough_avg": borough_avg,
        }

    return {
        "distance": overall_avg, "tier": "citywide",
        "note": f"neither the exact zone pair ({exact_count} trips) nor the "
                f"{pu_borough}\u2192{do_borough} borough pair ({borough_count} trips) had enough "
                f"history -- falling back to the citywide average across ALL trips. "
                f"This is the least reliable tier; treat this prediction with caution.",
        "exact_count": exact_count, "exact_avg": exact_avg,
        "borough_count": borough_count, "borough_avg": borough_avg,
    }


def haversine_miles(lat1, lon1, lat2, lon2) -> float:
    """Great-circle (straight-line) distance in miles. This is NOT the same
    as road distance -- real routes are longer due to street grids, bridges,
    and tunnels -- so it's shown only as a rough sanity cross-check, never
    as the actual distance fed to the model."""
    import math
    R = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(min(1, math.sqrt(a)))


@st.cache_data(show_spinner=False)
def try_load_zone_centroids() -> pd.DataFrame:
    """BEST-EFFORT ONLY. Attempts to read the TLC's zone shapefile directly
    from CloudFront (streamed, never downloaded to disk) via DuckDB's
    spatial extension, and compute each zone's centroid lat/lon. This
    chains GDAL's /vsizip/ and /vsicurl/ virtual filesystems, which is a
    real but occasionally unreliable combination depending on the DuckDB/
    GDAL build -- so this MUST fail gracefully. If it doesn't work, the app
    continues fine without the geographic cross-check; the actual distance
    fix (three-tier fallback above) does not depend on this at all."""
    try:
        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")
        df = con.execute(
            """
            SELECT LocationID AS zone_id,
                   ST_Y(ST_Centroid(ST_Transform(geom, 'EPSG:2263', 'EPSG:4326'))) AS lat,
                   ST_X(ST_Centroid(ST_Transform(geom, 'EPSG:2263', 'EPSG:4326'))) AS lon
            FROM ST_Read('/vsizip/vsicurl/https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip')
            """
        ).df()
        return df
    except Exception:
        return pd.DataFrame(columns=["zone_id", "lat", "lon"])


@st.cache_data(show_spinner=False)
def load_hourly_series(hourly_demand_path: Path) -> pd.Series:
    """Citywide trip_count indexed by pickup_hour_ts -- the raw series the
    demand model's lag/rolling features are computed from."""
    if not hourly_demand_path.exists():
        return pd.Series(dtype="float64")
    df = duckdb.sql(
        f"SELECT pickup_hour_ts, trip_count FROM read_parquet('{hourly_demand_path.as_posix()}') ORDER BY 1"
    ).df()
    df["pickup_hour_ts"] = pd.to_datetime(df["pickup_hour_ts"])
    return df.set_index("pickup_hour_ts")["trip_count"]


@st.cache_data(show_spinner=False)
def load_zone_series(zone_hourly_demand_path: Path, zone_id: int) -> pd.Series:
    """trip_count for ONE zone, indexed by pickup_hour_ts."""
    if not zone_hourly_demand_path.exists():
        return pd.Series(dtype="float64")
    df = duckdb.sql(
        f"""
        SELECT pickup_hour_ts, trip_count
        FROM read_parquet('{zone_hourly_demand_path.as_posix()}')
        WHERE zone_id = {zone_id}
        ORDER BY 1
        """
    ).df()
    df["pickup_hour_ts"] = pd.to_datetime(df["pickup_hour_ts"])
    return df.set_index("pickup_hour_ts")["trip_count"]


def build_lag_known(series: pd.Series, target_ts: pd.Timestamp, lags=(1, 2, 3, 24, 168)) -> tuple[dict, list, float]:
    """Builds the lag_Xh / rolling_mean_24h / hour / dow / is_weekend / month
    features the demand models expect, by looking up actual historical
    values at target_ts minus each lag.

    Both build_features.py and build_features_hvfhv.py reindex the full
    date range with fill_value=0 before lagging (an hour with zero trips
    still gets a row, value 0) -- so any timestamp INSIDE the data's overall
    date range that's simply absent from `series` is treated as 0 trips,
    not as missing data. Only timestamps outside the data's date range are
    reported as genuinely unavailable.
    """
    known, missing = {}, []
    if series.empty:
        return known, ["(no historical data available at all)"], None

    global_min, global_max = series.index.min(), series.index.max()

    def lookup(ts: pd.Timestamp):
        if ts in series.index:
            return float(series.loc[ts])
        if global_min <= ts <= global_max:
            return 0.0  # hour genuinely had zero trips, not missing data
        return None

    for lag in lags:
        ts = target_ts - pd.Timedelta(hours=lag)
        val = lookup(ts)
        if val is None:
            missing.append(f"lag_{lag}h ({ts:%Y-%m-%d %H:%M} is outside the available data range)")
        else:
            known[f"lag_{lag}h"] = val

    # rolling_mean_24h in training = mean of the 24 hourly values from
    # (target-24h) through (target-1h) inclusive -- see shift(1).rolling(24)
    window_start = target_ts - pd.Timedelta(hours=24)
    window_hours = pd.date_range(window_start, target_ts - pd.Timedelta(hours=1), freq="h")
    window_vals = [lookup(t) for t in window_hours]
    available_vals = [v for v in window_vals if v is not None]
    if available_vals:
        known["rolling_mean_24h"] = sum(available_vals) / len(available_vals)
        if len(available_vals) < 24:
            missing.append(
                f"rolling_mean_24h (only {len(available_vals)}/24 hours available -- "
                f"near the edge of the data's date range)"
            )
    else:
        missing.append("rolling_mean_24h (no hours in the 24h window are in range)")

    known["hour"] = target_ts.hour
    dow = duckdb_dow(target_ts.date())
    known["dow"] = dow
    known["is_weekend"] = 1 if dow in (0, 6) else 0
    known["month"] = target_ts.month

    actual = lookup(target_ts)
    return known, missing, actual



def parse_metrics_txt(path: Path) -> dict:
    """Parses the MAE/RMSE/R2 + feature-importances .txt written by the
    train_duration_*.py scripts."""
    if not path.exists():
        return {}
    text = path.read_text()
    metrics, importances = {}, {}
    in_importances = False
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("Feature importances"):
            in_importances = True
            continue
        if not in_importances:
            if line.startswith("MAE:"):
                metrics["MAE"] = float(line.split(":")[1].strip().split()[0])
            elif line.startswith("RMSE:"):
                metrics["RMSE"] = float(line.split(":")[1].strip().split()[0])
            elif line.startswith("R2:"):
                metrics["R2"] = float(line.split(":")[1].strip())
        else:
            parts = line.rsplit(None, 1)
            if len(parts) == 2:
                name, val = parts
                try:
                    importances[name] = float(val)
                except ValueError:
                    pass
    metrics["importances"] = pd.Series(importances).sort_values(ascending=False)
    return metrics


@st.cache_data(show_spinner=False)
def sql_regression_metrics(predictions_path: Path) -> dict:
    """Computes MAE / RMSE / R2 directly in SQL against a predictions
    parquet (actual, predicted columns) -- used for Tasks 2 & 3, which save
    per-row predictions (Task 1 currently doesn't, see dashboard note)."""
    if not predictions_path.exists():
        return {}
    row = duckdb.sql(
        f"""
        WITH p AS (
            SELECT actual, predicted FROM read_parquet('{predictions_path.as_posix()}')
        ),
        stats AS (
            SELECT avg(actual) AS mean_actual FROM p
        )
        SELECT
            avg(abs(actual - predicted))                                   AS mae,
            sqrt(avg(power(actual - predicted, 2)))                        AS rmse,
            1 - sum(power(actual - predicted, 2))
                / sum(power(actual - (SELECT mean_actual FROM stats), 2))  AS r2
        FROM p
        """
    ).fetchone()
    return {"MAE": row[0], "RMSE": row[1], "R2": row[2]}


# ---------------------------------------------------------------------------
# Feature-vector construction for prediction
# ---------------------------------------------------------------------------

def duckdb_dow(d: date) -> int:
    """Convert Python's Monday=0..Sunday=6 to DuckDB's Sunday=0..Saturday=6,
    matching dayofweek(pickup_datetime) as used in every build_features*.py."""
    python_dow = d.weekday()  # Mon=0 .. Sun=6
    return (python_dow + 1) % 7  # Sun=0 .. Sat=6


def build_feature_row(model, known: dict) -> tuple[pd.DataFrame, list]:
    """Builds a single-row DataFrame matching model.feature_names_in_ (the
    columns the model was actually fit on). Any expected column not covered
    by `known` is filled with 0 and reported back so the UI can flag it."""
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
    else:
        cols = list(known.keys())

    row, defaulted = {}, []
    for c in cols:
        if c in known and known[c] is not None:
            row[c] = known[c]
        else:
            row[c] = 0
            defaulted.append(c)
    return pd.DataFrame([row])[cols], defaulted


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


# ---------------------------------------------------------------------------
# Page: Predict
# ---------------------------------------------------------------------------

def predict_duration_tab():
    st.caption(
        "Estimates trip duration using the trained GradientBoostingRegressor "
        "duration model for the selected vehicle type."
    )

    vehicle = st.selectbox("Vehicle type", list(PATHS.keys()), key="duration_vehicle")
    cfg = PATHS[vehicle]

    model = load_model(cfg["duration_model"])
    if model is None:
        st.error(
            f"Model file not found at `{cfg['duration_model']}`. "
            f"Run the training script for {vehicle} first."
        )
        return

    # Full zone lookup (pickup AND dropoff-only zones), NOT the pickup-only
    # zone_hourly_demand list -- Green Taxi in particular can legally drop
    # off in zones it can never pick up from (e.g. core Manhattan), and
    # those need to be selectable as a dropoff even though they'd never
    # appear as a pickup option in reality.
    zones = load_full_zone_lookup(cfg["zone_hourly_demand"])
    if zones.empty:
        st.error(
            f"No zone data found at `{cfg['zone_hourly_demand']}`. "
            f"Run build_features{'_hvfhv' if 'HVFHV' in vehicle else ''}.py first."
        )
        return

    zone_options = {
        f"{row.zone_name} ({row.borough})": row.zone_id for row in zones.itertuples()
    }

    min_date, max_date = load_data_date_range(cfg["hourly_demand"])

    with st.form("predict_duration_form"):
        c1, c2 = st.columns(2)
        with c1:
            pickup_label = st.selectbox("Pickup location", list(zone_options.keys()))
            pickup_date = st.date_input(
                "Pickup date", value=min_date, min_value=min_date, max_value=max_date,
                help=f"Restricted to your data's range: {min_date} to {max_date}.",
            )
        with c2:
            dropoff_label = st.selectbox(
                "Drop-off location", list(zone_options.keys()),
                index=min(1, len(zone_options) - 1),
            )
            pickup_time = st.time_input("Pickup time", value=dtime(hour=9, minute=0))

        passenger_count = None
        if vehicle == "Green Taxi":
            passenger_count = st.number_input(
                "Passenger count", min_value=1, max_value=6, value=1, step=1
            )
        else:
            st.caption(
                "ℹ️ HVFHV (Uber/Lyft) trip records don't include passenger count -- "
                "this field isn't used for this vehicle type."
            )
            company_label = st.selectbox("Company", list(HVFHS_CODES.keys()), index=1)
            is_shared = st.checkbox("Shared ride request")
            is_wav = st.checkbox("Wheelchair-accessible vehicle (WAV) request")

        submitted = st.form_submit_button("Predict duration", use_container_width=True)

    if not submitted:
        return

    zone_id_to_borough = dict(zip(zones["zone_id"], zones["borough"]))
    pu_id = zone_options[pickup_label]
    do_id = zone_options[dropoff_label]
    pu_borough = zone_id_to_borough.get(pu_id, "Unknown")
    do_borough = zone_id_to_borough.get(do_id, "Unknown")

    # Distance estimation -- this field dominates the model's predictions
    # (see the feature-importance chart on the Dashboard page), so getting
    # it right matters more than any other input here. Three tiers, most
    # specific available wins: exact zone pair -> borough pair -> citywide.
    # See resolve_trip_distance() for why a naive "exact pair, else
    # citywide" approach silently produced bad numbers for rare pairs.
    distance_col = cfg["distance_col"]
    zone_pair_stats = load_zone_pair_distance_stats(cfg["trip_features"], distance_col)
    borough_pair_stats = load_borough_pair_distance_stats(
        cfg["trip_features"], cfg["zone_hourly_demand"], distance_col
    )
    overall_avg = load_overall_avg_trip_distance(cfg["trip_features"], distance_col)

    resolution = resolve_trip_distance(
        pu_id, do_id, pu_borough, do_borough, zone_pair_stats, borough_pair_stats, overall_avg
    )
    trip_distance = resolution["distance"]

    # Optional geographic cross-check -- straight-line distance between zone
    # centroids, purely as a sanity check shown alongside the number
    # actually used. Never fed to the model, and silently skipped if the
    # remote shapefile read doesn't work in this environment.
    centroids = try_load_zone_centroids()
    geo_distance = None
    if not centroids.empty:
        pu_c = centroids[centroids["zone_id"] == pu_id]
        do_c = centroids[centroids["zone_id"] == do_id]
        if not pu_c.empty and not do_c.empty:
            geo_distance = haversine_miles(
                pu_c["lat"].iloc[0], pu_c["lon"].iloc[0], do_c["lat"].iloc[0], do_c["lon"].iloc[0]
            )

    dow = duckdb_dow(pickup_date)
    known = {
        "PULocationID": pu_id,
        "DOLocationID": do_id,
        distance_col: trip_distance,
        "pickup_hour": pickup_time.hour,
        "pickup_dow": dow,
        "is_weekend": 1 if dow in (0, 6) else 0,
        "pickup_month": pickup_date.month,
        "passenger_count": passenger_count,
    }
    if vehicle != "Green Taxi":
        known["hvfhs_license_num"] = HVFHS_CODES[company_label]
        known["is_shared_request"] = int(is_shared)
        known["is_wav_request"] = int(is_wav)

    X, defaulted = build_feature_row(model, known)
    pred_seconds = float(model.predict(X)[0])

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted trip duration", format_duration(pred_seconds))
    m2.metric("Distance used by model", f"{trip_distance:.2f} mi",
              help=f"Tier: {resolution['tier']}. {resolution['note']}")
    m3.metric("Average speed", f"{(trip_distance / (pred_seconds / 3600)):.1f} mph" if pred_seconds > 0 else "n/a")

    # Full transparency block -- exactly which tier was used and why,
    # plus the geographic cross-check when available, so a wrong or
    # suspicious-looking prediction is diagnosable from the UI itself.
    tier_labels = {"exact_pair": "🟢 Exact zone pair", "borough_pair": "🟡 Borough-pair average",
                   "citywide": "🔴 Citywide average (least reliable)"}
    st.markdown(f"**Distance source: {tier_labels[resolution['tier']]}**")
    st.caption(resolution["note"])

    detail_cols = st.columns(3 if geo_distance is not None else 2)
    with detail_cols[0]:
        st.metric("Exact-pair avg distance",
                  f"{resolution['exact_avg']:.2f} mi" if resolution.get("exact_avg") is not None else "no data",
                  help=f"{resolution.get('exact_count', 0):,} historical trip(s) on this exact route")
    with detail_cols[1]:
        st.metric("Borough-pair avg distance",
                  f"{resolution['borough_avg']:.2f} mi" if resolution.get("borough_avg") is not None else "n/a",
                  help=f"{resolution.get('borough_count', 0):,} historical trip(s) between "
                       f"{pu_borough} and {do_borough}")
    if geo_distance is not None:
        with detail_cols[2]:
            ratio = trip_distance / geo_distance if geo_distance > 0 else float("nan")
            st.metric("Straight-line (geo) distance", f"{geo_distance:.2f} mi",
                      help="Great-circle distance between zone centroids -- a road trip is "
                           "always longer than this, so expect the model's distance to exceed it.")
            if ratio < 0.9:
                st.caption("⚠️ Distance used is LESS than the straight-line distance -- worth double-checking.")

    if resolution["tier"] == "citywide":
        st.warning(
            "This route has essentially no historical data at either the zone-pair or "
            "borough-pair level, so the distance (and therefore the duration prediction) "
            "is a rough citywide estimate, not specific to this route. Treat this "
            "prediction as low-confidence."
        )

    # Model confidence, read live from the metrics file (not hardcoded) --
    # see validate_duration_models.py for the cross-validated version of
    # these same numbers.
    metrics = parse_metrics_txt(cfg["duration_metrics"])
    if metrics.get("MAE") is not None:
        st.caption(
            f"Model accuracy on held-out test data: typical error \u00b1{metrics['MAE']:.0f} seconds "
            f"({metrics['MAE']/60:.1f} min), R\u00b2 = {metrics['R2']:.3f}. This prediction carries "
            f"that same general uncertainty, plus any extra uncertainty from the distance tier above."
        )

    if defaulted:
        st.warning(
            "The model expects some features this form doesn't collect, so they "
            "were filled with a default value of 0 for this prediction: "
            + ", ".join(f"`{c}`" for c in defaulted)
            + ". If these matter for your use case, extend the form to collect them."
        )

    with st.expander("Show model input row"):
        st.dataframe(X, use_container_width=True)


def predict_citywide_demand_tab():
    st.caption(
        "Forecasts total pickups in a given hour, citywide, using the trained "
        "demand model. This model looks back at recent hours' actual trip "
        "counts (1/2/3 hours ago, same hour yesterday, same hour last week), "
        "so pick an hour that has enough history behind it in your data."
    )

    vehicle = st.selectbox("Vehicle type", list(PATHS.keys()), key="citywide_vehicle")
    cfg = PATHS[vehicle]

    model = load_model(cfg["demand_model"])
    if model is None:
        st.error(
            f"Model file not found at `{cfg['demand_model']}`. "
            f"Run the demand training script for {vehicle} first."
        )
        return

    series = load_hourly_series(cfg["hourly_demand"])
    if series.empty:
        st.error(f"No hourly demand data found at `{cfg['hourly_demand']}`.")
        return

    st.caption(
        f"Data available from **{series.index.min():%Y-%m-%d %H:%M}** "
        f"to **{series.index.max():%Y-%m-%d %H:%M}**."
    )

    with st.form("predict_citywide_form"):
        c1, c2 = st.columns(2)
        with c1:
            target_date = st.date_input(
                "Forecast date", value=series.index.max().date(),
                min_value=series.index.min().date(), max_value=series.index.max().date(),
                key="citywide_date",
            )
        with c2:
            target_time = st.time_input("Forecast hour", value=dtime(hour=9), key="citywide_time")
        submitted = st.form_submit_button("Forecast citywide demand", use_container_width=True)

    if not submitted:
        return

    target_ts = pd.Timestamp.combine(target_date, target_time).floor("h")
    known, missing, actual = build_lag_known(series, target_ts)

    X, defaulted = build_feature_row(model, known)
    pred = float(model.predict(X)[0])

    st.divider()
    if actual is not None:
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted trips (citywide)", f"{pred:,.0f}")
        m2.metric("Actual trips (citywide)", f"{actual:,.0f}")
        m3.metric("Difference", f"{pred - actual:+,.0f}")
    else:
        st.metric("Predicted trips (citywide)", f"{pred:,.0f}")
        st.caption("No actual trip count available for this hour (it's outside your data's range) -- nothing to compare against.")

    if missing:
        st.warning(
            "Some inputs couldn't be computed from history and were left at "
            "their default: " + "; ".join(missing)
        )
    if defaulted:
        st.warning(
            "The model expects some features not covered above, defaulted to 0: "
            + ", ".join(f"`{c}`" for c in defaulted)
        )

    with st.expander("Show model input row"):
        st.dataframe(X, use_container_width=True)


def predict_zone_demand_tab():
    st.caption(
        "Forecasts next-hour pickups for a SPECIFIC zone, using the trained "
        "zone-demand model (one model, zone as a feature). Like the citywide "
        "forecast, it looks back at that zone's recent actual trip counts."
    )

    vehicle = st.selectbox("Vehicle type", list(PATHS.keys()), key="zone_vehicle")
    cfg = PATHS[vehicle]

    model = load_model(cfg["zone_demand_model"])
    if model is None:
        st.error(
            f"Model file not found at `{cfg['zone_demand_model']}`. "
            f"Run the zone demand training script for {vehicle} first."
        )
        return

    zones = load_zone_list(cfg["zone_hourly_demand"])
    if zones.empty:
        st.error(f"No zone data found at `{cfg['zone_hourly_demand']}`.")
        return
    zone_options = {
        f"{row.zone_name} ({row.borough})": row.zone_id for row in zones.itertuples()
    }

    # Zone-specific data isn't loaded until after the form submits (it
    # depends on which zone is picked, and forms don't rerun on every
    # widget change), so use the citywide date range as a proxy bound --
    # zones are reindexed to the same overall date span at build time
    # (see build_zone_features), so this is accurate, not approximate.
    min_date, max_date = load_data_date_range(cfg["hourly_demand"])

    with st.form("predict_zone_form"):
        zone_label = st.selectbox("Zone", list(zone_options.keys()))
        c1, c2 = st.columns(2)
        with c1:
            target_date = st.date_input(
                "Forecast date", value=min_date, min_value=min_date, max_value=max_date,
                help=f"Restricted to your data's range: {min_date} to {max_date}.",
                key="zone_date",
            )
        with c2:
            target_time = st.time_input("Forecast hour", value=dtime(hour=9), key="zone_time")
        submitted = st.form_submit_button("Forecast zone demand", use_container_width=True)

    if not submitted:
        return

    zone_id = zone_options[zone_label]
    series = load_zone_series(cfg["zone_hourly_demand"], zone_id)
    if series.empty:
        st.error(f"No historical data found for zone `{zone_label}` -- can't build a forecast.")
        return

    st.caption(
        f"Data for this zone available from **{series.index.min():%Y-%m-%d %H:%M}** "
        f"to **{series.index.max():%Y-%m-%d %H:%M}**."
    )

    target_ts = pd.Timestamp.combine(target_date, target_time).floor("h")
    known, missing, actual = build_lag_known(series, target_ts)
    known["zone_id"] = zone_id

    X, defaulted = build_feature_row(model, known)
    pred = float(model.predict(X)[0])

    st.divider()
    if actual is not None:
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted trips (this zone)", f"{pred:,.1f}")
        m2.metric("Actual trips (this zone)", f"{actual:,.0f}")
        m3.metric("Difference", f"{pred - actual:+,.1f}")
    else:
        st.metric("Predicted trips (this zone)", f"{pred:,.1f}")
        st.caption("No actual trip count available for this hour (it's outside your data's range) -- nothing to compare against.")

    if missing:
        st.warning(
            "Some inputs couldn't be computed from history and were left at "
            "their default: " + "; ".join(missing)
        )
    if defaulted:
        st.warning(
            "The model expects some features not covered above, defaulted to 0: "
            + ", ".join(f"`{c}`" for c in defaulted)
        )

    with st.expander("Show model input row"):
        st.dataframe(X, use_container_width=True)


def page_predict():
    st.header("🔮 Predict")
    tab1, tab2, tab3 = st.tabs(["Trip Duration", "Citywide Demand", "Zone Demand"])
    with tab1:
        predict_duration_tab()
    with tab2:
        predict_citywide_demand_tab()
    with tab3:
        predict_zone_demand_tab()


# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------

def explanation_box(text: str):
    st.markdown(
        f"<div style='background:#F3F5F9;border-left:4px solid #4C72B0;"
        f"padding:0.75rem 1rem;border-radius:4px;font-size:0.92rem;'>{text}</div>",
        unsafe_allow_html=True,
    )


def dashboard_duration(cfg: dict, vehicle: str):
    st.subheader("Task 1 — Trip Duration Prediction")
    metrics = parse_metrics_txt(cfg["duration_metrics"])
    if not metrics:
        st.info("No metrics file found yet -- run the duration training script first.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{metrics['MAE']:.0f} sec")
    c2.metric("RMSE", f"{metrics['RMSE']:.0f} sec")
    c3.metric("R²", f"{metrics['R2']:.3f}")

    explanation_box(
        f"On trips the model never saw during training, its duration estimate is "
        f"off by about <b>{metrics['MAE']:.0f} seconds ({metrics['MAE']/60:.1f} min)</b> "
        f"on average. The R² of <b>{metrics['R2']:.3f}</b> means the model explains "
        f"about {metrics['R2']*100:.0f}% of the variation in trip duration across "
        f"{vehicle} trips -- the remainder comes from factors the data doesn't "
        f"capture (traffic conditions, exact route taken, etc.)."
    )

    importances = metrics.get("importances")
    if importances is not None and not importances.empty:
        fig = px.bar(
            importances.sort_values().reset_index(),
            x=0, y="index", orientation="h",
            labels={"0": "Importance", "index": "Feature"},
            title="What drives the duration prediction?",
        )
        st.plotly_chart(fig, use_container_width=True)
        top_feat = importances.index[0]
        explanation_box(
            f"<b>{top_feat}</b> is the single strongest driver of predicted duration "
            f"(importance {importances.iloc[0]:.1%}), which lines up with intuition -- "
            f"distance is the main determinant of how long a trip takes."
        )


def dashboard_demand(cfg: dict, vehicle: str):
    st.subheader("Task 2 — Citywide Hourly Demand Forecasting")
    metrics = sql_regression_metrics(cfg["demand_predictions"])
    if not metrics:
        st.info("No predictions file found yet -- run the demand training script first.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{metrics['MAE']:.0f} trips/hr")
    c2.metric("RMSE", f"{metrics['RMSE']:.0f} trips/hr")
    c3.metric("R²", f"{metrics['R2']:.3f}")

    df = duckdb.sql(
        f"SELECT * FROM read_parquet('{cfg['demand_predictions'].as_posix()}')"
    ).df()
    ts_col = df.columns[0] if df.columns[0] not in ("actual", "predicted") else None
    if ts_col is None and "pickup_hour_ts" not in df.columns:
        df = df.reset_index().rename(columns={df.columns[0]: "pickup_hour_ts"})
    df = df.sort_values(df.columns[0] if "pickup_hour_ts" not in df.columns else "pickup_hour_ts")
    x_col = "pickup_hour_ts" if "pickup_hour_ts" in df.columns else df.columns[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x_col], y=df["actual"], name="Actual", line=dict(color="#333333")))
    fig.add_trace(go.Scatter(x=df[x_col], y=df["predicted"], name="Predicted", line=dict(color="#DD8452")))
    fig.update_layout(title="Citywide demand -- holdout period", yaxis_title="Trips / hour")
    st.plotly_chart(fig, use_container_width=True)

    explanation_box(
        f"This chart compares actual citywide trip volume (black) against the "
        f"model's next-hour forecast (orange) over trips held out from training. "
        f"An R² of <b>{metrics['R2']:.3f}</b> indicates the model tracks the daily "
        f"demand rhythm (morning/evening peaks, overnight lull) very closely for "
        f"{vehicle}."
    )

    resid = duckdb.sql(
        f"""
        SELECT hour(pickup_hour_ts) AS hr, avg(predicted - actual) AS mean_resid
        FROM read_parquet('{cfg['demand_predictions'].as_posix()}')
        GROUP BY 1 ORDER BY 1
        """
    ).df() if x_col == "pickup_hour_ts" else None

    if resid is not None and not resid.empty:
        fig2 = px.bar(
            resid, x="hr", y="mean_resid",
            color=resid["mean_resid"] >= 0,
            color_discrete_map={True: "#DD8452", False: "#4C72B0"},
            labels={"hr": "Hour of day", "mean_resid": "Mean error (predicted - actual)"},
            title="Where does the model over/under-predict?",
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        explanation_box(
            "Bars above zero mean the model tends to over-predict demand at that "
            "hour; bars below zero mean it under-predicts. Bars close to zero "
            "across the board indicate no strong time-of-day bias in the errors."
        )


def dashboard_zone_demand(cfg: dict, vehicle: str):
    st.subheader("Task 3 — Zone-Level Demand Forecasting")
    metrics = sql_regression_metrics(cfg["zone_demand_predictions"])
    if not metrics:
        st.info("No predictions file found yet -- run the zone demand training script first.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{metrics['MAE']:.2f} trips/zone/hr")
    c2.metric("RMSE", f"{metrics['RMSE']:.2f} trips/zone/hr")
    c3.metric("R²", f"{metrics['R2']:.3f}")

    explanation_box(
        f"This model predicts next-hour demand for every pickup zone individually "
        f"(one model, zone as a feature, rather than 264 separate models). "
        f"R² of <b>{metrics['R2']:.3f}</b> shows it generalizes well down to the "
        f"zone level, not just citywide totals."
    )

    preds_path = cfg["zone_demand_predictions"]
    zone_path = cfg["zone_hourly_demand"]
    if zone_path.exists():
        top = duckdb.sql(
            f"""
            SELECT p.zone_id, any_value(z.zone_name) AS zone_name,
                   any_value(z.borough) AS borough, sum(p.actual) AS total_actual
            FROM read_parquet('{preds_path.as_posix()}') p
            LEFT JOIN (
                SELECT DISTINCT zone_id, zone_name, borough
                FROM read_parquet('{zone_path.as_posix()}')
            ) z ON p.zone_id = z.zone_id
            GROUP BY p.zone_id
            ORDER BY total_actual DESC
            LIMIT 15
            """
        ).df()
        top["label"] = top["zone_name"].fillna(top["zone_id"].astype(str)) + " (" + top["borough"].fillna("?") + ")"
        fig = px.bar(
            top.sort_values("total_actual"), x="total_actual", y="label", orientation="h",
            title="Top 15 busiest zones (holdout period)",
            labels={"total_actual": "Total actual trips", "label": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

        by_borough = duckdb.sql(
            f"""
            SELECT any_value(z.borough) AS borough, avg(abs(p.actual - p.predicted)) AS mae
            FROM read_parquet('{preds_path.as_posix()}') p
            LEFT JOIN (
                SELECT DISTINCT zone_id, borough FROM read_parquet('{zone_path.as_posix()}')
            ) z ON p.zone_id = z.zone_id
            GROUP BY p.zone_id
            """
        ).df()
        if "borough" in by_borough.columns:
            by_borough_agg = (
                duckdb.sql("SELECT borough, avg(mae) AS mae FROM by_borough GROUP BY borough ORDER BY mae")
                .df()
            )
            fig2 = px.bar(
                by_borough_agg, x="mae", y="borough", orientation="h",
                title="Prediction error (MAE) by borough",
                labels={"mae": "MAE (trips/zone/hr)", "borough": ""},
            )
            st.plotly_chart(fig2, use_container_width=True)
            explanation_box(
                "Boroughs with higher average trip volume (e.g. Manhattan) "
                "naturally show a higher absolute error even when the model's "
                "relative accuracy is similar -- more trips means more room for "
                "the prediction to miss by a few."
            )


def page_dashboard():
    st.header("📊 Model Performance Dashboard")
    st.caption(
        "Live metrics computed via SQL against the saved model outputs -- "
        "suitable for a manager/stakeholder walkthrough."
    )

    tabs = st.tabs(["Executive Summary"] + list(PATHS.keys()))

    with tabs[0]:
        st.subheader("All 6 models at a glance")
        rows = []
        for vehicle, cfg in PATHS.items():
            dur = parse_metrics_txt(cfg["duration_metrics"])
            dem = sql_regression_metrics(cfg["demand_predictions"])
            zon = sql_regression_metrics(cfg["zone_demand_predictions"])
            if dur:
                rows.append({"Vehicle": vehicle, "Task": "Duration", "MAE": dur.get("MAE"), "R²": dur.get("R2")})
            if dem:
                rows.append({"Vehicle": vehicle, "Task": "Citywide Demand", "MAE": dem.get("MAE"), "R²": dem.get("R2")})
            if zon:
                rows.append({"Vehicle": vehicle, "Task": "Zone Demand", "MAE": zon.get("MAE"), "R²": zon.get("R2")})

        if rows:
            summary = pd.DataFrame(rows)
            st.dataframe(
                summary.style.format({"MAE": "{:.2f}", "R²": "{:.3f}"}),
                use_container_width=True, hide_index=True,
            )
            fig = px.bar(
                summary, x="Task", y="R²", color="Vehicle", barmode="group",
                title="Model accuracy (R²) by task and vehicle type",
                range_y=[0, 1],
            )
            st.plotly_chart(fig, use_container_width=True)
            explanation_box(
                "Higher R² means the model explains more of the real-world "
                "variation -- 1.0 would be a perfect predictor. Demand forecasting "
                "models score highest (predictable daily/weekly rhythms); "
                "trip duration is inherently noisier (traffic, routing) so scores "
                "there, while still strong, are comparatively lower."
            )
        else:
            st.info("No trained models found yet. Run the training scripts, then reload this page.")

    for tab, (vehicle, cfg) in zip(tabs[1:], PATHS.items()):
        with tab:
            dashboard_duration(cfg, vehicle)
            st.divider()
            dashboard_demand(cfg, vehicle)
            st.divider()
            dashboard_zone_demand(cfg, vehicle)


# ---------------------------------------------------------------------------
# App shell
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="NYC Taxi & HVFHV Analytics",
        page_icon="🚕",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; }
        [data-testid="stMetricValue"] { font-size: 1.6rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.title("🚕 NYC Taxi Analytics")
    page = st.sidebar.radio("Go to", ["🔮 Predict", "📊 Model Performance Dashboard"])
    st.sidebar.divider()
    st.sidebar.caption(
        "Green Taxi: yellow-cab-style street-hail data.\n\n"
        "HVFHV: Uber / Lyft / Via / Juno (High Volume For-Hire Vehicle)."
    )

    if page == "🔮 Predict":
        page_predict()
    else:
        page_dashboard()


if __name__ == "__main__":
    main()