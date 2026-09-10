"""
NYC Taxi Analytics Dashboard
============================

Two pages:

1. Predict
   - Trip Duration
   - Citywide Demand
   - Zone Demand

2. Model Performance Dashboard
   - Yellow Taxi: 3 models
   - FHV: 3 models
   - Green Taxi: 3 models
   - HVFHV: 3 models

Run:
    streamlit run src/dashboard_app.py
"""

from pathlib import Path
from datetime import date, time as dtime

import duckdb
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "outputs"


PATHS = {
    # ------------------------------------------------------------------------
    # YELLOW TAXI
    # ------------------------------------------------------------------------
    "Yellow Taxi": {
        "trip_features": PROCESSED_DIR / "yellow_trip_features.parquet",
        "hourly_demand": PROCESSED_DIR / "yellow_hourly_demand.parquet",
        "zone_hourly_demand": PROCESSED_DIR / "yellow_zone_hourly_demand.parquet",
        "duration_model": OUTPUTS_DIR / "yellow_duration_model.joblib",
        "duration_metrics": OUTPUTS_DIR / "yellow_duration_model_metrics.txt",
        "duration_predictions": OUTPUTS_DIR / "yellow_duration_predictions.parquet",
        "demand_model": OUTPUTS_DIR / "yellow_demand_model.joblib",
        "demand_predictions": OUTPUTS_DIR / "yellow_demand_predictions.parquet",
        "zone_demand_model": OUTPUTS_DIR / "yellow_zone_demand_model.joblib",
        "zone_demand_predictions": OUTPUTS_DIR / "yellow_zone_demand_predictions.parquet",
        "duration_type": "yellow",
        "distance_col": "trip_distance",
    },

    # ------------------------------------------------------------------------
    # FHV
    # ------------------------------------------------------------------------
    "FHV": {
        "trip_features": PROCESSED_DIR / "fhv_trip_features.parquet",
        "hourly_demand": PROCESSED_DIR / "fhv_hourly_demand.parquet",
        "zone_hourly_demand": PROCESSED_DIR / "fhv_zone_hourly_demand.parquet",
        "duration_model": OUTPUTS_DIR / "fhv_duration_model.joblib",
        "duration_metrics": OUTPUTS_DIR / "fhv_duration_model_metrics.txt",
        "duration_predictions": OUTPUTS_DIR / "fhv_duration_predictions.parquet",
        "demand_model": OUTPUTS_DIR / "fhv_demand_model.joblib",
        "demand_predictions": OUTPUTS_DIR / "fhv_demand_predictions.parquet",
        "zone_demand_model": OUTPUTS_DIR / "fhv_zone_demand_model.joblib",
        "zone_demand_predictions": OUTPUTS_DIR / "fhv_zone_demand_predictions.parquet",
        "duration_type": "fhv",
        "distance_col": "trip_distance",
    },

    # ------------------------------------------------------------------------
    # GREEN TAXI -- Vaihali's trained models
    # ------------------------------------------------------------------------
    "Green Taxi": {
        "trip_features": PROCESSED_DIR / "trip_features.parquet",
        "hourly_demand": PROCESSED_DIR / "hourly_demand.parquet",
        "zone_hourly_demand": PROCESSED_DIR / "zone_hourly_demand.parquet",
        "duration_model": OUTPUTS_DIR / "duration_model.joblib",
        "duration_metrics": OUTPUTS_DIR / "duration_model_metrics.txt",
        "duration_predictions": OUTPUTS_DIR / "duration_predictions.parquet",
        "demand_model": OUTPUTS_DIR / "demand_model.joblib",
        "demand_predictions": OUTPUTS_DIR / "demand_predictions.parquet",
        "zone_demand_model": OUTPUTS_DIR / "zone_demand_model.joblib",
        "zone_demand_predictions": OUTPUTS_DIR / "zone_demand_predictions.parquet",
        "duration_type": "green",
        "distance_col": "trip_distance",
    },

    # ------------------------------------------------------------------------
    # HVFHV (Uber/Lyft/Via/Juno) -- Vaihali's trained models
    # ------------------------------------------------------------------------
    "HVFHV (Uber/Lyft)": {
        "trip_features": PROCESSED_DIR / "hvfhv_trip_features.parquet",
        "hourly_demand": PROCESSED_DIR / "hvfhv_hourly_demand.parquet",
        "zone_hourly_demand": PROCESSED_DIR / "hvfhv_zone_hourly_demand.parquet",
        "duration_model": OUTPUTS_DIR / "hvfhv_duration_model.joblib",
        "duration_metrics": OUTPUTS_DIR / "hvfhv_duration_model_metrics.txt",
        "duration_predictions": OUTPUTS_DIR / "hvfhv_duration_predictions.parquet",
        "demand_model": OUTPUTS_DIR / "hvfhv_demand_model.joblib",
        "demand_predictions": OUTPUTS_DIR / "hvfhv_demand_predictions.parquet",
        "zone_demand_model": OUTPUTS_DIR / "hvfhv_zone_demand_model.joblib",
        "zone_demand_predictions": OUTPUTS_DIR / "hvfhv_zone_demand_predictions.parquet",
        "duration_type": "hvfhv",
        "distance_col": "trip_miles",
    },
}

# ============================================================================
# CACHED LOADERS
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_model(path: Path):

    if not path.exists():
        return None

    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_zone_list(zone_path: Path):

    if not zone_path.exists():
        return pd.DataFrame(
            columns=["zone_id", "zone_name", "borough"]
        )

    return duckdb.sql(
        f"""
        SELECT DISTINCT
            zone_id,
            zone_name,
            borough
        FROM read_parquet('{zone_path.as_posix()}')
        WHERE zone_name IS NOT NULL
        ORDER BY borough, zone_name
        """
    ).df()


@st.cache_data(show_spinner=False)
def load_hourly_series(hourly_path: Path):

    if not hourly_path.exists():
        return pd.Series(dtype="float64")

    df = duckdb.sql(
        f"""
        SELECT
            pickup_hour_ts,
            trip_count
        FROM read_parquet('{hourly_path.as_posix()}')
        ORDER BY pickup_hour_ts
        """
    ).df()

    df["pickup_hour_ts"] = pd.to_datetime(
        df["pickup_hour_ts"]
    )

    return df.set_index("pickup_hour_ts")["trip_count"]


@st.cache_data(show_spinner=False)
def load_zone_series(zone_path: Path, zone_id: int):

    if not zone_path.exists():
        return pd.Series(dtype="float64")

    df = duckdb.sql(
        f"""
        SELECT
            pickup_hour_ts,
            trip_count
        FROM read_parquet('{zone_path.as_posix()}')
        WHERE zone_id = {int(zone_id)}
        ORDER BY pickup_hour_ts
        """
    ).df()

    if df.empty:
        return pd.Series(dtype="float64")

    df["pickup_hour_ts"] = pd.to_datetime(
        df["pickup_hour_ts"]
    )

    return df.set_index("pickup_hour_ts")["trip_count"]


TAXI_ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
MIN_ZONE_PAIR_TRIPS = 5


@st.cache_data(show_spinner=False)
def load_full_zone_lookup(zone_hourly_demand_path: Path) -> pd.DataFrame:
    """Load all official TLC zones for pickup/drop-off selection."""
    if not zone_hourly_demand_path.exists():
        return pd.DataFrame(columns=["zone_id", "zone_name", "borough"])
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


@st.cache_data(show_spinner=False)
def load_zone_pair_distance_stats(
    trip_features_path: Path, distance_col: str
) -> pd.DataFrame:
    if not trip_features_path.exists():
        return pd.DataFrame(
            columns=["PULocationID", "DOLocationID", "avg_trip_distance", "trip_count"]
        )
    return duckdb.sql(
        f"""
        SELECT PULocationID, DOLocationID,
               avg({distance_col}) AS avg_trip_distance,
               count(*) AS trip_count
        FROM read_parquet('{trip_features_path.as_posix()}')
        WHERE {distance_col} > 0
        GROUP BY 1, 2
        """
    ).df()


@st.cache_data(show_spinner=False)
def load_borough_pair_distance_stats(
    trip_features_path: Path,
    zone_hourly_demand_path: Path,
    distance_col: str,
) -> pd.DataFrame:
    if not trip_features_path.exists() or not zone_hourly_demand_path.exists():
        return pd.DataFrame(
            columns=[
                "pickup_borough", "dropoff_borough",
                "avg_trip_distance", "trip_count"
            ]
        )
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
        WHERE t.{distance_col} > 0
        GROUP BY 1, 2
        """
    ).df()


@st.cache_data(show_spinner=False)
def load_overall_avg_trip_distance(
    trip_features_path: Path, distance_col: str
) -> float:
    if not trip_features_path.exists():
        return 3.0
    result = duckdb.sql(
        f"SELECT avg({distance_col}) FROM read_parquet('{trip_features_path.as_posix()}')"
    ).fetchone()
    return float(result[0]) if result and result[0] is not None else 3.0


def resolve_trip_distance(
    pu_id: int,
    do_id: int,
    pu_borough: str,
    do_borough: str,
    zone_pair_stats: pd.DataFrame,
    borough_pair_stats: pd.DataFrame,
    overall_avg: float,
) -> dict:
    """Resolve distance as exact pair -> borough pair -> citywide average."""
    exact = zone_pair_stats[
        (zone_pair_stats["PULocationID"] == pu_id)
        & (zone_pair_stats["DOLocationID"] == do_id)
    ]
    exact_count = int(exact["trip_count"].iloc[0]) if not exact.empty else 0
    exact_avg = float(exact["avg_trip_distance"].iloc[0]) if not exact.empty else None

    if exact_count >= MIN_ZONE_PAIR_TRIPS:
        return {
            "distance": exact_avg,
            "tier": "exact_pair",
            "note": f"based on {exact_count:,} historical trips between these exact zones",
            "exact_count": exact_count,
            "exact_avg": exact_avg,
        }

    borough_match = borough_pair_stats[
        (borough_pair_stats["pickup_borough"] == pu_borough)
        & (borough_pair_stats["dropoff_borough"] == do_borough)
    ]
    borough_count = (
        int(borough_match["trip_count"].iloc[0])
        if not borough_match.empty else 0
    )
    borough_avg = (
        float(borough_match["avg_trip_distance"].iloc[0])
        if not borough_match.empty else None
    )

    if borough_count >= MIN_ZONE_PAIR_TRIPS:
        exact_note = (
            f"only {exact_count} historical trip(s) between these exact zones (too few to trust)"
            if exact_count
            else "no historical trips between these exact zones"
        )
        return {
            "distance": borough_avg,
            "tier": "borough_pair",
            "note": (
                f"{exact_note} -- using the {pu_borough}→{do_borough} "
                f"borough-pair average from {borough_count:,} trips instead"
            ),
            "exact_count": exact_count,
            "exact_avg": exact_avg,
            "borough_count": borough_count,
            "borough_avg": borough_avg,
        }

    return {
        "distance": overall_avg,
        "tier": "citywide",
        "note": (
            f"neither the exact zone pair ({exact_count} trips) nor the "
            f"{pu_borough}→{do_borough} borough pair ({borough_count} trips) "
            "had enough history -- using the citywide average"
        ),
        "exact_count": exact_count,
        "exact_avg": exact_avg,
        "borough_count": borough_count,
        "borough_avg": borough_avg,
    }


@st.cache_data(show_spinner=False)
def load_data_date_range(hourly_demand_path: Path) -> tuple:
    if not hourly_demand_path.exists():
        return date(2025, 1, 1), date(2025, 12, 31)
    row = duckdb.sql(
        f"""
        SELECT min(pickup_hour_ts), max(pickup_hour_ts)
        FROM read_parquet('{hourly_demand_path.as_posix()}')
        """
    ).fetchone()
    if row and row[0] is not None:
        return pd.Timestamp(row[0]).date(), pd.Timestamp(row[1]).date()
    return date(2025, 1, 1), date(2025, 12, 31)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def duckdb_dow(d: date) -> int:
    """
    Python:
        Monday = 0
        Sunday = 6

    DuckDB:
        Sunday = 0
        Saturday = 6
    """

    return (d.weekday() + 1) % 7


def format_duration_minutes(seconds: float) -> str:
    """Display a duration consistently in minutes."""
    minutes = max(0.0, float(seconds)) / 60.0
    return f"{minutes:.1f} minutes"


def build_feature_row(model, known: dict):

    if hasattr(model, "feature_names_in_"):
        columns = list(model.feature_names_in_)
    else:
        columns = list(known.keys())

    row = {}
    defaulted = []

    for column in columns:

        if column in known and known[column] is not None:

            row[column] = known[column]

        else:

            row[column] = 0
            defaulted.append(column)

    return pd.DataFrame([row])[columns], defaulted


# ============================================================================
# DEMAND FEATURE CREATION
# ============================================================================

def build_lag_known(
    series: pd.Series,
    target_ts: pd.Timestamp,
    lags=(1, 2, 3, 24, 168),
):

    known = {}
    missing = []

    if series.empty:

        return known, ["No historical data available"], None

    global_min = series.index.min()
    global_max = series.index.max()

    def lookup(ts):

        if ts in series.index:

            return float(series.loc[ts])

        if global_min <= ts <= global_max:

            return 0.0

        return None

    # ------------------------------------------------------------------------
    # LAG FEATURES
    # ------------------------------------------------------------------------

    for lag in lags:

        ts = target_ts - pd.Timedelta(hours=lag)

        value = lookup(ts)

        if value is None:

            missing.append(
                f"lag_{lag}h unavailable"
            )

        else:

            known[f"lag_{lag}h"] = value

    # ------------------------------------------------------------------------
    # ROLLING 24 HOUR MEAN
    # ------------------------------------------------------------------------

    window_start = target_ts - pd.Timedelta(hours=24)

    window_hours = pd.date_range(
        window_start,
        target_ts - pd.Timedelta(hours=1),
        freq="h",
    )

    window_values = [
        lookup(ts)
        for ts in window_hours
    ]

    available_values = [
        value
        for value in window_values
        if value is not None
    ]

    if available_values:

        known["rolling_mean_24h"] = (
            sum(available_values)
            / len(available_values)
        )

    else:

        missing.append(
            "rolling_mean_24h unavailable"
        )

    # ------------------------------------------------------------------------
    # CALENDAR FEATURES
    # ------------------------------------------------------------------------

    known["hour"] = target_ts.hour

    dow = duckdb_dow(target_ts.date())

    known["dow"] = dow

    known["is_weekend"] = (
        1 if dow in (0, 6) else 0
    )

    known["month"] = target_ts.month

    # Actual target value if available
    actual = lookup(target_ts)

    return known, missing, actual


# ============================================================================
# METRICS
# ============================================================================

def parse_metrics_txt(path: Path):

    if not path.exists():
        return {}

    text = path.read_text()

    metrics = {}
    importances = {}

    in_importances = False

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        if line.startswith("Feature importances"):

            in_importances = True
            continue

        if not in_importances:

            # Parse only well-formed metric lines. Using split(":", 1)
            # prevents the dashboard from crashing if a metric line contains
            # additional colons or is malformed.
            if line.startswith("MAE:"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    try:
                        metrics["MAE"] = float(
                            parts[1].strip().split()[0]
                        )
                    except (ValueError, IndexError):
                        pass

            elif line.startswith("RMSE:"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    try:
                        metrics["RMSE"] = float(
                            parts[1].strip().split()[0]
                        )
                    except (ValueError, IndexError):
                        pass

            elif line.startswith("R2:"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    try:
                        metrics["R2"] = float(
                            parts[1].strip().split()[0]
                        )
                    except (ValueError, IndexError):
                        pass

            elif line.startswith("R²:"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    try:
                        metrics["R2"] = float(
                            parts[1].strip().split()[0]
                        )
                    except (ValueError, IndexError):
                        pass

        else:

            parts = line.rsplit(None, 1)

            if len(parts) == 2:

                name, value = parts

                try:

                    importances[name] = float(value)

                except ValueError:

                    pass

    metrics["importances"] = (
        pd.Series(importances)
        .sort_values(ascending=False)
    )

    return metrics


@st.cache_data(show_spinner=False)
def sql_regression_metrics(predictions_path: Path):

    if not predictions_path.exists():
        return {}

    row = duckdb.sql(
        f"""
        WITH p AS (
            SELECT
                actual,
                predicted
            FROM read_parquet(
                '{predictions_path.as_posix()}'
            )
        ),

        stats AS (
            SELECT
                avg(actual) AS mean_actual
            FROM p
        )

        SELECT

            avg(
                abs(actual - predicted)
            ) AS mae,

            sqrt(
                avg(
                    power(actual - predicted, 2)
                )
            ) AS rmse,

            1 -

            sum(
                power(actual - predicted, 2)
            )

            /

            sum(
                power(
                    actual -
                    (SELECT mean_actual FROM stats),
                    2
                )
            ) AS r2

        FROM p
        """
    ).fetchone()

    if row is None:
        return {}

    return {
        "MAE": row[0],
        "RMSE": row[1],
        "R2": row[2],
    }


# ============================================================================
# PAGE 1 — PREDICT
# ============================================================================

# ----------------------------------------------------------------------------
# TASK 1 — TRIP DURATION
# ----------------------------------------------------------------------------

def get_duration_reference(
    vehicle: str,
    pu_id: int,
    do_id: int,
    pickup_date: date,
    pickup_hour: int,
):
    """Return the route distance used for models that require distance.

    Green and Yellow use their own historical trip data. HVFHV uses its own
    trip_miles when the model expects it. FHV has no distance field, so its
    display-only distance is estimated from Yellow Taxi history.
    """
    cfg = PATHS[vehicle]

    # FHV has no trip-distance field.
    source_vehicle = "Yellow Taxi" if vehicle == "FHV" else vehicle
    source_cfg = PATHS[source_vehicle]
    trip_features = source_cfg["trip_features"]
    distance_col = source_cfg["distance_col"]

    if not trip_features.exists():
        return None

    # Use the same robust three-tier route resolution used by Vaihali's
    # Green/HVFHV dashboard.
    try:
        zone_pair_stats = load_zone_pair_distance_stats(trip_features, distance_col)
        borough_pair_stats = load_borough_pair_distance_stats(
            trip_features,
            source_cfg["zone_hourly_demand"],
            distance_col,
        )

        zones = load_full_zone_lookup(source_cfg["zone_hourly_demand"])
        borough_map = dict(zip(zones["zone_id"], zones["borough"]))
        pu_borough = borough_map.get(pu_id, "Unknown")
        do_borough = borough_map.get(do_id, "Unknown")

        overall_avg = load_overall_avg_trip_distance(trip_features, distance_col)
        resolution = resolve_trip_distance(
            pu_id,
            do_id,
            pu_borough,
            do_borough,
            zone_pair_stats,
            borough_pair_stats,
            overall_avg,
        )
        return {
            "distance": float(resolution["distance"]),
            "source": resolution["tier"],
            "note": resolution["note"],
            "resolution": resolution,
            "distance_col": distance_col,
        }
    except Exception:
        return None


def predict_duration_tab():
    st.subheader("Trip Duration Prediction")
    st.caption(
        "Predict the expected trip duration using the trained model for the "
        "selected vehicle type."
    )

    vehicle = st.selectbox(
        "Vehicle type",
        list(PATHS.keys()),
        key="duration_vehicle",
    )
    cfg = PATHS[vehicle]
    model = load_model(cfg["duration_model"])

    if model is None:
        st.error(f"Model not found:\n\n{cfg['duration_model']}")
        return

    # Full TLC lookup so legitimate drop-off-only zones remain selectable.
    zones = load_full_zone_lookup(cfg["zone_hourly_demand"])
    if zones.empty:
        st.error(
            "Zone data was not found. Please make sure the feature-building "
            "step has been completed."
        )
        return

    zone_options = {}
    for row in zones.itertuples():
        label = f"{row.zone_name} ({row.borough}) -- #{row.zone_id}"
        zone_options[label] = int(row.zone_id)

    min_date, max_date = load_data_date_range(cfg["hourly_demand"])

    with st.form("duration_prediction_form"):
        c1, c2 = st.columns(2)

        with c1:
            pickup_label = st.selectbox(
                "Pickup zone",
                list(zone_options.keys()),
            )
            pickup_date = st.date_input(
                "Pickup date",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
            )

        with c2:
            dropoff_label = st.selectbox(
                "Drop-off zone",
                list(zone_options.keys()),
                index=min(1, len(zone_options) - 1),
            )
            pickup_time = st.time_input(
                "Pickup time",
                value=dtime(hour=9, minute=0),
            )

        submitted = st.form_submit_button(
            "Predict trip duration",
            use_container_width=True,
        )

    if not submitted:
        return

    pu_id = zone_options[pickup_label]
    do_id = zone_options[dropoff_label]
    same_zone = pu_id == do_id
    dow = duckdb_dow(pickup_date)
    pickup_hour = pickup_time.hour

    reference = get_duration_reference(
        vehicle,
        pu_id,
        do_id,
        pickup_date,
        pickup_hour,
    )

    # Build only features appropriate to the selected model. Any additional
    # model-specific feature not collected by this simple form is safely
    # defaulted by build_feature_row().
    known = {
        "PULocationID": pu_id,
        "DOLocationID": do_id,
        "PUlocationID": pu_id,
        "DOlocationID": do_id,
        "pickup_hour": pickup_hour,
        "pickup_dow": dow,
        "is_weekend": 1 if dow in (0, 6) else 0,
        "pickup_month": pickup_date.month,
        "passenger_count": 1,
        "RatecodeID": 1,
        "payment_type": 1,
        "trip_type": 1,
        "SR_Flag": 0,
        "hvfhs_license_num": 0,
        "is_shared_request": 0,
        "is_wav_request": 0,
    }

    # Distance is a model input for Yellow/Green. For FHV it is display-only.
    if reference is not None:
        known[reference["distance_col"]] = reference["distance"]

    X, defaulted = build_feature_row(model, known)

    try:
        prediction_seconds = max(
            0.0,
            float(model.predict(X)[0]),
        )
    except Exception as exc:
        st.error(
            "The selected model could not make a prediction from the "
            f"constructed feature row: {exc}"
        )
        with st.expander("Show model input"):
            st.dataframe(X, use_container_width=True)
        return

    st.divider()

    # Keep the blue same-zone message from the original UI.
    if same_zone:
        st.info(
            f"📍 **Same zone trip** — pickup and drop-off are both "
            f"**{pickup_label}**. The model is predicting a trip that "
            f"starts and ends within the same taxi zone."
        )
    else:
        st.caption(
            f"Route type: **Different zones** — "
            f"{pickup_label} → {dropoff_label}"
        )

    c1, c2 = st.columns(2)
    c1.metric(
        "Predicted duration",
        format_duration_minutes(prediction_seconds),
    )

    if reference is not None:
        c2.metric(
            "Trip distance",
            f"{reference['distance']:.1f} miles",
        )
        if vehicle == "FHV":
            st.caption(
                "Distance is shown for reference only because FHV trip records "
                "do not contain trip distance."
            )
        elif reference["source"] != "exact_pair":
            st.caption(
                f"Distance estimated using {reference['source'].replace('_', ' ')} "
                "historical data."
            )
    else:
        c2.metric("Trip distance", "Not available")

    if defaulted:
        with st.expander("Show model input"):
            st.dataframe(X, use_container_width=True)
    else:
        with st.expander("Show model input"):
            st.dataframe(X, use_container_width=True)


# ----------------------------------------------------------------------------
# TASK 2 — CITYWIDE DEMAND
# ----------------------------------------------------------------------------

def predict_citywide_demand_tab():

    st.subheader(
        "Citywide Hourly Demand Prediction"
    )

    st.caption(
        "Forecast the number of trips for a selected hour "
        "using recent historical demand."
    )

    vehicle = st.selectbox(
        "Vehicle type",
        list(PATHS.keys()),
        key="citywide_vehicle",
    )

    cfg = PATHS[vehicle]

    model = load_model(
        cfg["demand_model"]
    )

    if model is None:

        st.error(
            f"Demand model not found:\n\n"
            f"{cfg['demand_model']}"
        )

        return

    series = load_hourly_series(
        cfg["hourly_demand"]
    )

    if series.empty:

        st.error(
            "Hourly demand data was not found."
        )

        return

    st.caption(
        f"Available historical data: "
        f"**{series.index.min():%Y-%m-%d %H:%M}** "
        f"to "
        f"**{series.index.max():%Y-%m-%d %H:%M}**"
    )

    # ------------------------------------------------------------------------
    # INPUT
    # ------------------------------------------------------------------------

    with st.form(
        "citywide_prediction_form"
    ):

        c1, c2 = st.columns(2)

        with c1:

            target_date = st.date_input(
                "Forecast date",
                value=series.index.max().date(),
                key="citywide_date",
            )

        with c2:

            target_time = st.time_input(
                "Forecast hour",
                value=dtime(hour=9),
                key="citywide_time",
            )

        submitted = st.form_submit_button(
            "Predict citywide demand",
            use_container_width=True,
        )

    if not submitted:
        return

    target_ts = pd.Timestamp.combine(
        target_date,
        target_time,
    ).floor("h")

    known, missing, actual = build_lag_known(
        series,
        target_ts,
    )

    X, defaulted = build_feature_row(
        model,
        known,
    )

    prediction = max(
        0,
        float(model.predict(X)[0]),
    )

    # ------------------------------------------------------------------------
    # RESULTS
    # ------------------------------------------------------------------------

    st.divider()

    if actual is not None:

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Predicted trips",
            f"{prediction:,.0f}",
        )

        c2.metric(
            "Actual trips",
            f"{actual:,.0f}",
        )

        c3.metric(
            "Prediction difference",
            f"{prediction - actual:+,.0f}",
        )

    else:

        st.metric(
            "Predicted trips",
            f"{prediction:,.0f}",
        )

        st.info(
            "This timestamp is outside the available "
            "historical range, so an actual value cannot "
            "be displayed."
        )

    if missing:

        st.warning(
            "Some historical features were unavailable: "
            + "; ".join(missing)
        )

    if defaulted:

        st.warning(
            "Some model features were not supplied and "
            "were defaulted to 0: "
            + ", ".join(
                f"`{x}`"
                for x in defaulted
            )
        )

    with st.expander(
        "Show model input"
    ):

        st.dataframe(
            X,
            use_container_width=True,
        )


# ----------------------------------------------------------------------------
# TASK 3 — ZONE DEMAND
# ----------------------------------------------------------------------------

def predict_zone_demand_tab():

    st.subheader(
        "Zone-Level Hourly Demand Prediction"
    )

    st.caption(
        "Forecast the number of pickups for one specific "
        "NYC pickup zone and hour."
    )

    vehicle = st.selectbox(
        "Vehicle type",
        list(PATHS.keys()),
        key="zone_vehicle",
    )

    cfg = PATHS[vehicle]

    model = load_model(
        cfg["zone_demand_model"]
    )

    if model is None:

        st.error(
            f"Zone demand model not found:\n\n"
            f"{cfg['zone_demand_model']}"
        )

        return

    zones = load_zone_list(
        cfg["zone_hourly_demand"]
    )

    if zones.empty:

        st.error(
            "Zone demand data was not found."
        )

        return

    zone_options = {}

    for row in zones.itertuples():

        label = (
            f"{row.zone_name} "
            f"({row.borough}) "
            f"-- #{row.zone_id}"
        )

        zone_options[label] = int(row.zone_id)

    # ------------------------------------------------------------------------
    # INPUT
    # ------------------------------------------------------------------------

    with st.form(
        "zone_prediction_form"
    ):

        zone_label = st.selectbox(
            "Pickup zone",
            list(zone_options.keys()),
        )

        c1, c2 = st.columns(2)

        with c1:

            target_date = st.date_input(
                "Forecast date",
                value=date(2025, 12, 1),
                key="zone_date",
            )

        with c2:

            target_time = st.time_input(
                "Forecast hour",
                value=dtime(hour=9),
                key="zone_time",
            )

        submitted = st.form_submit_button(
            "Predict zone demand",
            use_container_width=True,
        )

    if not submitted:
        return

    zone_id = zone_options[
        zone_label
    ]

    series = load_zone_series(
        cfg["zone_hourly_demand"],
        zone_id,
    )

    if series.empty:

        st.error(
            "No historical demand was found "
            "for this zone."
        )

        return

    target_ts = pd.Timestamp.combine(
        target_date,
        target_time,
    ).floor("h")

    known, missing, actual = build_lag_known(
        series,
        target_ts,
    )

    # The zone model was trained with zone_id
    known["zone_id"] = zone_id

    X, defaulted = build_feature_row(
        model,
        known,
    )

    prediction = max(
        0,
        float(model.predict(X)[0]),
    )

    # ------------------------------------------------------------------------
    # RESULTS
    # ------------------------------------------------------------------------

    st.divider()

    if actual is not None:

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Predicted trips",
            f"{prediction:,.1f}",
        )

        c2.metric(
            "Actual trips",
            f"{actual:,.0f}",
        )

        c3.metric(
            "Prediction difference",
            f"{prediction - actual:+,.1f}",
        )

    else:

        st.metric(
            "Predicted trips",
            f"{prediction:,.1f}",
        )

        st.info(
            "No actual trip count exists for this "
            "timestamp in the historical data."
        )

    if missing:

        st.warning(
            "Some historical features were unavailable: "
            + "; ".join(missing)
        )

    if defaulted:

        st.warning(
            "Some model features were not supplied "
            "and were defaulted to 0: "
            + ", ".join(
                f"`{x}`"
                for x in defaulted
            )
        )

    with st.expander(
        "Show model input"
    ):

        st.dataframe(
            X,
            use_container_width=True,
        )


# ============================================================================
# PREDICT PAGE
# ============================================================================

def page_predict():

    st.header(
        "🔮 Taxi Demand & Duration Prediction"
    )

    st.caption(
        "Interactive predictions using the trained "
        "taxi and for-hire-vehicle machine-learning models."
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "Trip Duration",
            "Citywide Demand",
            "Zone Demand",
        ]
    )

    with tab1:
        predict_duration_tab()

    with tab2:
        predict_citywide_demand_tab()

    with tab3:
        predict_zone_demand_tab()


# ============================================================================
# DASHBOARD HELPERS
# ============================================================================

def explanation_box(text: str):

    st.markdown(
        f"""
        <div style="
            background:#F3F5F9;
            border-left:4px solid #4C72B0;
            padding:0.75rem 1rem;
            border-radius:4px;
            font-size:0.92rem;
        ">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# TASK 1 PERFORMANCE
# ============================================================================

def dashboard_duration(
    cfg: dict,
    vehicle: str,
):

    st.subheader("Task 1 — Trip Duration Prediction")

    metrics = parse_metrics_txt(
        cfg["duration_metrics"]
    )

    if not metrics:
        st.info("Duration metrics were not found.")
        return

    # Metrics are stored internally in seconds.
    mae = metrics.get("MAE")
    rmse = metrics.get("RMSE")
    r2 = metrics.get("R2")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "MAE",
        f"{mae / 60.0:.1f} min" if mae is not None else "N/A",
        help="Mean Absolute Error",
    )

    c2.metric(
        "RMSE",
        f"{rmse / 60.0:.1f} min" if rmse is not None else "N/A",
        help="Root Mean Squared Error",
    )

    c3.metric(
        "R²",
        f"{r2:.3f}" if r2 is not None else "N/A",
    )

    explanation_box(
        f"""
        For <b>{vehicle}</b>, the model's average prediction error
        is approximately <b>{mae / 60.0:.1f} minutes</b>.
        """ if mae is not None else
        f"""
        Duration metrics for <b>{vehicle}</b> were only partially available.
        """
    )

    # ------------------------------------------------------------------------
    # ACTUAL VS PREDICTED DURATION
    # ------------------------------------------------------------------------

    predictions_path = cfg["duration_predictions"]

    if predictions_path.exists():

        duration_df = duckdb.sql(
            f"""
            SELECT *
            FROM read_parquet(
                '{predictions_path.as_posix()}'
            )
            """
        ).df()

        if (
            not duration_df.empty
            and "actual" in duration_df.columns
            and "predicted" in duration_df.columns
        ):

            duration_df["actual_minutes"] = (
                duration_df["actual"] / 60.0
            )

            duration_df["predicted_minutes"] = (
                duration_df["predicted"] / 60.0
            )

            # Keep the chart readable.
            chart_df = duration_df.head(5000).copy()

            chart_df["Trip"] = range(1, len(chart_df) + 1)

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=chart_df["Trip"],
                    y=chart_df["actual_minutes"],
                    name="Actual",
                    mode="lines",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=chart_df["Trip"],
                    y=chart_df["predicted_minutes"],
                    name="Predicted",
                    mode="lines",
                )
            )

            fig.update_layout(
                title=(
                    f"{vehicle} — Actual vs Predicted "
                    "Trip Duration"
                ),
                xaxis_title="Test trips",
                yaxis_title="Duration (minutes)",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            explanation_box(
                """
                This chart compares the <b>actual trip duration</b>
                with the <b>model-predicted duration</b> for trips
                in the saved test predictions. Both values are shown
                in minutes so the comparison is directly understandable.
                """
            )

    # ------------------------------------------------------------------------
    # FEATURE IMPORTANCE
    # ------------------------------------------------------------------------

    importances = metrics.get("importances")

    if (
        importances is not None
        and not importances.empty
    ):

        fig2 = px.bar(
            importances.sort_values().reset_index(),
            x=0,
            y="index",
            orientation="h",
            labels={
                "0": "Importance",
                "index": "Feature",
            },
            title="Features influencing trip duration",
        )

        st.plotly_chart(
            fig2,
            use_container_width=True,
        )


# ============================================================================
# TASK 2 PERFORMANCE
# ============================================================================

def dashboard_demand(
    cfg: dict,
    vehicle: str,
):

    st.subheader(
        "Task 2 — Citywide Hourly Demand"
    )

    metrics = sql_regression_metrics(
        cfg["demand_predictions"]
    )

    if not metrics:

        st.info(
            "Demand prediction results were not found."
        )

        return

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "MAE",
        f"{metrics['MAE']:.1f} trips/hr",
    )

    c2.metric(
        "RMSE",
        f"{metrics['RMSE']:.1f} trips/hr",
    )

    c3.metric(
        "R²",
        f"{metrics['R2']:.3f}",
    )

    # ------------------------------------------------------------------------
    # ACTUAL VS PREDICTED
    # ------------------------------------------------------------------------

    pred_path = cfg[
        "demand_predictions"
    ]

    df = duckdb.sql(
        f"""
        SELECT *
        FROM read_parquet(
            '{pred_path.as_posix()}'
        )
        """
    ).df()

    if "pickup_hour_ts" in df.columns:

        df["pickup_hour_ts"] = pd.to_datetime(
            df["pickup_hour_ts"]
        )

        df = df.sort_values(
            "pickup_hour_ts"
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df["pickup_hour_ts"],
                y=df["actual"],
                name="Actual",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df["pickup_hour_ts"],
                y=df["predicted"],
                name="Predicted",
            )
        )

        fig.update_layout(
            title=(
                f"{vehicle} — Actual vs Predicted "
                "Citywide Demand"
            ),
            xaxis_title="Time",
            yaxis_title="Trips / hour",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    explanation_box(
        f"""
        The citywide demand model for <b>{vehicle}</b>
        achieved an R² of <b>{metrics['R2']:.3f}</b>.
        The actual-vs-predicted chart shows how closely
        the model follows the real hourly demand pattern
        during the held-out test period.
        """
    )


# ============================================================================
# TASK 3 PERFORMANCE
# ============================================================================

def dashboard_zone_demand(
    cfg: dict,
    vehicle: str,
):

    st.subheader(
        "Task 3 — Zone-Level Demand"
    )

    metrics = sql_regression_metrics(
        cfg["zone_demand_predictions"]
    )

    if not metrics:

        st.info(
            "Zone demand prediction results "
            "were not found."
        )

        return

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "MAE",
        f"{metrics['MAE']:.2f} trips/zone/hr",
    )

    c2.metric(
        "RMSE",
        f"{metrics['RMSE']:.2f} trips/zone/hr",
    )

    c3.metric(
        "R²",
        f"{metrics['R2']:.3f}",
    )

    explanation_box(
        f"""
        The zone-level model predicts hourly pickup
        demand for individual NYC zones.
        For <b>{vehicle}</b>, the model achieved
        an R² of <b>{metrics['R2']:.3f}</b>.
        """
    )

    preds_path = cfg[
        "zone_demand_predictions"
    ]

    zone_path = cfg[
        "zone_hourly_demand"
    ]

    if not zone_path.exists():
        return

    # ------------------------------------------------------------------------
    # TOP BUSIEST ZONES
    # ------------------------------------------------------------------------

    top = duckdb.sql(
        f"""
        SELECT

            p.zone_id,

            any_value(z.zone_name)
                AS zone_name,

            any_value(z.borough)
                AS borough,

            sum(p.actual)
                AS total_actual

        FROM read_parquet(
            '{preds_path.as_posix()}'
        ) p

        LEFT JOIN (

            SELECT DISTINCT
                zone_id,
                zone_name,
                borough

            FROM read_parquet(
                '{zone_path.as_posix()}'
            )

        ) z

        ON p.zone_id = z.zone_id

        GROUP BY p.zone_id

        ORDER BY total_actual DESC

        LIMIT 15
        """
    ).df()

    if not top.empty:

        top["label"] = (
            top["zone_name"].fillna(
                top["zone_id"].astype(str)
            )
            + " ("
            + top["borough"].fillna("?")
            + ")"
        )

        fig = px.bar(
            top.sort_values(
                "total_actual"
            ),
            x="total_actual",
            y="label",
            orientation="h",
            title=(
                f"{vehicle} — Top 15 "
                "Busiest Pickup Zones"
            ),
            labels={
                "total_actual":
                    "Actual trips",
                "label": "",
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # ------------------------------------------------------------------------
    # ACTUAL VS PREDICTED SAMPLE
    # ------------------------------------------------------------------------

    zone_pred = duckdb.sql(
        f"""
        SELECT
            pickup_hour_ts,
            zone_id,
            actual,
            predicted
        FROM read_parquet(
            '{preds_path.as_posix()}'
        )
        ORDER BY pickup_hour_ts
        LIMIT 5000
        """
    ).df()

    if (
        not zone_pred.empty
        and "pickup_hour_ts"
        in zone_pred.columns
    ):

        zone_pred[
            "pickup_hour_ts"
        ] = pd.to_datetime(
            zone_pred["pickup_hour_ts"]
        )

        # Select a representative zone
        selected_zone = int(
            zone_pred["zone_id"].iloc[0]
        )

        zone_sample = zone_pred[
            zone_pred["zone_id"]
            == selected_zone
        ].copy()

        if not zone_sample.empty:

            fig2 = go.Figure()

            fig2.add_trace(
                go.Scatter(
                    x=zone_sample[
                        "pickup_hour_ts"
                    ],
                    y=zone_sample[
                        "actual"
                    ],
                    name="Actual",
                )
            )

            fig2.add_trace(
                go.Scatter(
                    x=zone_sample[
                        "pickup_hour_ts"
                    ],
                    y=zone_sample[
                        "predicted"
                    ],
                    name="Predicted",
                )
            )

            fig2.update_layout(
                title=(
                    f"{vehicle} — Zone "
                    f"{selected_zone}: "
                    "Actual vs Predicted"
                ),
                xaxis_title="Time",
                yaxis_title="Trips / hour",
            )

            st.plotly_chart(
                fig2,
                use_container_width=True,
            )


# ============================================================================
# PERFORMANCE DASHBOARD PAGE
# ============================================================================

def page_dashboard():

    st.header(
        "📊 Model Performance Dashboard"
    )

    st.caption(
        "Performance of all twelve trained models: "
        "Yellow Taxi, FHV, Green Taxi and HVFHV."
    )

    tabs = st.tabs(
        [
            "Executive Summary",
            "Yellow Taxi",
            "FHV",
        ]
    )

    # ------------------------------------------------------------------------
    # EXECUTIVE SUMMARY
    # ------------------------------------------------------------------------

    with tabs[0]:

        st.subheader(
            "All 12 Models at a Glance"
        )

        rows = []

        for vehicle, cfg in PATHS.items():

            # Duration
            dur = parse_metrics_txt(
                cfg["duration_metrics"]
            )

            if dur:

                rows.append(
                    {
                        "Vehicle":
                            vehicle,

                        "Task":
                            "Trip Duration",

                        "MAE":
                            dur.get("MAE"),

                        "R²":
                            dur.get("R2"),
                    }
                )

            # Citywide
            dem = sql_regression_metrics(
                cfg["demand_predictions"]
            )

            if dem:

                rows.append(
                    {
                        "Vehicle":
                            vehicle,

                        "Task":
                            "Citywide Demand",

                        "MAE":
                            dem.get("MAE"),

                        "R²":
                            dem.get("R2"),
                    }
                )

            # Zone
            zon = sql_regression_metrics(
                cfg["zone_demand_predictions"]
            )

            if zon:

                rows.append(
                    {
                        "Vehicle":
                            vehicle,

                        "Task":
                            "Zone Demand",

                        "MAE":
                            zon.get("MAE"),

                        "R²":
                            zon.get("R2"),
                    }
                )

        if rows:

            summary = pd.DataFrame(
                rows
            )

            st.dataframe(
                summary.style.format(
                    {
                        "MAE":
                            "{:.2f}",

                        "R²":
                            "{:.3f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            # --------------------------------------------------------------
            # R2 COMPARISON
            # --------------------------------------------------------------

            fig = px.bar(
                summary,
                x="Task",
                y="R²",
                color="Vehicle",
                barmode="group",
                title=(
                    "Model R² by Task "
                    "and Vehicle"
                ),
                range_y=[0, 1],
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            explanation_box(
                """
                R² measures how much of the variation
                in the target the model explains.
                Values closer to 1 indicate stronger
                predictive performance.
                """
            )

        else:

            st.info(
                "No trained model results were found."
            )

    # ------------------------------------------------------------------------
    # YELLOW TAXI
    # ------------------------------------------------------------------------

    with tabs[1]:

        cfg = PATHS["Yellow Taxi"]

        st.header(
            "🚕 Yellow Taxi"
        )

        dashboard_duration(
            cfg,
            "Yellow Taxi",
        )

        st.divider()

        dashboard_demand(
            cfg,
            "Yellow Taxi",
        )

        st.divider()

        dashboard_zone_demand(
            cfg,
            "Yellow Taxi",
        )

    # ------------------------------------------------------------------------
    # FHV
    # ------------------------------------------------------------------------

    with tabs[2]:

        cfg = PATHS["FHV"]

        st.header(
            "🚗 For-Hire Vehicle (FHV)"
        )

        dashboard_duration(
            cfg,
            "FHV",
        )

        st.divider()

        dashboard_demand(
            cfg,
            "FHV",
        )

        st.divider()

        dashboard_zone_demand(
            cfg,
            "FHV",
        )


# ============================================================================
# MAIN APP
# ============================================================================

def main():

    st.set_page_config(
        page_title="NYC Taxi Analytics",
        page_icon="🚕",
        layout="wide",
    )

    st.markdown(
        """
        <style>

        .block-container {
            padding-top: 2rem;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.6rem;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------------
    # SIDEBAR
    # ------------------------------------------------------------------------

    st.sidebar.title(
        "🚕 NYC Taxi Analytics"
    )

    page = st.sidebar.radio(
        "Go to",
        [
            "🔮 Predict",
            "📊 Model Performance Dashboard",
        ],
    )

    st.sidebar.divider()

    st.sidebar.caption(
        """
        **Yellow Taxi**

        NYC traditional yellow taxi trips.

        **FHV**

        For-Hire Vehicle trips.

        **Green Taxi**

        NYC green taxi trips.

        **HVFHV**

        High Volume For-Hire Vehicle trips.

        ---

        12 trained ML models:

        • 4 Trip Duration models

        • 4 Citywide Demand models

        • 4 Zone Demand models
        """
    )

    # ------------------------------------------------------------------------
    # PAGE ROUTING
    # ------------------------------------------------------------------------

    if page == "🔮 Predict":

        page_predict()

    else:

        page_dashboard()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()