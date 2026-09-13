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

# Every dataset in this project should be scoped to calendar-year 2025 only.
YEAR_START = date(2025, 1, 1)
YEAR_END = date(2025, 12, 31)

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
        WHERE pickup_hour_ts >= TIMESTAMP '{YEAR_START.isoformat()}'
          AND pickup_hour_ts < TIMESTAMP '{(date(YEAR_END.year + 1, 1, 1)).isoformat()}'
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
          AND pickup_hour_ts >= TIMESTAMP '{YEAR_START.isoformat()}'
          AND pickup_hour_ts < TIMESTAMP '{(date(YEAR_END.year + 1, 1, 1)).isoformat()}'
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
        return YEAR_START, YEAR_END
    row = duckdb.sql(
        f"""
        SELECT min(pickup_hour_ts), max(pickup_hour_ts)
        FROM read_parquet('{hourly_demand_path.as_posix()}')
        WHERE pickup_hour_ts >= TIMESTAMP '{YEAR_START.isoformat()}'
          AND pickup_hour_ts < TIMESTAMP '{(date(YEAR_END.year + 1, 1, 1)).isoformat()}'
        """
    ).fetchone()
    if row and row[0] is not None:
        lo = pd.Timestamp(row[0]).date()
        hi = pd.Timestamp(row[1]).date()
        # Clip defensively to 2025 even if the file somehow contains
        # stray timestamps outside the target year.
        lo = max(lo, YEAR_START)
        hi = min(hi, YEAR_END)
        return lo, hi
    return YEAR_START, YEAR_END


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
def duration_metrics_with_fallback(
    metrics_path: Path,
    predictions_path: Path,
):
    """
    Load duration metrics from the saved metrics TXT.

    If that TXT is missing/incomplete, calculate MAE/RMSE/R² directly
    from the saved actual-vs-predicted duration Parquet. This is useful
    for older Yellow Taxi outputs where the prediction file exists but
    the separate metrics TXT was not saved.
    """
    metrics = parse_metrics_txt(metrics_path)

    required = ("MAE", "RMSE", "R2")
    if all(metrics.get(key) is not None for key in required):
        return metrics

    if not predictions_path.exists():
        return metrics

    try:
        row = duckdb.sql(
            f"""
            WITH p AS (
                SELECT
                    actual,
                    predicted
                FROM read_parquet('{predictions_path.as_posix()}')
                WHERE actual IS NOT NULL
                  AND predicted IS NOT NULL
            ),
            stats AS (
                SELECT avg(actual) AS mean_actual
                FROM p
            )
            SELECT
                avg(abs(actual - predicted)) AS mae,
                sqrt(avg(power(actual - predicted, 2))) AS rmse,
                CASE
                    WHEN sum(
                        power(
                            actual - (SELECT mean_actual FROM stats),
                            2
                        )
                    ) = 0
                    THEN NULL
                    ELSE
                        1
                        - sum(power(actual - predicted, 2))
                        /
                        sum(
                            power(
                                actual - (SELECT mean_actual FROM stats),
                                2
                            )
                        )
                END AS r2
            FROM p
            """
        ).fetchone()

        if row and any(value is not None for value in row):
            fallback = {
                "MAE": row[0],
                "RMSE": row[1],
                "R2": row[2],
            }

            # Preserve feature importances if the TXT existed.
            if "importances" in metrics:
                fallback["importances"] = metrics["importances"]

            return fallback

    except Exception:
        pass

    return metrics


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
# SHARED ACTUAL-VS-PREDICTED CHART HELPERS
# ============================================================================

def actual_vs_predicted_duration_chart(predictions_path: Path, vehicle: str):
    """
    Build the actual-vs-predicted trip-duration line chart from the saved
    test-set predictions parquet. Returns None if unavailable.
    """
    if not predictions_path.exists():
        return None

    duration_df = duckdb.sql(
        f"""
        SELECT *
        FROM read_parquet('{predictions_path.as_posix()}')
        """
    ).df()

    if (
        duration_df.empty
        or "actual" not in duration_df.columns
        or "predicted" not in duration_df.columns
    ):
        return None

    duration_df["actual_minutes"] = duration_df["actual"] / 60.0
    duration_df["predicted_minutes"] = duration_df["predicted"] / 60.0

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
        title=f"{vehicle} — Actual vs Predicted Trip Duration (test trips)",
        xaxis_title="Test trips",
        yaxis_title="Duration (minutes)",
    )
    return fig


def actual_vs_predicted_series_chart(
    predictions_path: Path,
    vehicle: str,
    title_suffix: str,
    y_title: str = "Trips / hour",
):
    """
    Build a time-indexed actual-vs-predicted chart from a saved demand
    predictions parquet (citywide or zone level). Returns None if the
    file is unavailable or lacks the expected columns.
    """
    if not predictions_path.exists():
        return None

    df = duckdb.sql(
        f"""
        SELECT *
        FROM read_parquet('{predictions_path.as_posix()}')
        """
    ).df()

    if df.empty or "actual" not in df.columns or "predicted" not in df.columns:
        return None

    if "pickup_hour_ts" in df.columns:
        df["pickup_hour_ts"] = pd.to_datetime(df["pickup_hour_ts"])
        df = df.sort_values("pickup_hour_ts")
        x = df["pickup_hour_ts"]
        x_title = "Time"
    else:
        df = df.reset_index(drop=True)
        x = df.index
        x_title = "Test rows"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=df["actual"], name="Actual"))
    fig.add_trace(go.Scatter(x=x, y=df["predicted"], name="Predicted"))
    fig.update_layout(
        title=f"{vehicle} — Actual vs Predicted {title_suffix}",
        xaxis_title=x_title,
        yaxis_title=y_title,
    )
    return fig


def highlight_point_on_chart(fig: go.Figure, x_value, y_value, label: str):
    """Overlay a single marker (this prediction) on an existing chart."""
    fig.add_trace(
        go.Scatter(
            x=[x_value],
            y=[y_value],
            mode="markers",
            name=label,
            marker=dict(size=14, symbol="star", color="red"),
        )
    )
    return fig


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


@st.cache_data(show_spinner=False)
def load_top_pickup_zones(zone_path: Path, limit: int = 10) -> pd.DataFrame:
    """Return the busiest pickup zones for the selected vehicle."""
    if not zone_path.exists():
        return pd.DataFrame(
            columns=["zone_id", "zone_name", "borough", "total_trips"]
        )

    try:
        return duckdb.sql(
            f"""
            SELECT
                zone_id,
                any_value(zone_name) AS zone_name,
                any_value(borough) AS borough,
                SUM(trip_count) AS total_trips
            FROM read_parquet('{zone_path.as_posix()}')
            GROUP BY zone_id
            ORDER BY total_trips DESC
            LIMIT {int(limit)}
            """
        ).df()
    except Exception:
        return pd.DataFrame(
            columns=["zone_id", "zone_name", "borough", "total_trips"]
        )


def vehicle_focus_section(vehicle: str, cfg: dict):
    """
    Show where the selected vehicle type is most concentrated.

    The ranking is calculated directly from the vehicle's zone-level
    hourly demand data, so the dashboard reports the actual top pickup
    zones in the project dataset rather than generic NYC assumptions.
    """
    st.divider()
    st.subheader(f"Where {vehicle} Trips Mainly Concentrate")

    st.caption(
        "Top pickup zones based on the total number of trips recorded "
        "in the selected vehicle's zone-level demand dataset."
    )

    top = load_top_pickup_zones(
        cfg["zone_hourly_demand"],
        limit=10,
    )

    if top.empty:
        st.info(
            "Zone-level demand data is not available for this vehicle."
        )
        return

    top["zone_label"] = (
        top["zone_name"].fillna(top["zone_id"].astype(str))
        + " ("
        + top["borough"].fillna("Unknown")
        + ")"
    )

    # The first row is the exact busiest pickup zone in this dataset.
    busiest = top.iloc[0]

    st.success(
        f"**Main focus:** {busiest['zone_label']} has the highest "
        f"pickup concentration for {vehicle}, with "
        f"**{busiest['total_trips']:,.0f} trips** in the dataset."
    )

    chart_df = top.sort_values("total_trips")

    fig = px.bar(
        chart_df,
        x="total_trips",
        y="zone_label",
        orientation="h",
        title=f"Top 10 Pickup Zones — {vehicle}",
        labels={
            "total_trips": "Total trips",
            "zone_label": "Pickup zone",
        },
    )

    fig.update_layout(
        height=500,
        showlegend=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    display_df = top[
        ["zone_id", "zone_name", "borough", "total_trips"]
    ].copy()

    display_df.columns = [
        "Zone ID",
        "Zone",
        "Borough",
        "Total trips",
    ]

    display_df["Total trips"] = display_df["Total trips"].round(0).astype(int)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


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

    prediction_minutes = prediction_seconds / 60.0

    st.divider()

    # Same-zone trips: show only the informational message.
    # Do not display trip duration or trip distance for this case.
    if same_zone:
        st.info(
            f"📍 **Same zone trip** — pickup and drop-off are both "
            f"**{pickup_label}**. The model is predicting a trip that "
            f"starts and ends within the same taxi zone."
        )
        return

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

    with st.expander("Show model input"):
        st.dataframe(X, use_container_width=True)

    # ------------------------------------------------------------------
    # ACTUAL VS PREDICTED — every prediction gets a chart, since Oct/Nov/
    # Dec pickup dates fall in the held-out test period for these models.
    # ------------------------------------------------------------------
    st.divider()
    st.markdown("#### How this prediction compares to the test set")

    dur_fig = actual_vs_predicted_duration_chart(
        cfg["duration_predictions"], vehicle
    )
    if dur_fig is not None:
        highlight_point_on_chart(
            dur_fig,
            x_value=1,
            y_value=prediction_minutes,
            label="This prediction",
        )
        st.plotly_chart(dur_fig, use_container_width=True)
        st.caption(
            "The line chart shows actual vs. predicted duration across the "
            "model's saved test trips. The red star marks this prediction's "
            "duration for reference, not its exact position in the test set."
        )
    else:
        st.info(
            "Saved test-set predictions were not found for this vehicle, "
            "so an actual-vs-predicted comparison chart can't be shown."
        )

    # Vehicle-specific concentration analysis.
    # This updates automatically when the user changes Vehicle type.
    vehicle_focus_section(
        vehicle,
        cfg,
    )


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
                min_value=YEAR_START,
                max_value=YEAR_END,
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

    # ------------------------------------------------------------------
    # ACTUAL VS PREDICTED CHART FOR THIS VEHICLE'S TEST SET
    # ------------------------------------------------------------------
    st.divider()
    st.markdown("#### How this prediction compares to the test set")

    demand_fig = actual_vs_predicted_series_chart(
        cfg["demand_predictions"],
        vehicle,
        title_suffix="Citywide Demand",
        y_title="Trips / hour",
    )
    if demand_fig is not None:
        highlight_point_on_chart(
            demand_fig,
            x_value=target_ts,
            y_value=prediction,
            label="This prediction",
        )
        st.plotly_chart(demand_fig, use_container_width=True)
        st.caption(
            "The line chart shows actual vs. predicted citywide demand "
            "across the model's saved test hours. The red star marks this "
            "prediction's value at the selected forecast time."
        )
    else:
        st.info(
            "Saved test-set predictions were not found for this vehicle, "
            "so an actual-vs-predicted comparison chart can't be shown."
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
                min_value=YEAR_START,
                max_value=YEAR_END,
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

    # ------------------------------------------------------------------
    # ACTUAL VS PREDICTED CHART FOR THIS ZONE'S TEST SET
    # ------------------------------------------------------------------
    st.divider()
    st.markdown("#### How this prediction compares to the test set")

    zone_pred_path = cfg["zone_demand_predictions"]
    zone_fig = None

    if zone_pred_path.exists():
        try:
            zone_pred_df = duckdb.sql(
                f"""
                SELECT pickup_hour_ts, actual, predicted
                FROM read_parquet('{zone_pred_path.as_posix()}')
                WHERE zone_id = {int(zone_id)}
                ORDER BY pickup_hour_ts
                """
            ).df()
        except Exception:
            zone_pred_df = pd.DataFrame()

        if not zone_pred_df.empty:
            zone_pred_df["pickup_hour_ts"] = pd.to_datetime(
                zone_pred_df["pickup_hour_ts"]
            )
            zone_fig = go.Figure()
            zone_fig.add_trace(
                go.Scatter(
                    x=zone_pred_df["pickup_hour_ts"],
                    y=zone_pred_df["actual"],
                    name="Actual",
                )
            )
            zone_fig.add_trace(
                go.Scatter(
                    x=zone_pred_df["pickup_hour_ts"],
                    y=zone_pred_df["predicted"],
                    name="Predicted",
                )
            )
            zone_fig.update_layout(
                title=f"{vehicle} — Zone {zone_id}: Actual vs Predicted (test hours)",
                xaxis_title="Time",
                yaxis_title="Trips / hour",
            )

    if zone_fig is None:
        # Fall back to the vehicle-wide zone-demand test set if this
        # specific zone has no saved test rows.
        zone_fig = actual_vs_predicted_series_chart(
            zone_pred_path,
            vehicle,
            title_suffix="Zone Demand (all zones, test set)",
            y_title="Trips / hour",
        )

    if zone_fig is not None:
        highlight_point_on_chart(
            zone_fig,
            x_value=target_ts,
            y_value=prediction,
            label="This prediction",
        )
        st.plotly_chart(zone_fig, use_container_width=True)
        st.caption(
            "The line chart shows actual vs. predicted demand for this zone "
            "across the model's saved test hours. The red star marks this "
            "prediction's value at the selected forecast time."
        )
    else:
        st.info(
            "Saved test-set predictions were not found for this vehicle, "
            "so an actual-vs-predicted comparison chart can't be shown."
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
        "taxi and for-hire-vehicle machine-learning models "
        "(2025 data only)."
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

    metrics = duration_metrics_with_fallback(
        cfg["duration_metrics"],
        cfg["duration_predictions"],
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

    dur_fig = actual_vs_predicted_duration_chart(
        cfg["duration_predictions"], vehicle
    )

    if dur_fig is not None:
        st.plotly_chart(dur_fig, use_container_width=True)

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

    demand_fig = actual_vs_predicted_series_chart(
        cfg["demand_predictions"],
        vehicle,
        title_suffix="Citywide Demand",
        y_title="Trips / hour",
    )

    if demand_fig is not None:
        st.plotly_chart(demand_fig, use_container_width=True)

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

@st.cache_data(show_spinner=False)
def select_accurate_zone_for_chart(
    preds_path: Path,
    min_rows: int = 200,
    min_avg_actual: float = 5.0,
) -> int:
    """
    Pick a zone whose saved test-set rows make a clear, trustworthy
    actual-vs-predicted chart: enough test rows, enough real trip volume
    to show meaningful variation, and low relative error (MAE relative
    to average actual trips). Returns None if no zone qualifies.
    """
    if not preds_path.exists():
        return None

    try:
        df = duckdb.sql(
            f"""
            SELECT
                zone_id,
                count(*) AS n,
                avg(actual) AS avg_actual,
                avg(abs(actual - predicted)) AS mae
            FROM read_parquet('{preds_path.as_posix()}')
            GROUP BY zone_id
            HAVING count(*) >= {min_rows}
               AND avg(actual) >= {min_avg_actual}
            """
        ).df()
    except Exception:
        return None

    if df.empty:
        return None

    df["relative_error"] = df["mae"] / df["avg_actual"]
    best = df.sort_values("relative_error").iloc[0]
    return int(best["zone_id"])


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

    # Yellow Taxi and Green Taxi: pick a zone with enough test data and
    # low relative error, so the sample chart is actually representative
    # instead of whichever zone happens to appear first chronologically.
    if vehicle in ("Yellow Taxi", "Green Taxi"):
        accurate_zone = select_accurate_zone_for_chart(preds_path)
    else:
        accurate_zone = None

    if accurate_zone is not None:

        zone_sample = duckdb.sql(
            f"""
            SELECT pickup_hour_ts, actual, predicted
            FROM read_parquet('{preds_path.as_posix()}')
            WHERE zone_id = {accurate_zone}
            ORDER BY pickup_hour_ts
            """
        ).df()

        if not zone_sample.empty:
            zone_sample["pickup_hour_ts"] = pd.to_datetime(
                zone_sample["pickup_hour_ts"]
            )

            fig2 = go.Figure()

            fig2.add_trace(
                go.Scatter(
                    x=zone_sample["pickup_hour_ts"],
                    y=zone_sample["actual"],
                    name="Actual",
                )
            )

            fig2.add_trace(
                go.Scatter(
                    x=zone_sample["pickup_hour_ts"],
                    y=zone_sample["predicted"],
                    name="Predicted",
                )
            )

            fig2.update_layout(
                title=(
                    f"{vehicle} — Zone {accurate_zone}: "
                    "Actual vs Predicted"
                ),
                xaxis_title="Time",
                yaxis_title="Trips / hour",
            )

            st.plotly_chart(
                fig2,
                use_container_width=True,
            )

        return

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
        "Yellow Taxi, FHV, Green Taxi and HVFHV (2025 data only)."
    )

    tabs = st.tabs(
        [
            "Executive Summary",
            "Yellow Taxi",
            "FHV",
            "Green Taxi",
            "HVFHV (Uber/Lyft)",
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

            # --------------------------------------------------------------
            # Task 1 — Trip Duration
            # --------------------------------------------------------------
            dur = duration_metrics_with_fallback(
                cfg["duration_metrics"],
                cfg["duration_predictions"],
            )

            rows.append(
                {
                    "Vehicle": vehicle,
                    "Task": "Trip Duration",
                    "MAE": (
                        dur.get("MAE") / 60.0
                        if dur.get("MAE") is not None
                        else None
                    ),
                    "R²": dur.get("R2"),
                    "Status": (
                        "Available"
                        if dur.get("MAE") is not None
                        and dur.get("R2") is not None
                        else "Unavailable"
                    ),
                }
            )

            # --------------------------------------------------------------
            # Task 2 — Citywide Demand
            # --------------------------------------------------------------
            dem = sql_regression_metrics(
                cfg["demand_predictions"]
            )

            rows.append(
                {
                    "Vehicle": vehicle,
                    "Task": "Citywide Demand",
                    "MAE": dem.get("MAE") if dem else None,
                    "R²": dem.get("R2") if dem else None,
                    "Status": (
                        "Available"
                        if dem
                        and dem.get("MAE") is not None
                        and dem.get("R2") is not None
                        else "Unavailable"
                    ),
                }
            )

            # --------------------------------------------------------------
            # Task 3 — Zone Demand
            # --------------------------------------------------------------
            zon = sql_regression_metrics(
                cfg["zone_demand_predictions"]
            )

            rows.append(
                {
                    "Vehicle": vehicle,
                    "Task": "Zone Demand",
                    "MAE": zon.get("MAE") if zon else None,
                    "R²": zon.get("R2") if zon else None,
                    "Status": (
                        "Available"
                        if zon
                        and zon.get("MAE") is not None
                        and zon.get("R2") is not None
                        else "Unavailable"
                    ),
                }
            )

        summary = pd.DataFrame(rows)

        # Keep all 12 models visible instead of silently dropping models
        # whose saved result file is missing.
        st.dataframe(
            summary.style.format(
                {
                    "MAE": lambda value: (
                        "N/A"
                        if pd.isna(value)
                        else f"{value:.2f}"
                    ),
                    "R²": lambda value: (
                        "N/A"
                        if pd.isna(value)
                        else f"{value:.3f}"
                    ),
                }
            ),
            use_container_width=True,
            hide_index=True,
            height=520,
        )

        st.caption(
            "Trip Duration MAE is shown in minutes. "
            "Citywide and Zone Demand MAE values are shown in their "
            "respective trip-count units. N/A means the saved evaluation "
            "result for that model/task is not available."
        )

        # --------------------------------------------------------------
        # R² COMPARISON
        # --------------------------------------------------------------
        chart_df = summary.dropna(subset=["R²"]).copy()

        if not chart_df.empty:
            chart_df["Model"] = (
                chart_df["Vehicle"]
                + " — "
                + chart_df["Task"]
            )

            fig = px.bar(
                chart_df.sort_values("R²"),
                x="R²",
                y="Model",
                orientation="h",
                range_x=[0, 1],
                title="R² Comparison — Available Models",
                labels={
                    "R²": "R² (higher is better)",
                    "Model": "",
                },
            )

            fig.update_layout(
                height=560,
                showlegend=False,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            missing_models = summary[
                summary["R²"].isna()
            ]["Vehicle"] + " — " + summary[
                summary["R²"].isna()
            ]["Task"]

            if not missing_models.empty:
                st.warning(
                    "The following model results are not plotted because "
                    "their saved evaluation metrics are unavailable: "
                    + ", ".join(missing_models.tolist())
                    + "."
                )

        explanation_box(
            """
            R² measures how much of the variation in the target the model
            explains. Values closer to 1 indicate stronger predictive
            performance. The chart compares only models with a saved R²;
            missing results are explicitly listed rather than treated as zero.
            """
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


    # ------------------------------------------------------------------------
    # GREEN TAXI
    # ------------------------------------------------------------------------

    with tabs[3]:

        cfg = PATHS["Green Taxi"]

        st.header(
            "🚕 Green Taxi"
        )

        dashboard_duration(
            cfg,
            "Green Taxi",
        )

        st.divider()

        dashboard_demand(
            cfg,
            "Green Taxi",
        )

        st.divider()

        dashboard_zone_demand(
            cfg,
            "Green Taxi",
        )

    # ------------------------------------------------------------------------
    # HVFHV
    # ------------------------------------------------------------------------

    with tabs[4]:

        cfg = PATHS["HVFHV (Uber/Lyft)"]

        st.header(
            "🚗 High Volume For-Hire Vehicle (Uber/Lyft)"
        )

        dashboard_duration(
            cfg,
            "HVFHV (Uber/Lyft)",
        )

        st.divider()

        dashboard_demand(
            cfg,
            "HVFHV (Uber/Lyft)",
        )

        st.divider()

        dashboard_zone_demand(
            cfg,
            "HVFHV (Uber/Lyft)",
        )


# ============================================================================
# PAGE 3 — DATA ANALYSIS REPORT
# ============================================================================

ANALYSIS_COLORS = [
    "#2563EB", "#7C3AED", "#DB2777", "#EA580C", "#059669",
    "#0891B2", "#CA8A04", "#4F46E5", "#16A34A", "#DC2626",
]


def _sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


@st.cache_data(show_spinner=False)
def analysis_schema(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["column_name", "column_type"])
    try:
        return duckdb.sql(
            f"DESCRIBE SELECT * FROM read_parquet('{_sql_path(path)}')"
        ).df()[["column_name", "column_type"]]
    except Exception:
        return pd.DataFrame(columns=["column_name", "column_type"])


def _pick_column(schema: pd.DataFrame, candidates) -> str | None:
    names = {str(x).lower(): str(x) for x in schema.get("column_name", [])}
    for candidate in candidates:
        if candidate.lower() in names:
            return names[candidate.lower()]
    for candidate in candidates:
        for low, original in names.items():
            if candidate.lower() in low:
                return original
    return None


def _qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


@st.cache_data(show_spinner=False)
def analysis_vehicle_monthly(vehicle: str) -> pd.DataFrame:
    cfg = PATHS[vehicle]
    path = cfg["hourly_demand"]
    if not path.exists():
        return pd.DataFrame(columns=["month", "month_name", "trips"])
    try:
        return duckdb.sql(
            f"""
            SELECT EXTRACT(month FROM pickup_hour_ts)::INTEGER AS month,
                   strftime(pickup_hour_ts, '%b') AS month_name,
                   SUM(trip_count) AS trips
            FROM read_parquet('{_sql_path(path)}')
            WHERE pickup_hour_ts >= TIMESTAMP '{YEAR_START.isoformat()}'
              AND pickup_hour_ts < TIMESTAMP '2026-01-01'
            GROUP BY 1, 2 ORDER BY 1
            """
        ).df()
    except Exception:
        return pd.DataFrame(columns=["month", "month_name", "trips"])


@st.cache_data(show_spinner=False)
def analysis_vehicle_hour_dow(vehicle: str) -> pd.DataFrame:
    path = PATHS[vehicle]["hourly_demand"]
    if not path.exists():
        return pd.DataFrame()
    try:
        return duckdb.sql(
            f"""
            SELECT EXTRACT(dow FROM pickup_hour_ts)::INTEGER AS dow,
                   EXTRACT(hour FROM pickup_hour_ts)::INTEGER AS hour,
                   SUM(trip_count) AS trips
            FROM read_parquet('{_sql_path(path)}')
            WHERE pickup_hour_ts >= TIMESTAMP '{YEAR_START.isoformat()}'
              AND pickup_hour_ts < TIMESTAMP '2026-01-01'
            GROUP BY 1, 2 ORDER BY 1, 2
            """
        ).df()
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def analysis_top_zones(vehicle: str, limit: int = 15) -> pd.DataFrame:
    path = PATHS[vehicle]["zone_hourly_demand"]
    if not path.exists():
        return pd.DataFrame()
    try:
        return duckdb.sql(
            f"""
            SELECT zone_id,
                   any_value(zone_name) AS zone_name,
                   any_value(borough) AS borough,
                   SUM(trip_count) AS trips
            FROM read_parquet('{_sql_path(path)}')
            WHERE pickup_hour_ts >= TIMESTAMP '{YEAR_START.isoformat()}'
              AND pickup_hour_ts < TIMESTAMP '2026-01-01'
            GROUP BY zone_id
            ORDER BY trips DESC
            LIMIT {int(limit)}
            """
        ).df()
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def analysis_vehicle_summary(vehicle: str) -> dict:
    monthly = analysis_vehicle_monthly(vehicle)
    hourly = analysis_vehicle_hour_dow(vehicle)
    zones = analysis_top_zones(vehicle, 15)
    out = {"trips": None, "busiest_month": None, "peak_hour": None,
           "peak_dow": None, "top_zone": None, "top_zone_trips": None}
    if not monthly.empty:
        out["trips"] = float(monthly["trips"].sum())
        row = monthly.loc[monthly["trips"].idxmax()]
        out["busiest_month"] = str(row["month_name"])
    if not hourly.empty:
        row = hourly.loc[hourly["trips"].idxmax()]
        out["peak_hour"] = int(row["hour"])
        out["peak_dow"] = int(row["dow"])
    if not zones.empty:
        out["top_zone"] = str(zones.iloc[0].get("zone_name", zones.iloc[0]["zone_id"]))
        out["top_zone_trips"] = float(zones.iloc[0]["trips"])
    return out


def _analysis_model_table() -> pd.DataFrame:
    rows = []
    for vehicle, cfg in PATHS.items():
        dur = duration_metrics_with_fallback(cfg["duration_metrics"], cfg["duration_predictions"])
        dem = sql_regression_metrics(cfg["demand_predictions"])
        zone = sql_regression_metrics(cfg["zone_demand_predictions"])
        rows += [
            {"Vehicle": vehicle, "Task": "Trip Duration", "MAE": dur.get("MAE") / 60 if dur.get("MAE") is not None else None,
             "RMSE": dur.get("RMSE") / 60 if dur.get("RMSE") is not None else None, "R²": dur.get("R2")},
            {"Vehicle": vehicle, "Task": "Citywide Demand", "MAE": dem.get("MAE") if dem else None,
             "RMSE": dem.get("RMSE") if dem else None, "R²": dem.get("R2") if dem else None},
            {"Vehicle": vehicle, "Task": "Zone Demand", "MAE": zone.get("MAE") if zone else None,
             "RMSE": zone.get("RMSE") if zone else None, "R²": zone.get("R2") if zone else None},
        ]
    return pd.DataFrame(rows)


def _analysis_peak_label(dow: int | None, hour: int | None) -> str:
    days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    if dow is None or hour is None:
        return "Not available"
    return f"{days[dow]} at {hour:02d}:00"


def page_data_analysis():
    st.header("📈 Data Analysis Report — NYC Taxi & For-Hire Vehicles")
    st.caption("A visual, evidence-based exploration of the four 2025 vehicle datasets, their demand patterns, trip characteristics, zones, and ML results.")

    summaries = {v: analysis_vehicle_summary(v) for v in PATHS}
    available = [v for v, s in summaries.items() if s["trips"] is not None]
    if not available:
        st.error("No processed demand datasets were found. Run the feature-building pipeline first.")
        return

    # ------------------------------------------------------------------
    # EXECUTIVE SCORECARD
    # ------------------------------------------------------------------
    total_city_trips = sum(summaries[v]["trips"] or 0 for v in available)
    leader = max(available, key=lambda v: summaries[v]["trips"] or 0)
    c = st.columns(5)
    c[0].metric("🚕 Vehicles analysed", len(available))
    c[1].metric("🧭 2025 trips", f"{total_city_trips:,.0f}")
    c[2].metric("🏆 Highest volume", leader)
    c[3].metric("📅 Peak month", max(available, key=lambda v: summaries[v]["trips"] or 0) and summaries[leader]["busiest_month"] or "N/A")
    c[4].metric("⏱️ Peak hour", _analysis_peak_label(summaries[leader]["peak_dow"], summaries[leader]["peak_hour"]))

    st.info("**How to read this page:** blue/purple bars compare scale, heatmaps reveal when demand is concentrated, tables show exact values, and R²/MAE/RMSE describe regression quality. Regression models do not have a conventional classification 'accuracy' percentage; R² is shown as the main goodness-of-fit measure.")

    tabs = st.tabs(["🌆 Executive Overview", "📅 Time Patterns", "📍 Zones & Geography", "🤖 Model Results", "🧠 Findings"])

    with tabs[0]:
        rows = []
        for v in available:
            s = summaries[v]
            rows.append({"Vehicle": v, "2025 Trips": s["trips"], "Share %": s["trips"] / total_city_trips * 100,
                         "Busiest Month": s["busiest_month"] or "N/A", "Peak Period": _analysis_peak_label(s["peak_dow"], s["peak_hour"])})
        overview = pd.DataFrame(rows).sort_values("2025 Trips", ascending=False)
        fig = px.bar(overview, x="Vehicle", y="2025 Trips", color="Vehicle", text="2025 Trips",
                     title="Total Trips by Vehicle Type — 2025", color_discrete_sequence=ANALYSIS_COLORS)
        fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig.update_layout(showlegend=False, height=450)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            pie = px.pie(overview, names="Vehicle", values="2025 Trips", hole=.52,
                         title="Share of Recorded Trips", color="Vehicle", color_discrete_sequence=ANALYSIS_COLORS)
            pie.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(pie, use_container_width=True)
        with col2:
            st.dataframe(overview.style.format({"2025 Trips": "{:,.0f}", "Share %": "{:.1f}%"}), use_container_width=True, hide_index=True)

        st.subheader("Monthly volume comparison")
        month_frames = []
        for v in available:
            m = analysis_vehicle_monthly(v)
            if not m.empty:
                m["Vehicle"] = v
                month_frames.append(m)
        if month_frames:
            mm = pd.concat(month_frames, ignore_index=True)
            fig = px.line(mm, x="month_name", y="trips", color="Vehicle", markers=True,
                          category_orders={"month_name": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]},
                          title="Month-by-Month Trip Volume", color_discrete_sequence=ANALYSIS_COLORS)
            fig.update_layout(yaxis_title="Trips", xaxis_title="Month")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.subheader("When do people travel? — Hour × Day-of-Week demand")
        vehicle = st.selectbox("Vehicle type", available, key="analysis_time_vehicle")
        hd = analysis_vehicle_hour_dow(vehicle)
        if not hd.empty:
            days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
            pivot = hd.pivot(index="hour", columns="dow", values="trips").fillna(0)
            pivot = pivot.reindex(index=range(24), columns=range(7), fill_value=0)
            pivot.columns = days
            heat = px.imshow(pivot, aspect="auto", color_continuous_scale="Turbo",
                             labels={"x":"Day", "y":"Hour of day", "color":"Trips"},
                             title=f"{vehicle} — Demand Heatmap")
            heat.update_layout(height=620)
            st.plotly_chart(heat, use_container_width=True)

            by_hour = hd.groupby("hour", as_index=False)["trips"].sum()
            fig = px.area(by_hour, x="hour", y="trips", markers=True, title=f"{vehicle} — 24-Hour Demand Profile")
            st.plotly_chart(fig, use_container_width=True)

            by_dow = hd.groupby("dow", as_index=False)["trips"].sum()
            by_dow["Day"] = by_dow["dow"].map(dict(enumerate(days)))
            fig = px.bar(by_dow, x="Day", y="trips", color="Day", title=f"{vehicle} — Demand by Day of Week", color_discrete_sequence=ANALYSIS_COLORS)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Monthly pattern table")
        m = analysis_vehicle_monthly(vehicle)
        if not m.empty:
            m2 = m.copy(); m2["trips"] = m2["trips"].round().astype("int64")
            st.dataframe(m2[["month_name", "trips"]].rename(columns={"month_name":"Month", "trips":"Trips"}), use_container_width=True, hide_index=True)

    with tabs[2]:
        st.subheader("Where are trips concentrated?")
        cols = st.columns(2)
        with cols[0]:
            vehicle = st.selectbox("Vehicle type", available, key="analysis_zone_vehicle")
        with cols[1]:
            n_zones = st.slider("Top zones to display", 5, 20, 10, key="analysis_zone_n")
        zones = analysis_top_zones(vehicle, n_zones)
        if not zones.empty:
            zones["label"] = zones["zone_name"].fillna(zones["zone_id"].astype(str)) + " (" + zones["borough"].fillna("Unknown") + ")"
            fig = px.bar(zones.sort_values("trips"), x="trips", y="label", orientation="h", color="trips",
                         color_continuous_scale="Turbo", title=f"Top {n_zones} Pickup Zones — {vehicle}")
            fig.update_layout(yaxis_title="Pickup zone", xaxis_title="Trips")
            st.plotly_chart(fig, use_container_width=True)
            display = zones[["zone_id", "zone_name", "borough", "trips"]].copy()
            display.columns = ["Zone ID", "Zone", "Borough", "Trips"]
            st.dataframe(display.style.format({"Trips":"{:,.0f}"}), use_container_width=True, hide_index=True)

        st.subheader("All-vehicle zone concentration comparison")
        compare = []
        for v in available:
            z = analysis_top_zones(v, 1)
            if not z.empty:
                compare.append({"Vehicle":v, "Busiest Zone":z.iloc[0]["zone_name"], "Borough":z.iloc[0]["borough"], "Trips":z.iloc[0]["trips"]})
        if compare:
            st.dataframe(pd.DataFrame(compare).style.format({"Trips":"{:,.0f}"}), use_container_width=True, hide_index=True)

    with tabs[3]:
        st.subheader("How strong are the prediction models?")
        model_df = _analysis_model_table()
        if not model_df.empty:
            st.dataframe(model_df.style.format({"MAE":"{:.2f}", "RMSE":"{:.2f}", "R²":"{:.3f}"}), use_container_width=True, hide_index=True)

            r2_df = model_df.dropna(subset=["R²"])
            if not r2_df.empty:
                fig = px.bar(r2_df, x="Vehicle", y="R²", color="Task", barmode="group",
                             title="R² Comparison Across Vehicles and Tasks", color_discrete_sequence=ANALYSIS_COLORS)
                fig.add_hline(y=0, line_dash="dash")
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("What do MAE, RMSE and R² mean?")
            explanation_box("<b>MAE</b> = average absolute error. <b>RMSE</b> penalizes large errors more strongly. <b>R²</b> measures how much variation the model explains relative to a simple mean baseline. For trip-duration models the dashboard converts MAE/RMSE from seconds to minutes. For demand models the error is trips per hour. There is no honest single 'accuracy %' for these regression tasks unless a separate tolerance-based accuracy definition is explicitly chosen.")

            # Feature importance cards where the saved metrics contain importances.
            for vehicle in available:
                metrics = duration_metrics_with_fallback(PATHS[vehicle]["duration_metrics"], PATHS[vehicle]["duration_predictions"])
                imp = metrics.get("importances") if metrics else None
                if imp is not None and not imp.empty:
                    top = imp.head(8).sort_values()
                    fig = px.bar(top.reset_index(), x=0, y="index", orientation="h", color=0,
                                 color_continuous_scale="Turbo", title=f"{vehicle} — Top Duration Features")
                    fig.update_layout(showlegend=False, yaxis_title="Feature", xaxis_title="Importance")
                    st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        st.subheader("🧠 Automatically generated project findings")
        all_summaries = summaries
        total = sum(s["trips"] or 0 for s in all_summaries.values())
        leader = max(available, key=lambda v: all_summaries[v]["trips"] or 0)
        smallest = min(available, key=lambda v: all_summaries[v]["trips"] or 0)
        busiest_month_vehicle = max(available, key=lambda v: analysis_vehicle_monthly(v)["trips"].max() if not analysis_vehicle_monthly(v).empty else 0)
        findings = [
            f"**Overall scale:** the four vehicle datasets contain approximately **{total:,.0f} recorded trips** in the 2025 analysis window represented by the processed demand files.",
            f"**Volume leader:** **{leader}** has the largest recorded trip volume among the four vehicle datasets, while **{smallest}** has the smallest.",
            f"**Strongest monthly spike:** {busiest_month_vehicle} reaches its highest monthly trip volume in **{all_summaries[busiest_month_vehicle]['busiest_month'] or 'the available peak month'}**.",
        ]
        for v in available:
            s = all_summaries[v]
            if s["top_zone"]:
                findings.append(f"**{v} geography:** the busiest pickup concentration is **{s['top_zone']}**, with about **{s['top_zone_trips']:,.0f} trips** in the zone-demand dataset.")
            if s["peak_hour"] is not None:
                findings.append(f"**{v} timing:** its strongest hour/day combination in the hourly demand data is **{_analysis_peak_label(s['peak_dow'], s['peak_hour'])}**.")
        st.markdown("\n\n".join(f"• {x}" for x in findings))

        st.subheader("What this project demonstrates")
        st.markdown("""
        • **Descriptive analytics:** where trips happen, when they happen, how volume changes month-to-month, and which vehicle types dominate the observed records.  
        • **Diagnostic analytics:** heatmaps, time-pattern charts and zone comparisons expose peak periods and concentration rather than only reporting totals.  
        • **Predictive analytics:** the project predicts trip duration, citywide hourly demand and zone-level hourly demand for each vehicle type.  
        • **Model validation:** actual-vs-predicted plots plus MAE, RMSE and R² make the model behavior visible rather than presenting a misleading 'accuracy' number.  
        • **Decision support:** the findings can be used to explain demand concentration, plan capacity, understand temporal peaks and identify where prediction errors are strongest.
        """)

        st.warning("These findings are calculated from the processed 2025 files available to the dashboard. They are intentionally generated from the data at runtime, so the dashboard will not hard-code generic NYC assumptions or fabricate missing statistics.")

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
            "📈 Data Analysis Report",
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

        12 trained ML models (2025 data only):

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

    elif page == "📊 Model Performance Dashboard":

        page_dashboard()

    else:

        page_data_analysis()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()