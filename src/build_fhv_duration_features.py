"""
Build simple FHV trip-duration features for 2025.

Output:
    data/processed/fhv_duration_features.parquet

This is a separate feature-building pipeline for the FHV
trip-duration prediction model.

It does not modify the existing FHV demand/trip feature files.
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

REFERENCE_DIR = DATA_DIR / "reference"

PROCESSED_DIR = DATA_DIR / "processed"

ZONE_ZIP = REFERENCE_DIR / "taxi_zones.zip"

ZONE_EXTRACT_DIR = REFERENCE_DIR / "taxi_zones"

ZONE_CENTROIDS_FILE = (
    REFERENCE_DIR / "taxi_zone_centroids.csv"
)

TEMP_BASE_FILE = (
    PROCESSED_DIR / "_fhv_duration_base.parquet"
)

OUTPUT_FILE = (
    PROCESSED_DIR / "fhv_duration_features.parquet"
)


# ============================================================================
# OFFICIAL TLC URLS
# ============================================================================

BASE_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data"
)

ZONE_URL = (
    "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
)


# ============================================================================
# DIRECTORY SETUP
# ============================================================================

def ensure_directories():

    REFERENCE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================================
# BUILD REMOTE URLS
# ============================================================================

def build_remote_urls(months):

    return [
        f"{BASE_URL}/fhv_tripdata_{month}.parquet"
        for month in months
    ]


# ============================================================================
# DOWNLOAD TAXI ZONES
# ============================================================================

def download_zone_zip():

    if (
        ZONE_ZIP.exists()
        and ZONE_ZIP.stat().st_size > 0
    ):

        print(
            "Taxi-zone ZIP already exists:"
        )

        print(
            ZONE_ZIP
        )

        return

    print(
        "Downloading official TLC taxi-zone shapefile..."
    )

    print(
        ZONE_URL
    )

    print()

    urllib.request.urlretrieve(
        ZONE_URL,
        ZONE_ZIP
    )

    print(
        "Taxi-zone download complete."
    )


# ============================================================================
# LOAD TAXI ZONE CENTROIDS
# ============================================================================

def load_zone_centroids():

    print()
    print("=" * 72)
    print("LOADING NYC TAXI-ZONE GEOMETRIES")
    print("=" * 72)

    ensure_directories()

    download_zone_zip()

    print()

    # ------------------------------------------------------------------------
    # Remove previous extraction
    # ------------------------------------------------------------------------

    if ZONE_EXTRACT_DIR.exists():

        print(
            "Removing previous taxi-zone extraction..."
        )

        shutil.rmtree(
            ZONE_EXTRACT_DIR
        )

    ZONE_EXTRACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------------------
    # Extract ZIP
    # ------------------------------------------------------------------------

    print(
        "Extracting taxi-zone shapefile..."
    )

    with zipfile.ZipFile(
        ZONE_ZIP,
        "r"
    ) as archive:

        archive.extractall(
            ZONE_EXTRACT_DIR
        )

    print(
        "Extraction complete."
    )

    print()

    # ------------------------------------------------------------------------
    # Find SHP recursively
    # ------------------------------------------------------------------------

    shp_files = list(
        ZONE_EXTRACT_DIR.rglob("*.shp")
    )

    if not shp_files:

        raise FileNotFoundError(
            "No .shp file found inside "
            f"{ZONE_EXTRACT_DIR}"
        )

    shp_path = shp_files[0]

    print(
        "Found shapefile:"
    )

    print(
        shp_path
    )

    print()

    # ------------------------------------------------------------------------
    # Read zones
    # ------------------------------------------------------------------------

    print(
        "Reading taxi-zone geometries..."
    )

    zones = gpd.read_file(
        shp_path
    )

    print(
        f"Loaded {len(zones):,} taxi-zone geometries."
    )

    # ------------------------------------------------------------------------
    # Find LocationID
    # ------------------------------------------------------------------------

    location_column = None

    for candidate in [
        "LocationID",
        "locationid",
        "LOCATIONID",
        "location_id"
    ]:

        if candidate in zones.columns:

            location_column = candidate

            break

    if location_column is None:

        raise KeyError(
            "LocationID column not found.\n"
            f"Available columns: {list(zones.columns)}"
        )

    print(
        f"Location ID column: {location_column}"
    )

    # ------------------------------------------------------------------------
    # Remove invalid geometries
    # ------------------------------------------------------------------------

    zones = zones[
        zones.geometry.notna()
        &
        ~zones.geometry.is_empty
    ].copy()

    print(
        f"Valid geometries remaining: {len(zones):,}"
    )

    # ------------------------------------------------------------------------
    # CRS
    # ------------------------------------------------------------------------

    if zones.crs is None:

        raise ValueError(
            "Taxi-zone shapefile has no CRS."
        )

    print(
        f"Original CRS: {zones.crs}"
    )

    # ------------------------------------------------------------------------
    # Project to NYC State Plane
    # ------------------------------------------------------------------------

    zones_projected = zones.to_crs(
        epsg=2263
    )

    # ------------------------------------------------------------------------
    # Calculate centroids
    # ------------------------------------------------------------------------

    zones_projected["centroid"] = (
        zones_projected.geometry.centroid
    )

    # ------------------------------------------------------------------------
    # Convert centroids to WGS84
    # ------------------------------------------------------------------------

    centroids = gpd.GeoDataFrame(
        zones_projected[
            [
                location_column,
                "centroid"
            ]
        ].copy(),
        geometry="centroid",
        crs="EPSG:2263"
    )

    centroids = centroids.to_crs(
        epsg=4326
    )

    # ------------------------------------------------------------------------
    # Latitude / longitude
    # ------------------------------------------------------------------------

    centroids["zone_latitude"] = (
        centroids.geometry.y
    )

    centroids["zone_longitude"] = (
        centroids.geometry.x
    )

    centroids["LocationID"] = pd.to_numeric(
        centroids[location_column],
        errors="coerce"
    )

    # ------------------------------------------------------------------------
    # Keep only required columns
    # ------------------------------------------------------------------------

    centroids = centroids[
        [
            "LocationID",
            "zone_latitude",
            "zone_longitude"
        ]
    ].copy()

    centroids = centroids.dropna(
        subset=[
            "LocationID",
            "zone_latitude",
            "zone_longitude"
        ]
    )

    centroids["LocationID"] = (
        centroids["LocationID"]
        .astype(int)
    )

    centroids = (
        centroids
        .drop_duplicates(
            subset=["LocationID"]
        )
        .sort_values(
            "LocationID"
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------------

    centroids.to_csv(
        ZONE_CENTROIDS_FILE,
        index=False
    )

    print()

    print(
        f"Saved {len(centroids):,} zone centroids:"
    )

    print(
        ZONE_CENTROIDS_FILE
    )

    print()

    print(
        "Centroid preview:"
    )

    print(
        centroids
        .head(10)
        .to_string(index=False)
    )

    print()

    return centroids


# ============================================================================
# HAVERSINE DISTANCE
# ============================================================================

def haversine_miles(
    lat1,
    lon1,
    lat2,
    lon2
):
    """
    Calculate great-circle distance between two coordinates.

    Returns:
        Distance in miles.

    IMPORTANT:
        This implementation uses NumPy functions.
        Pandas Series do NOT have .sin() or .cos() methods.
    """

    # ------------------------------------------------------------------------
    # Convert to numeric
    # ------------------------------------------------------------------------

    lat1 = pd.to_numeric(
        lat1,
        errors="coerce"
    )

    lon1 = pd.to_numeric(
        lon1,
        errors="coerce"
    )

    lat2 = pd.to_numeric(
        lat2,
        errors="coerce"
    )

    lon2 = pd.to_numeric(
        lon2,
        errors="coerce"
    )

    # ------------------------------------------------------------------------
    # Convert degrees to radians
    # ------------------------------------------------------------------------

    lat1_rad = np.radians(
        lat1
    )

    lon1_rad = np.radians(
        lon1
    )

    lat2_rad = np.radians(
        lat2
    )

    lon2_rad = np.radians(
        lon2
    )

    # ------------------------------------------------------------------------
    # Differences
    # ------------------------------------------------------------------------

    dlat = (
        lat2_rad
        -
        lat1_rad
    )

    dlon = (
        lon2_rad
        -
        lon1_rad
    )

    # ------------------------------------------------------------------------
    # Haversine formula
    #
    # ONLY NumPy trigonometric functions are used here.
    # ------------------------------------------------------------------------

    a = (
        np.sin(
            dlat / 2.0
        ) ** 2
        +
        np.cos(
            lat1_rad
        )
        *
        np.cos(
            lat2_rad
        )
        *
        np.sin(
            dlon / 2.0
        ) ** 2
    )

    # ------------------------------------------------------------------------
    # Numerical protection
    # ------------------------------------------------------------------------

    a = np.clip(
        a,
        0.0,
        1.0
    )

    # ------------------------------------------------------------------------
    # Angular distance
    # ------------------------------------------------------------------------

    c = (
        2.0
        *
        np.arcsin(
            np.sqrt(a)
        )
    )

    # ------------------------------------------------------------------------
    # Earth radius
    # ------------------------------------------------------------------------

    earth_radius_miles = 3958.7613

    return (
        earth_radius_miles
        *
        c
    )


# ============================================================================
# BUILD BASE FHV DATA
# ============================================================================

def build_base_fhv_data(
    con,
    months
):

    print()
    print("=" * 72)
    print("READING FHV DATA")
    print("=" * 72)

    urls = build_remote_urls(
        months
    )

    for month, url in zip(
        months,
        urls
    ):

        print(
            f"{month}:"
        )

        print(
            f"  {url}"
        )

    print()

    print(
        "Reading FHV Parquet files directly from TLC CloudFront..."
    )

    print(
        "This can take some time."
    )

    print()

    # ------------------------------------------------------------------------
    # DuckDB URL list
    # ------------------------------------------------------------------------

    url_list = (
        "["
        +
        ", ".join(
            "'"
            +
            url.replace(
                "'",
                "''"
            )
            +
            "'"
            for url in urls
        )
        +
        "]"
    )

    # ------------------------------------------------------------------------
    # Base query
    #
    # Uses the exact original FHV column names:
    #
    # PUlocationID
    # DOlocationID
    #
    # ------------------------------------------------------------------------

    query = f"""
    COPY (

        SELECT

            pickup_datetime,

            dropOff_datetime,

            PUlocationID,

            DOlocationID,

            COALESCE(
                SR_Flag,
                0
            ) AS SR_Flag,

            dispatching_base_num,

            Affiliated_base_number,

            EXTRACT(
                EPOCH FROM
                (
                    dropOff_datetime
                    -
                    pickup_datetime
                )
            ) AS trip_duration_s

        FROM read_parquet(
            {url_list},
            union_by_name = true
        )

        WHERE

            pickup_datetime IS NOT NULL

            AND dropOff_datetime IS NOT NULL

            AND PUlocationID IS NOT NULL

            AND DOlocationID IS NOT NULL

            AND dropOff_datetime > pickup_datetime

            AND EXTRACT(
                EPOCH FROM
                (
                    dropOff_datetime
                    -
                    pickup_datetime
                )
            ) BETWEEN 60 AND 10800

            AND pickup_datetime >= TIMESTAMP '2025-01-01'

            AND pickup_datetime < TIMESTAMP '2026-01-01'

    )

    TO '{TEMP_BASE_FILE.as_posix()}'

    (
        FORMAT PARQUET,
        COMPRESSION ZSTD
    );
    """

    print(
        "Executing DuckDB query..."
    )

    print()

    con.execute(
        query
    )

    print(
        "Base FHV table created:"
    )

    print(
        TEMP_BASE_FILE
    )

    print()


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(
    base_df,
    centroids
):
    """
    Create the simple FHV trip-duration feature set.

    Standard FHV records do not contain a recorded trip_distance field.
    Therefore, trip_distance is estimated from the great-circle distance
    between the pickup and drop-off taxi-zone centroids.

    Final model features:
        trip_distance
        PUlocationID
        DOlocationID
        SR_Flag
        pickup_hour
        pickup_dow
        is_weekend
        pickup_month
    """

    print()
    print("=" * 72)
    print("ENGINEERING SIMPLE FHV DURATION FEATURES")
    print("=" * 72)

    df = base_df.copy()

    print(f"Initial rows: {len(df):,}")

    # -------------------------------------------------------------------------
    # DATETIME
    # -------------------------------------------------------------------------

    print("Converting datetime columns...")

    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"],
        errors="coerce"
    )

    df["dropOff_datetime"] = pd.to_datetime(
        df["dropOff_datetime"],
        errors="coerce"
    )

    # -------------------------------------------------------------------------
    # REQUIRED VALUES
    # -------------------------------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=[
            "pickup_datetime",
            "dropOff_datetime",
            "PUlocationID",
            "DOlocationID",
            "trip_duration_s"
        ]
    ).copy()

    print(
        "After removing missing required values: "
        f"{len(df):,} ({before - len(df):,} removed)"
    )

    # -------------------------------------------------------------------------
    # 2025
    # -------------------------------------------------------------------------

    before = len(df)

    df = df[
        (df["pickup_datetime"] >= pd.Timestamp("2025-01-01"))
        &
        (df["pickup_datetime"] < pd.Timestamp("2026-01-01"))
    ].copy()

    print(f"After restricting to 2025: {len(df):,}")

    # -------------------------------------------------------------------------
    # DURATION
    # -------------------------------------------------------------------------

    before = len(df)

    df = df[
        (df["trip_duration_s"] >= 60)
        &
        (df["trip_duration_s"] <= 10800)
    ].copy()

    print(
        "After duration filtering (1 minute to 3 hours): "
        f"{len(df):,} ({before - len(df):,} removed)"
    )

    # -------------------------------------------------------------------------
    # LOCATION IDS
    # -------------------------------------------------------------------------

    df["PUlocationID"] = pd.to_numeric(
        df["PUlocationID"],
        errors="coerce"
    )

    df["DOlocationID"] = pd.to_numeric(
        df["DOlocationID"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["PUlocationID", "DOlocationID"]
    ).copy()

    df["PUlocationID"] = df["PUlocationID"].astype(int)
    df["DOlocationID"] = df["DOlocationID"].astype(int)

    # -------------------------------------------------------------------------
    # SR FLAG
    # -------------------------------------------------------------------------

    df["SR_Flag"] = pd.to_numeric(
        df["SR_Flag"],
        errors="coerce"
    ).fillna(0)

    # -------------------------------------------------------------------------
    # SIMPLE TIME FEATURES
    # -------------------------------------------------------------------------

    print("Creating simple time features...")

    df["pickup_hour"] = (
        df["pickup_datetime"].dt.hour.astype(int)
    )

    df["pickup_dow"] = (
        df["pickup_datetime"].dt.dayofweek.astype(int)
    )

    df["pickup_month"] = (
        df["pickup_datetime"].dt.month.astype(int)
    )

    df["is_weekend"] = (
        df["pickup_dow"].isin([5, 6]).astype(int)
    )

    # -------------------------------------------------------------------------
    # PICKUP / DROPOFF CENTROIDS
    # -------------------------------------------------------------------------

    print("Merging pickup-zone centroids...")

    pickup_centroids = centroids.rename(
        columns={
            "LocationID": "PUlocationID",
            "zone_latitude": "pickup_zone_latitude",
            "zone_longitude": "pickup_zone_longitude",
        }
    )

    df = df.merge(
        pickup_centroids[
            [
                "PUlocationID",
                "pickup_zone_latitude",
                "pickup_zone_longitude",
            ]
        ],
        on="PUlocationID",
        how="left",
    )

    print("Merging dropoff-zone centroids...")

    dropoff_centroids = centroids.rename(
        columns={
            "LocationID": "DOlocationID",
            "zone_latitude": "dropoff_zone_latitude",
            "zone_longitude": "dropoff_zone_longitude",
        }
    )

    df = df.merge(
        dropoff_centroids[
            [
                "DOlocationID",
                "dropoff_zone_latitude",
                "dropoff_zone_longitude",
            ]
        ],
        on="DOlocationID",
        how="left",
    )

    coordinate_available = (
        df["pickup_zone_latitude"].notna()
        &
        df["pickup_zone_longitude"].notna()
        &
        df["dropoff_zone_latitude"].notna()
        &
        df["dropoff_zone_longitude"].notna()
    )

    print(
        "Rows with both pickup and dropoff zone coordinates: "
        f"{coordinate_available.sum():,} / {len(df):,}"
    )

    # -------------------------------------------------------------------------
    # ESTIMATED TRIP DISTANCE
    # -------------------------------------------------------------------------

    print(
        "Calculating estimated trip distance from taxi-zone centroids..."
    )

    df["trip_distance"] = haversine_miles(
        df["pickup_zone_latitude"],
        df["pickup_zone_longitude"],
        df["dropoff_zone_latitude"],
        df["dropoff_zone_longitude"],
    )

    # We require a valid distance because it is a model feature.
    before = len(df)

    df = df.dropna(
        subset=["trip_distance"]
    ).copy()

    print(
        "Rows after requiring valid trip distance: "
        f"{len(df):,} ({before - len(df):,} removed)"
    )

    # -------------------------------------------------------------------------
    # DISTANCE SANITY CHECK
    # -------------------------------------------------------------------------

    before = len(df)

    df = df[
        (df["trip_distance"] >= 0)
        &
        (df["trip_distance"] <= 100)
    ].copy()

    print(
        "Rows after trip-distance sanity check: "
        f"{len(df):,} ({before - len(df):,} removed)"
    )

    # -------------------------------------------------------------------------
    # FINAL SIMPLE MODEL COLUMNS
    # -------------------------------------------------------------------------

    final_columns = [
        "trip_distance",
        "PUlocationID",
        "DOlocationID",
        "SR_Flag",
        "pickup_hour",
        "pickup_dow",
        "is_weekend",
        "pickup_month",
        "trip_duration_s",
        "pickup_datetime",
    ]

    df = df[final_columns].copy()

    # -------------------------------------------------------------------------
    # DUPLICATES
    # -------------------------------------------------------------------------

    before = len(df)

    df = (
        df
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print(
        "Duplicate rows removed: "
        f"{before - len(df):,}"
    )

    # -------------------------------------------------------------------------
    # SORT
    # -------------------------------------------------------------------------

    df = (
        df
        .sort_values("pickup_datetime")
        .reset_index(drop=True)
    )

    return df


# ============================================================================
# DATASET SUMMARY
# ============================================================================

def print_summary(
    df
):

    print()
    print("=" * 72)
    print("FINAL FHV DURATION DATASET SUMMARY")
    print("=" * 72)

    print()

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns):,}"
    )

    # ------------------------------------------------------------------------
    # Columns
    # ------------------------------------------------------------------------

    print()

    print(
        "Columns:"
    )

    for column in df.columns:

        print(
            f"  - {column}"
        )

    # ------------------------------------------------------------------------
    # Duration
    # ------------------------------------------------------------------------

    duration_minutes = (
        df["trip_duration_s"]
        /
        60.0
    )

    print()

    print(
        "Trip duration statistics:"
    )

    print(
        f"  Mean:   {duration_minutes.mean():.2f} minutes"
    )

    print(
        f"  Median: {duration_minutes.median():.2f} minutes"
    )

    print(
        f"  Std:    {duration_minutes.std():.2f} minutes"
    )

    print(
        f"  Min:    {duration_minutes.min():.2f} minutes"
    )

    print(
        f"  Max:    {duration_minutes.max():.2f} minutes"
    )

    # ------------------------------------------------------------------------
    # Distance
    # ------------------------------------------------------------------------

    distance = (
        df["trip_distance"]
    )

    print()

    print(
        "Estimated trip-distance statistics:"
    )

    print(
        f"  Mean:   {distance.mean():.2f} miles"
    )

    print(
        f"  Median: {distance.median():.2f} miles"
    )

    print(
        f"  Min:    {distance.min():.2f} miles"
    )

    print(
        f"  Max:    {distance.max():.2f} miles"
    )

    # ------------------------------------------------------------------------
    # Weekend
    # ------------------------------------------------------------------------

    weekend_count = int(
        df["is_weekend"].sum()
    )

    print()

    print(
        "Weekend trips:"
    )

    print(
        f"  {weekend_count:,} "
        f"({weekend_count / len(df) * 100:.2f}%)"
    )

    # ------------------------------------------------------------------------
    # Top pickup zones
    # ------------------------------------------------------------------------

    print()

    print(
        "Top pickup zones:"
    )

    print(
        df["PUlocationID"]
        .value_counts()
        .head(10)
        .to_string()
    )

    # ------------------------------------------------------------------------
    # Top dropoff zones
    # ------------------------------------------------------------------------

    print()

    print(
        "Top dropoff zones:"
    )

    print(
        df["DOlocationID"]
        .value_counts()
        .head(10)
        .to_string()
    )


# ============================================================================
# SAVE
# ============================================================================

def save_dataset(
    df
):

    print()
    print("=" * 72)
    print("SAVING FHV DURATION FEATURES")
    print("=" * 72)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_parquet(
        OUTPUT_FILE,
        index=False,
        engine="pyarrow",
        compression="zstd"
    )

    print()

    print(
        "Saved:"
    )

    print(
        OUTPUT_FILE
    )

    print()

    size_mb = (
        OUTPUT_FILE.stat().st_size
        /
        1024
        /
        1024
    )

    print(
        f"File size: {size_mb:.2f} MB"
    )


# ============================================================================
# CLEAN TEMPORARY FILE
# ============================================================================

def cleanup_temp():

    if TEMP_BASE_FILE.exists():

        try:

            TEMP_BASE_FILE.unlink()

            print()

            print(
                "Removed temporary file:"
            )

            print(
                TEMP_BASE_FILE
            )

        except Exception as exc:

            print()

            print(
                "Warning: could not remove temporary file:"
            )

            print(
                TEMP_BASE_FILE
            )

            print(
                f"Error: {exc}"
            )


# ============================================================================
# VALIDATE MONTHS
# ============================================================================

def validate_months(
    months
):

    validated = []

    for month in months:

        parts = month.split("-")

        if len(parts) != 2:

            raise ValueError(
                f"Invalid month format: {month}. "
                "Expected YYYY-MM."
            )

        year = parts[0]

        month_number = parts[1]

        if (
            len(year) != 4
            or not year.isdigit()
            or not month_number.isdigit()
        ):

            raise ValueError(
                f"Invalid month format: {month}. "
                "Expected YYYY-MM."
            )

        month_int = int(
            month_number
        )

        if (
            month_int < 1
            or month_int > 12
        ):

            raise ValueError(
                f"Invalid month: {month}"
            )

        validated.append(
            f"{year}-{month_int:02d}"
        )

    return validated


# ============================================================================
# ARGUMENTS
# ============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Build simple FHV duration features."
        )
    )

    parser.add_argument(
        "--months",
        nargs="+",
        required=True,
        help=(
            "Months to process, "
            "for example: "
            "2025-01 2025-02"
        )
    )

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main():

    args = parse_args()

    months = validate_months(
        args.months
    )

    print()
    print("=" * 72)
    print("ENHANCED FHV DURATION FEATURE BUILDER")
    print("=" * 72)

    print()

    print(
        "Months:"
    )

    print(
        ", ".join(months)
    )

    print()

    # =========================================================================
    # DIRECTORIES
    # =========================================================================

    ensure_directories()

    # =========================================================================
    # TAXI ZONE CENTROIDS
    # =========================================================================

    centroids = (
        load_zone_centroids()
    )

    # =========================================================================
    # DUCKDB
    # =========================================================================

    print(
        "Configuring DuckDB HTTP access..."
    )

    con = duckdb.connect(config={
        "custom_user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36"
    })

    try:

        try:

            con.execute(
                "INSTALL httpfs;"
            )

        except Exception:

            pass

        con.execute(
            "LOAD httpfs;"
        )


        # ---------------------------------------------------------------------
        # Build base data
        # ---------------------------------------------------------------------

        build_base_fhv_data(
            con,
            months
        )

    finally:

        con.close()

    # =========================================================================
    # LOAD BASE DATA
    # =========================================================================

    print()
    print("=" * 72)
    print("LOADING TEMPORARY FHV DATA")
    print("=" * 72)

    base_df = pd.read_parquet(
        TEMP_BASE_FILE
    )

    print(
        f"Loaded {len(base_df):,} rows."
    )

    # =========================================================================
    # FEATURE ENGINEERING
    # =========================================================================

    final_df = engineer_features(
        base_df,
        centroids
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print_summary(
        final_df
    )

    # =========================================================================
    # SAVE
    # =========================================================================

    save_dataset(
        final_df
    )

    # =========================================================================
    # CLEAN TEMP
    # =========================================================================

    cleanup_temp()

    # =========================================================================
    # COMPLETE
    # =========================================================================

    print()
    print("=" * 72)
    print("FHV DURATION FEATURE BUILD COMPLETE")
    print("=" * 72)

    print()

    print(
        "Output:"
    )

    print(
        OUTPUT_FILE
    )

    print()

    print(
        "Enhanced FHV duration features are ready."
    )

    print()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    main()