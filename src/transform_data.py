from database import get_connection


def transform_data():
    conn = get_connection()

    print("\n" + "=" * 60)
    print("              DATA TRANSFORMATION")
    print("=" * 60)

    # ============================================================
    # 1. DROP ANALYSIS TABLE IF IT EXISTS
    # ============================================================

    conn.execute("""
        DROP TABLE IF EXISTS taxi_trips_analysis
    """)

    print("\nExisting analysis table removed (if it existed).")

    # ============================================================
    # 2. CREATE ANALYSIS-READY TABLE
    # ============================================================

    conn.execute("""
        CREATE TABLE taxi_trips_analysis AS

        SELECT
            vendor_id,
            pickup_datetime,
            dropoff_datetime,
            passenger_count,
            trip_distance,
            rate_code_id,
            store_and_fwd_flag,
            pickup_location_id,
            dropoff_location_id,
            payment_type,
            fare_amount,
            extra,
            mta_tax,
            tip_amount,
            tolls_amount,
            improvement_surcharge,
            total_amount,
            congestion_surcharge,
            airport_fee,
            cbd_congestion_fee,

            trip_duration_minutes,

            -- Date
            CAST(pickup_datetime AS DATE) AS pickup_date,

            -- Year
            YEAR(pickup_datetime) AS pickup_year,

            -- Month number
            MONTH(pickup_datetime) AS pickup_month,

            -- Month name
            STRFTIME(pickup_datetime, '%B') AS pickup_month_name,

            -- Day of month
            DAY(pickup_datetime) AS pickup_day,

            -- Day name
            STRFTIME(pickup_datetime, '%A') AS pickup_day_name,

            -- Hour
            HOUR(pickup_datetime) AS pickup_hour,

            -- Trip duration in hours
            trip_duration_minutes / 60.0 AS trip_duration_hours,

            -- Average speed in miles per hour
            CASE
                WHEN trip_duration_minutes > 0
                     AND trip_distance > 0
                THEN
                    trip_distance / (trip_duration_minutes / 60.0)
                ELSE NULL
            END AS average_speed_mph,

            -- Tip percentage
            CASE
                WHEN fare_amount > 0
                THEN
                    (tip_amount / fare_amount) * 100.0
                ELSE NULL
            END AS tip_percentage

        FROM taxi_trips_clean
    """)

    print("Analysis table created successfully.")

    # ============================================================
    # 3. COUNT ROWS
    # ============================================================

    clean_count = conn.execute("""
        SELECT COUNT(*)
        FROM taxi_trips_clean
    """).fetchone()[0]

    analysis_count = conn.execute("""
        SELECT COUNT(*)
        FROM taxi_trips_analysis
    """).fetchone()[0]

    print("\n=== ROW COUNT CHECK ===")
    print("Clean rows:", clean_count)
    print("Analysis rows:", analysis_count)

    if clean_count == analysis_count:
        print("Row count check: PASSED")
    else:
        print("Row count check: FAILED")

    # ============================================================
    # 4. DATE RANGE
    # ============================================================

    date_range = conn.execute("""
        SELECT
            MIN(pickup_date),
            MAX(pickup_date)
        FROM taxi_trips_analysis
    """).fetchone()

    print("\n=== DATE RANGE ===")
    print("Earliest pickup date:", date_range[0])
    print("Latest pickup date:", date_range[1])

    # ============================================================
    # 5. MONTH CHECK
    # ============================================================

    month_count = conn.execute("""
        SELECT COUNT(DISTINCT pickup_month)
        FROM taxi_trips_analysis
    """).fetchone()[0]

    print("\n=== MONTH CHECK ===")
    print("Number of months:", month_count)

    # ============================================================
    # 6. DERIVED COLUMN CHECK
    # ============================================================

    derived_check = conn.execute("""
        SELECT
            COUNT(pickup_date),
            COUNT(pickup_year),
            COUNT(pickup_month),
            COUNT(pickup_month_name),
            COUNT(pickup_day),
            COUNT(pickup_day_name),
            COUNT(pickup_hour),
            COUNT(trip_duration_hours)
        FROM taxi_trips_analysis
    """).fetchone()

    print("\n=== DERIVED COLUMN CHECK ===")
    print("Pickup date values:", derived_check[0])
    print("Pickup year values:", derived_check[1])
    print("Pickup month values:", derived_check[2])
    print("Pickup month name values:", derived_check[3])
    print("Pickup day values:", derived_check[4])
    print("Pickup day name values:", derived_check[5])
    print("Pickup hour values:", derived_check[6])
    print("Trip duration hours values:", derived_check[7])

    # ============================================================
    # 7. SAMPLE RECORDS
    # ============================================================

    print("\n=== SAMPLE ANALYSIS RECORDS ===")

    sample_records = conn.execute("""
        SELECT
            pickup_datetime,
            dropoff_datetime,
            trip_distance,
            trip_duration_minutes,
            pickup_date,
            pickup_month,
            pickup_month_name,
            pickup_day_name,
            pickup_hour,
            average_speed_mph,
            fare_amount,
            tip_amount,
            tip_percentage,
            total_amount
        FROM taxi_trips_analysis
        LIMIT 5
    """).fetchall()

    for row in sample_records:
        print(row)

    # ============================================================
    # 8. FINAL MESSAGE
    # ============================================================

    print("\n" + "=" * 60)
    print("          DATA TRANSFORMATION COMPLETED")
    print("=" * 60)

    conn.close()


if __name__ == "__main__":
    transform_data()