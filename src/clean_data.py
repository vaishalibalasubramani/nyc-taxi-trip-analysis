from database import get_connection


def clean_data():
    conn = get_connection()

    print("\n" + "=" * 60)
    print("                 DATA CLEANING")
    print("=" * 60)

    # ============================================================
    # 1. DROP CLEAN TABLE IF IT ALREADY EXISTS
    # ============================================================

    conn.execute("""
        DROP TABLE IF EXISTS taxi_trips_clean
    """)

    print("\nExisting clean table removed (if it existed).")

    # ============================================================
    # 2. CREATE CLEAN TABLE
    # ============================================================

    conn.execute("""
        CREATE TABLE taxi_trips_clean AS

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
            cbd_congestion_fee

        FROM taxi_trips

        WHERE
            -- Keep 2025 data only
            pickup_datetime >= TIMESTAMP '2025-01-01 00:00:00'
            AND pickup_datetime < TIMESTAMP '2026-01-01 00:00:00'

            -- Remove invalid timestamps
            AND dropoff_datetime >= pickup_datetime

            -- Remove negative fares
            AND fare_amount >= 0

            -- Remove negative total amounts
            AND total_amount >= 0
    """)

    print("Clean table created successfully.")

    # ============================================================
    # 3. COUNT CLEAN ROWS
    # ============================================================

    clean_count = conn.execute("""
        SELECT COUNT(*)
        FROM taxi_trips_clean
    """).fetchone()[0]

    print("\n=== CLEAN DATA COUNT ===")
    print("Clean rows:", clean_count)

    # ============================================================
    # 4. ROWS REMOVED
    # ============================================================

    raw_count = conn.execute("""
        SELECT COUNT(*)
        FROM taxi_trips
    """).fetchone()[0]

    removed_count = raw_count - clean_count

    print("Raw rows:", raw_count)
    print("Rows removed:", removed_count)

    # ============================================================
    # 5. CHECK DATE RANGE
    # ============================================================

    date_range = conn.execute("""
        SELECT
            MIN(pickup_datetime),
            MAX(pickup_datetime)
        FROM taxi_trips_clean
    """).fetchone()

    print("\n=== CLEAN DATE RANGE ===")
    print("Earliest pickup:", date_range[0])
    print("Latest pickup:", date_range[1])

    # ============================================================
    # 6. CHECK INVALID TIME VALUES
    # ============================================================

    invalid_times = conn.execute("""
        SELECT COUNT(*)
        FROM taxi_trips_clean
        WHERE dropoff_datetime < pickup_datetime
    """).fetchone()[0]

    print("\n=== INVALID TIME CHECK ===")
    print("Dropoff before pickup:", invalid_times)

    # ============================================================
    # 7. CHECK NEGATIVE FARES
    # ============================================================

    negative_fares = conn.execute("""
        SELECT COUNT(*)
        FROM taxi_trips_clean
        WHERE fare_amount < 0
    """).fetchone()[0]

    print("\n=== NEGATIVE FARE CHECK ===")
    print("Negative fares:", negative_fares)

    # ============================================================
    # 8. CHECK NEGATIVE TOTAL AMOUNTS
    # ============================================================

    negative_total = conn.execute("""
        SELECT COUNT(*)
        FROM taxi_trips_clean
        WHERE total_amount < 0
    """).fetchone()[0]

    print("\n=== NEGATIVE TOTAL AMOUNT CHECK ===")
    print("Negative total amounts:", negative_total)

    # ============================================================
    # 9. CHECK NEGATIVE DISTANCES
    # ============================================================

    negative_distance = conn.execute("""
        SELECT COUNT(*)
        FROM taxi_trips_clean
        WHERE trip_distance < 0
    """).fetchone()[0]

    print("\n=== NEGATIVE DISTANCE CHECK ===")
    print("Negative distances:", negative_distance)

    # ============================================================
    # 10. CHECK NULL VALUES
    # ============================================================

    null_check = conn.execute("""
        SELECT
            COUNT(*) AS total_rows,
            COUNT(*) - COUNT(pickup_datetime),
            COUNT(*) - COUNT(dropoff_datetime),
            COUNT(*) - COUNT(trip_distance),
            COUNT(*) - COUNT(fare_amount),
            COUNT(*) - COUNT(total_amount)
        FROM taxi_trips_clean
    """).fetchone()

    print("\n=== IMPORTANT NULL CHECK ===")
    print("Pickup datetime NULL:", null_check[1])
    print("Dropoff datetime NULL:", null_check[2])
    print("Trip distance NULL:", null_check[3])
    print("Fare amount NULL:", null_check[4])
    print("Total amount NULL:", null_check[5])

    # ============================================================
    # 11. CREATE TRIP DURATION COLUMN
    # ============================================================

    conn.execute("""
        ALTER TABLE taxi_trips_clean
        ADD COLUMN trip_duration_minutes DOUBLE
    """)

    conn.execute("""
        UPDATE taxi_trips_clean
        SET trip_duration_minutes =
            EPOCH(dropoff_datetime - pickup_datetime) / 60.0
    """)

    print("\nTrip duration column created.")

    # ============================================================
    # 12. CHECK TRIP DURATION
    # ============================================================

    duration_stats = conn.execute("""
        SELECT
            MIN(trip_duration_minutes),
            MAX(trip_duration_minutes),
            AVG(trip_duration_minutes)
        FROM taxi_trips_clean
    """).fetchone()

    print("\n=== TRIP DURATION ===")
    print("Minimum minutes:", duration_stats[0])
    print("Maximum minutes:", duration_stats[1])
    print("Average minutes:", duration_stats[2])

    # ============================================================
    # 13. FINAL VALIDATION
    # ============================================================

    print("\n" + "=" * 60)
    print("              CLEANING COMPLETED")
    print("=" * 60)

    print("Raw rows:", raw_count)
    print("Clean rows:", clean_count)
    print("Rows removed:", removed_count)

    print("\nFinal checks:")
    print("Invalid times:", invalid_times)
    print("Negative fares:", negative_fares)
    print("Negative total amounts:", negative_total)
    print("Negative distances:", negative_distance)

    conn.close()


if __name__ == "__main__":
    clean_data()