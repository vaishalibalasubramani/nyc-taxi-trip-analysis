from database import get_connection


def validate_clean_data():
    conn = get_connection()

    print("\n" + "=" * 60)
    print("              CLEAN DATA VALIDATION")
    print("=" * 60)

    # ============================================================
    # 1. TOTAL ROWS
    # ============================================================

    total_rows = conn.execute("""
        SELECT COUNT(*)
        FROM taxi_trips_clean
    """).fetchone()[0]

    print("\n=== 1. TOTAL CLEAN ROWS ===")
    print("Clean rows:", total_rows)

    # ============================================================
    # 2. DATE RANGE
    # ============================================================

    date_range = conn.execute("""
        SELECT
            MIN(pickup_datetime),
            MAX(pickup_datetime)
        FROM taxi_trips_clean
    """).fetchone()

    print("\n=== 2. DATE RANGE ===")
    print("Earliest pickup:", date_range[0])
    print("Latest pickup:", date_range[1])

    # ============================================================
    # 3. MONTHLY TRIP COUNTS
    # ============================================================

    monthly_counts = conn.execute("""
        SELECT
            YEAR(pickup_datetime) AS year,
            MONTH(pickup_datetime) AS month,
            COUNT(*) AS trips
        FROM taxi_trips_clean
        GROUP BY
            YEAR(pickup_datetime),
            MONTH(pickup_datetime)
        ORDER BY
            year,
            month
    """).fetchall()

    print("\n=== 3. MONTHLY TRIP COUNTS ===")

    for row in monthly_counts:
        print(
            f"Year: {row[0]} | "
            f"Month: {row[1]:02d} | "
            f"Trips: {row[2]:,}"
        )

    # ============================================================
    # 4. NULL VALUES
    # ============================================================

    null_counts = conn.execute("""
        SELECT
            COUNT(*) - COUNT(vendor_id),
            COUNT(*) - COUNT(pickup_datetime),
            COUNT(*) - COUNT(dropoff_datetime),
            COUNT(*) - COUNT(passenger_count),
            COUNT(*) - COUNT(trip_distance),
            COUNT(*) - COUNT(rate_code_id),
            COUNT(*) - COUNT(store_and_fwd_flag),
            COUNT(*) - COUNT(pickup_location_id),
            COUNT(*) - COUNT(dropoff_location_id),
            COUNT(*) - COUNT(payment_type),
            COUNT(*) - COUNT(fare_amount),
            COUNT(*) - COUNT(extra),
            COUNT(*) - COUNT(mta_tax),
            COUNT(*) - COUNT(tip_amount),
            COUNT(*) - COUNT(tolls_amount),
            COUNT(*) - COUNT(improvement_surcharge),
            COUNT(*) - COUNT(total_amount),
            COUNT(*) - COUNT(congestion_surcharge),
            COUNT(*) - COUNT(airport_fee),
            COUNT(*) - COUNT(cbd_congestion_fee),
            COUNT(*) - COUNT(trip_duration_minutes)
        FROM taxi_trips_clean
    """).fetchone()

    column_names = [
        "vendor_id",
        "pickup_datetime",
        "dropoff_datetime",
        "passenger_count",
        "trip_distance",
        "rate_code_id",
        "store_and_fwd_flag",
        "pickup_location_id",
        "dropoff_location_id",
        "payment_type",
        "fare_amount",
        "extra",
        "mta_tax",
        "tip_amount",
        "tolls_amount",
        "improvement_surcharge",
        "total_amount",
        "congestion_surcharge",
        "airport_fee",
        "cbd_congestion_fee",
        "trip_duration_minutes"
    ]

    print("\n=== 4. NULL VALUES ===")

    for name, count in zip(column_names, null_counts):
        print(f"{name}: {count}")

    # ============================================================
    # 5. NEGATIVE VALUES
    # ============================================================

    negative_values = conn.execute("""
        SELECT
            COUNT(*) FILTER (WHERE trip_distance < 0),
            COUNT(*) FILTER (WHERE fare_amount < 0),
            COUNT(*) FILTER (WHERE total_amount < 0),
            COUNT(*) FILTER (WHERE passenger_count < 0)
        FROM taxi_trips_clean
    """).fetchone()

    print("\n=== 5. NEGATIVE VALUES ===")
    print("Negative trip distances:", negative_values[0])
    print("Negative fares:", negative_values[1])
    print("Negative total amounts:", negative_values[2])
    print("Negative passenger counts:", negative_values[3])

    # ============================================================
    # 6. PASSENGER COUNT
    # ============================================================

    passenger_stats = conn.execute("""
        SELECT
            MIN(passenger_count),
            MAX(passenger_count),
            AVG(passenger_count)
        FROM taxi_trips_clean
    """).fetchone()

    print("\n=== 6. PASSENGER COUNT ===")
    print("Minimum:", passenger_stats[0])
    print("Maximum:", passenger_stats[1])
    print("Average:", passenger_stats[2])

    # ============================================================
    # 7. ZERO PASSENGER COUNT
    # ============================================================

    zero_passengers = conn.execute("""
        SELECT COUNT(*)
        FROM taxi_trips_clean
        WHERE passenger_count = 0
    """).fetchone()[0]

    print("\n=== 7. ZERO PASSENGER COUNT ===")
    print("Zero passenger trips:", zero_passengers)

    # ============================================================
    # 8. ZERO DISTANCE
    # ============================================================

    zero_distance = conn.execute("""
        SELECT COUNT(*)
        FROM taxi_trips_clean
        WHERE trip_distance = 0
    """).fetchone()[0]

    print("\n=== 8. ZERO DISTANCE ===")
    print("Zero-distance trips:", zero_distance)

    # ============================================================
    # 9. ZERO FARE
    # ============================================================

    zero_fare = conn.execute("""
        SELECT COUNT(*)
        FROM taxi_trips_clean
        WHERE fare_amount = 0
    """).fetchone()[0]

    print("\n=== 9. ZERO FARE ===")
    print("Zero-fare trips:", zero_fare)

    # ============================================================
    # 10. TRIP DURATION
    # ============================================================

    duration_stats = conn.execute("""
        SELECT
            MIN(trip_duration_minutes),
            MAX(trip_duration_minutes),
            AVG(trip_duration_minutes)
        FROM taxi_trips_clean
    """).fetchone()

    print("\n=== 10. TRIP DURATION ===")
    print("Minimum minutes:", duration_stats[0])
    print("Maximum minutes:", duration_stats[1])
    print("Average minutes:", duration_stats[2])

    # ============================================================
    # 11. INVALID TIME VALUES
    # ============================================================

    invalid_times = conn.execute("""
        SELECT COUNT(*)
        FROM taxi_trips_clean
        WHERE dropoff_datetime < pickup_datetime
    """).fetchone()[0]

    print("\n=== 11. INVALID TIME VALUES ===")
    print("Dropoff before pickup:", invalid_times)

    # ============================================================
    # 12. ZERO-DURATION TRIPS
    # ============================================================

    zero_duration = conn.execute("""
        SELECT COUNT(*)
        FROM taxi_trips_clean
        WHERE trip_duration_minutes = 0
    """).fetchone()[0]

    print("\n=== 12. ZERO-DURATION TRIPS ===")
    print("Zero-duration trips:", zero_duration)

    # ============================================================
    # 13. VERY SHORT TRIPS
    # ============================================================

    very_short = conn.execute("""
        SELECT COUNT(*)
        FROM taxi_trips_clean
        WHERE trip_duration_minutes > 0
          AND trip_duration_minutes < 1
    """).fetchone()[0]

    print("\n=== 13. VERY SHORT TRIPS ===")
    print("Trips shorter than 1 minute:", very_short)

    # ============================================================
    # 14. VERY LONG TRIPS
    # ============================================================

    very_long = conn.execute("""
        SELECT COUNT(*)
        FROM taxi_trips_clean
        WHERE trip_duration_minutes > 180
    """).fetchone()[0]

    print("\n=== 14. VERY LONG TRIPS ===")
    print("Trips longer than 3 hours:", very_long)

    # ============================================================
    # 15. FINAL SUMMARY
    # ============================================================

    print("\n" + "=" * 60)
    print("              VALIDATION SUMMARY")
    print("=" * 60)

    print("Clean rows:", total_rows)
    print("Invalid times:", invalid_times)
    print("Negative fares:", negative_values[1])
    print("Negative distances:", negative_values[0])
    print("Zero passengers:", zero_passengers)
    print("Zero distance:", zero_distance)
    print("Zero fare:", zero_fare)
    print("Very long trips (>3 hours):", very_long)

    print("\nClean data validation completed successfully.")

    conn.close()


if __name__ == "__main__":
    validate_clean_data()