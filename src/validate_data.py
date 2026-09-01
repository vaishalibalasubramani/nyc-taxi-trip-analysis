from database import get_connection


def validate_data():
    conn = get_connection()

    print("\n" + "=" * 60)
    print("                 DATA VALIDATION")
    print("=" * 60)

    # ============================================================
    # 1. TOTAL ROW COUNT
    # ============================================================

    total_rows = conn.execute("""
        SELECT COUNT(*)
        FROM taxi_trips
    """).fetchone()[0]

    print("\n=== 1. TOTAL ROWS ===")
    print("Total rows:", total_rows)

    # ============================================================
    # 2. DATE RANGE
    # ============================================================

    date_range = conn.execute("""
        SELECT
            MIN(pickup_datetime),
            MAX(pickup_datetime)
        FROM taxi_trips
    """).fetchone()

    print("\n=== 2. DATE RANGE ===")
    print("Earliest pickup:", date_range[0])
    print("Latest pickup:", date_range[1])

    # ============================================================
    # 3. NULL VALUES
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
            COUNT(*) - COUNT(cbd_congestion_fee)
        FROM taxi_trips
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
        "cbd_congestion_fee"
    ]

    print("\n=== 3. NULL VALUES ===")

    for column, count in zip(column_names, null_counts):
        print(f"{column}: {count}")

    # ============================================================
    # 4. NEGATIVE TRIP DISTANCE
    # ============================================================

    negative_distance = conn.execute("""
        SELECT COUNT(*)
        FROM taxi_trips
        WHERE trip_distance < 0
    """).fetchone()[0]

    print("\n=== 4. NEGATIVE TRIP DISTANCE ===")
    print("Negative trip distances:", negative_distance)

    # ============================================================
    # 5. NEGATIVE FARES
    # ============================================================

    negative_fare = conn.execute("""
        SELECT COUNT(*)
        FROM taxi_trips
        WHERE fare_amount < 0
    """).fetchone()[0]

    print("\n=== 5. NEGATIVE FARES ===")
    print("Negative fares:", negative_fare)

    if negative_fare > 0:

        negative_fare_details = conn.execute("""
            SELECT
                COUNT(*),
                MIN(fare_amount),
                MAX(fare_amount)
            FROM taxi_trips
            WHERE fare_amount < 0
        """).fetchone()

        print("Count:", negative_fare_details[0])
        print("Minimum fare:", negative_fare_details[1])
        print("Maximum negative fare:", negative_fare_details[2])

    # ============================================================
    # 6. PASSENGER COUNT
    # ============================================================

    passenger_stats = conn.execute("""
        SELECT
            MIN(passenger_count),
            MAX(passenger_count),
            AVG(passenger_count)
        FROM taxi_trips
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
        FROM taxi_trips
        WHERE passenger_count = 0
    """).fetchone()[0]

    print("\n=== 7. ZERO PASSENGER COUNT ===")
    print("Zero passenger trips:", zero_passengers)

    # ============================================================
    # 8. NEGATIVE PASSENGER COUNT
    # ============================================================

    negative_passengers = conn.execute("""
        SELECT COUNT(*)
        FROM taxi_trips
        WHERE passenger_count < 0
    """).fetchone()[0]

    print("\n=== 8. NEGATIVE PASSENGER COUNT ===")
    print("Negative passenger trips:", negative_passengers)

    # ============================================================
    # 9. ZERO DISTANCE
    # ============================================================

    zero_distance = conn.execute("""
        SELECT COUNT(*)
        FROM taxi_trips
        WHERE trip_distance = 0
    """).fetchone()[0]

    print("\n=== 9. ZERO DISTANCE ===")
    print("Zero-distance trips:", zero_distance)

    # ============================================================
    # 10. ZERO FARE
    # ============================================================

    zero_fare = conn.execute("""
        SELECT COUNT(*)
        FROM taxi_trips
        WHERE fare_amount = 0
    """).fetchone()[0]

    print("\n=== 10. ZERO FARE ===")
    print("Zero-fare trips:", zero_fare)

    # ============================================================
    # 11. TRIP DURATION
    # ============================================================

    duration_stats = conn.execute("""
        SELECT
            MIN(dropoff_datetime - pickup_datetime),
            MAX(dropoff_datetime - pickup_datetime)
        FROM taxi_trips
        WHERE pickup_datetime IS NOT NULL
          AND dropoff_datetime IS NOT NULL
    """).fetchone()

    print("\n=== 11. TRIP DURATION ===")
    print("Shortest duration:", duration_stats[0])
    print("Longest duration:", duration_stats[1])

    # ============================================================
    # 12. DROPOFF BEFORE PICKUP
    # ============================================================

    invalid_time_count = conn.execute("""
        SELECT COUNT(*)
        FROM taxi_trips
        WHERE dropoff_datetime < pickup_datetime
    """).fetchone()[0]

    print("\n=== 12. INVALID TIME VALUES ===")
    print("Dropoff before pickup:", invalid_time_count)

    # ============================================================
    # 13. SAME PICKUP AND DROPOFF TIME
    # ============================================================

    same_time_count = conn.execute("""
        SELECT COUNT(*)
        FROM taxi_trips
        WHERE dropoff_datetime = pickup_datetime
    """).fetchone()[0]

    print("\n=== 13. SAME PICKUP/DROPOFF TIME ===")
    print("Trips with zero duration:", same_time_count)

    # ============================================================
    # 14. VERY LONG TRIPS
    # ============================================================

    long_trips = conn.execute("""
        SELECT COUNT(*)
        FROM taxi_trips
        WHERE dropoff_datetime > pickup_datetime
          AND dropoff_datetime - pickup_datetime > INTERVAL '3 hours'
    """).fetchone()[0]

    print("\n=== 14. VERY LONG TRIPS ===")
    print("Trips longer than 3 hours:", long_trips)

    # ============================================================
    # 15. VERY SHORT TRIPS
    # ============================================================

    short_trips = conn.execute("""
        SELECT COUNT(*)
        FROM taxi_trips
        WHERE dropoff_datetime > pickup_datetime
          AND dropoff_datetime - pickup_datetime < INTERVAL '1 minute'
    """).fetchone()[0]

    print("\n=== 15. VERY SHORT TRIPS ===")
    print("Trips shorter than 1 minute:", short_trips)

    # ============================================================
    # 16. NEGATIVE FARE SAMPLE
    # ============================================================

    print("\n=== 16. SAMPLE NEGATIVE FARE RECORDS ===")

    negative_fare_samples = conn.execute("""
        SELECT
            vendor_id,
            pickup_datetime,
            dropoff_datetime,
            passenger_count,
            trip_distance,
            fare_amount,
            tip_amount,
            total_amount,
            payment_type
        FROM taxi_trips
        WHERE fare_amount < 0
        LIMIT 10
    """).fetchall()

    if negative_fare_samples:
        for row in negative_fare_samples:
            print(row)
    else:
        print("No negative fare records found.")

    # ============================================================
    # 17. ZERO PASSENGER SAMPLE
    # ============================================================

    print("\n=== 17. SAMPLE ZERO PASSENGER RECORDS ===")

    zero_passenger_samples = conn.execute("""
        SELECT
            vendor_id,
            pickup_datetime,
            dropoff_datetime,
            passenger_count,
            trip_distance,
            fare_amount,
            total_amount,
            payment_type
        FROM taxi_trips
        WHERE passenger_count = 0
        LIMIT 10
    """).fetchall()

    if zero_passenger_samples:
        for row in zero_passenger_samples:
            print(row)
    else:
        print("No zero passenger records found.")

    # ============================================================
    # 18. ZERO DISTANCE SAMPLE
    # ============================================================

    print("\n=== 18. SAMPLE ZERO DISTANCE RECORDS ===")

    zero_distance_samples = conn.execute("""
        SELECT
            vendor_id,
            pickup_datetime,
            dropoff_datetime,
            passenger_count,
            trip_distance,
            fare_amount,
            total_amount,
            payment_type
        FROM taxi_trips
        WHERE trip_distance = 0
        LIMIT 10
    """).fetchall()

    if zero_distance_samples:
        for row in zero_distance_samples:
            print(row)
    else:
        print("No zero-distance records found.")

    # ============================================================
    # 19. INVALID TIME SAMPLE
    # ============================================================

    print("\n=== 19. INVALID TIME RECORDS ===")

    invalid_time_samples = conn.execute("""
        SELECT
            vendor_id,
            pickup_datetime,
            dropoff_datetime,
            passenger_count,
            trip_distance,
            fare_amount,
            total_amount
        FROM taxi_trips
        WHERE dropoff_datetime < pickup_datetime
        LIMIT 10
    """).fetchall()

    if invalid_time_samples:
        for row in invalid_time_samples:
            print(row)
    else:
        print("No invalid time records found.")

    # ============================================================
    # 20. VALIDATION SUMMARY
    # ============================================================

    print("\n" + "=" * 60)
    print("                 VALIDATION SUMMARY")
    print("=" * 60)

    print("Total rows:", total_rows)
    print("Negative distances:", negative_distance)
    print("Negative fares:", negative_fare)
    print("Zero passengers:", zero_passengers)
    print("Zero distance:", zero_distance)
    print("Zero fare:", zero_fare)
    print("Dropoff before pickup:", invalid_time_count)

    print("\nValidation completed successfully.")

    conn.close()


if __name__ == "__main__":
    validate_data()