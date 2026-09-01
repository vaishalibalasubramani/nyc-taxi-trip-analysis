from database import get_connection


def initialize_database():
    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS taxi_trips (
            vendor_id INTEGER,
            pickup_datetime TIMESTAMP,
            dropoff_datetime TIMESTAMP,
            passenger_count DOUBLE,
            trip_distance DOUBLE,
            rate_code_id INTEGER,
            store_and_fwd_flag VARCHAR,
            pickup_location_id INTEGER,
            dropoff_location_id INTEGER,
            payment_type INTEGER,
            fare_amount DOUBLE,
            extra DOUBLE,
            mta_tax DOUBLE,
            tip_amount DOUBLE,
            tolls_amount DOUBLE,
            improvement_surcharge DOUBLE,
            total_amount DOUBLE,
            congestion_surcharge DOUBLE,
            airport_fee DOUBLE,
            cbd_congestion_fee DOUBLE
        )
    """)

    print("taxi_trips table created successfully!")

    conn.close()


if __name__ == "__main__":
    initialize_database()