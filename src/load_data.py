from database import get_connection


# Direct NYC TLC CloudFront URL
TLC_DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"


def load_taxi_data(year, month):
    """
    Load one month of NYC Yellow Taxi data directly from HTTPS
    into DuckDB without downloading the Parquet file.
    """

    file_name = f"yellow_tripdata_{year}-{month:02d}.parquet"
    file_url = f"{TLC_DATA_URL}/{file_name}"

    print("\n" + "=" * 60)
    print("LOADING TAXI DATA")
    print("=" * 60)

    print("Year:", year)
    print("Month:", month)
    print("URL:", file_url)

    conn = get_connection()

    try:
        # Insert only the columns that exist in our table.
        conn.execute(f"""
            INSERT INTO taxi_trips (
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
            )
            SELECT
                VendorID,
                tpep_pickup_datetime,
                tpep_dropoff_datetime,
                passenger_count,
                trip_distance,
                RatecodeID,
                store_and_fwd_flag,
                PULocationID,
                DOLocationID,
                payment_type,
                fare_amount,
                extra,
                mta_tax,
                tip_amount,
                tolls_amount,
                improvement_surcharge,
                total_amount,
                congestion_surcharge,
                Airport_fee,
                cbd_congestion_fee
            FROM read_parquet('{file_url}')
        """)

        count = conn.execute("""
            SELECT COUNT(*)
            FROM taxi_trips
            WHERE
                EXTRACT(YEAR FROM pickup_datetime) = ?
                AND EXTRACT(MONTH FROM pickup_datetime) = ?
        """, [year, month]).fetchone()[0]

        print("Successfully loaded.")
        print("Rows for this month:", f"{count:,}")

    except Exception as e:
        print("\nERROR while loading data:")
        print(e)

        raise

    finally:
        conn.close()


def load_all_2025_data():
    """
    Load all 12 months of 2025 directly from HTTPS.
    """

    print("\n")
    print("=" * 60)
    print("LOADING ALL 2025 YELLOW TAXI DATA")
    print("=" * 60)

    for month in range(1, 13):
        load_taxi_data(2025, month)

    print("\n")
    print("=" * 60)
    print("ALL 2025 DATA LOADED")
    print("=" * 60)


if __name__ == "__main__":
    load_all_2025_data()