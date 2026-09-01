from database import get_connection


def check_months():
    conn = get_connection()

    print("\n" + "=" * 60)
    print("              MONTHLY DATA CHECK")
    print("=" * 60)

    result = conn.execute("""
        SELECT
            EXTRACT(YEAR FROM pickup_datetime) AS year,
            EXTRACT(MONTH FROM pickup_datetime) AS month,
            COUNT(*) AS trip_count
        FROM taxi_trips
        GROUP BY 1, 2
        ORDER BY 1, 2
    """).fetchall()

    if not result:
        print("No data found.")
    else:
        for row in result:
            print(
                f"Year: {int(row[0])} | "
                f"Month: {int(row[1]):02d} | "
                f"Trips: {row[2]:,}"
            )

    conn.close()


if __name__ == "__main__":
    check_months()