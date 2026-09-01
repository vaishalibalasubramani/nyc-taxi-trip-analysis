import duckdb

from config import DB_PATH


def get_connection():
    """Create and return a DuckDB connection."""
    return duckdb.connect(str(DB_PATH))


if __name__ == "__main__":
    conn = get_connection()

    # 1. Check available tables
    print("\n--- TABLES ---")
    tables = conn.execute("SHOW TABLES").fetchall()
    print(tables)

    # 2. Check total number of taxi records
    print("\n--- ROW COUNT ---")
    row_count = conn.execute(
        "SELECT COUNT(*) FROM taxi_trips"
    ).fetchone()

    print("Total rows:", row_count[0])

    # 3. Check table structure
    print("\n--- TABLE STRUCTURE ---")
    columns = conn.execute(
        "DESCRIBE taxi_trips"
    ).fetchall()

    for column in columns:
        print(column)

    # 4. Show 5 sample records
    print("\n--- SAMPLE DATA ---")
    sample_data = conn.execute("""
        SELECT *
        FROM taxi_trips
        LIMIT 5
    """).fetchall()

    for row in sample_data:
        print(row)

    conn.close()