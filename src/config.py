from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Main directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATABASE_DIR = PROJECT_ROOT / "database"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# DuckDB database
DB_PATH = DATABASE_DIR / "nyc_taxi.duckdb"

# NYC TLC Yellow Taxi data
TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"

print("Project root:", PROJECT_ROOT)
print("Database path:", DB_PATH)