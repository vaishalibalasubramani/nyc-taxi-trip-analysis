from pathlib import Path
from urllib.request import urlretrieve

from config import RAW_DATA_DIR, TLC_BASE_URL


def download_yellow_taxi_data(year: int, month: int):
    filename = f"yellow_tripdata_{year}-{month:02d}.parquet"

    url = f"{TLC_BASE_URL}/{filename}"
    output_path = RAW_DATA_DIR / filename

    print(f"Downloading:")
    print(url)
    print()
    print(f"Saving to:")
    print(output_path)

    urlretrieve(url, output_path)

    print()
    print("Download completed successfully!")
    print(f"File: {output_path}")


if __name__ == "__main__":
    download_yellow_taxi_data(2025, 1)