"""Download the Northwind SQLite DB into ./data. Run: `make data`.

Northwind = orders / customers / products / employees (e-commerce flavoured).
Fallback dataset (music store), if you prefer: Chinook —
  https://github.com/lerocha/chinook-database/raw/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite
"""
import os
import urllib.request
from src.config import settings

URL = "https://github.com/jpwhite3/northwind-SQLite3/raw/main/dist/northwind.db"


def main():
    os.makedirs(os.path.dirname(settings.db_path), exist_ok=True)
    if os.path.exists(settings.db_path):
        print(f"Already present: {settings.db_path}")
        return
    print(f"Downloading Northwind -> {settings.db_path} ...")
    urllib.request.urlretrieve(URL, settings.db_path)
    print("Done.")


if __name__ == "__main__":
    main()
