"""
Download OISSS v2 SSS files for 2025 from PO.DAAC (NASA Earthdata).

Usage:
  python download_oisss_2025.py

Authentication:
  Reads credentials from ~/.netrc (recommended):
    machine urs.earthdata.nasa.gov login <user> password <pass>
  Or set env vars EARTHDATA_USER and EARTHDATA_PASS.

Output directory: /home/jupyter-daniela/suyana/sources/OISSS/2025/
"""
import os
import netrc
import getpass
import requests
from pathlib import Path
from tqdm import tqdm

COLLECTION_ID = "C2589160971-POCLOUD"
CMR_URL       = "https://cmr.earthdata.nasa.gov/search/granules.json"
AUTH_HOST     = "urs.earthdata.nasa.gov"
OUT_DIR       = Path("/home/jupyter-daniela/suyana/sources/OISSS/2025")

YEAR = 2025


def get_credentials():
    user = os.environ.get("EARTHDATA_USER")
    pwd  = os.environ.get("EARTHDATA_PASS")
    if user and pwd:
        return user, pwd
    try:
        n = netrc.netrc()
        auth = n.authenticators(AUTH_HOST)
        if auth:
            return auth[0], auth[2]
    except Exception:
        pass
    print("NASA Earthdata credentials not found in ~/.netrc or environment.")
    user = input("Earthdata username: ").strip()
    pwd  = getpass.getpass("Earthdata password: ")
    return user, pwd


def get_granule_urls(year):
    """Query CMR for all granules in the given year, return list of (filename, url)."""
    urls = []
    page = 1
    while True:
        r = requests.get(CMR_URL, params={
            "concept_id": COLLECTION_ID,
            "temporal[]": f"{year}-01-01T00:00:00Z,{year}-12-31T23:59:59Z",
            "page_size":  100,
            "page_num":   page,
            "sort_key":   "start_date",
        }, timeout=30)
        r.raise_for_status()
        entries = r.json().get("feed", {}).get("entry", [])
        if not entries:
            break
        for e in entries:
            for link in e.get("links", []):
                if "data#" in link.get("rel", "") and link["href"].endswith(".nc"):
                    urls.append((e["title"] + ".nc", link["href"]))
                    break
        page += 1
    return urls


def download_file(url, dest, session):
    if dest.exists():
        print(f"  already exists, skipping: {dest.name}")
        return
    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    user, pwd = get_credentials()

    print(f"Querying CMR for OISSS v2 granules in {YEAR}...")
    granules = get_granule_urls(YEAR)
    print(f"Found {len(granules)} granules")

    if not granules:
        print("No granules found. The data may not be available yet.")
        return

    session = requests.Session()
    session.auth = (user, pwd)

    for fname, url in granules:
        dest = OUT_DIR / fname
        print(f"Downloading {fname}...")
        try:
            download_file(url, dest, session)
        except Exception as e:
            print(f"  ERROR: {e}")

    downloaded = list(OUT_DIR.glob("*.nc"))
    print(f"\nDone. {len(downloaded)} files in {OUT_DIR}")


if __name__ == "__main__":
    main()
