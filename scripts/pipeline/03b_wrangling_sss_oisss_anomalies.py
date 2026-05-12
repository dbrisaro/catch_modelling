"""
Step 03b - SSS Data Preparation (OISSS L4 Multimission)

Processes ~4-day OISSS sea surface salinity files into per-year netCDF
files cropped to the Peru fishing region, computes a day-of-year
climatology, and derives anomalies.

Source files are global (720x1440), dim order (latitude, longitude, time).
Output files use (time, lat, lon) with renamed coords.

Inputs:
  OISSS_SRC/{year}/OISSS_L4_multimission_global_7d_v1.0_{date}.nc

Outputs:
  FEATURES/sss_weekly_{year}.nc              (Peru-cropped raw SSS)
  FEATURES/sss_climatology_doy.nc            (DOY climatology)
  FEATURES/sss_anomaly_weekly_{year}.nc      (anomaly = raw - clim)

Peru bbox: lat [-16, -4], lon [-83, -74]
Climatology base: 2015-2024 (10-year reference period).
Skip logic: per-year files are skipped if they already exist.
"""
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import OISSS_SRC, FEATURES, SSS_YEARS

LAT_MIN, LAT_MAX = -16.0, -4.0
LON_MIN, LON_MAX = -83.0, -74.0

CLIM_YEARS = list(range(2015, 2025))   # 10-year reference period


def load_year(year: int) -> xr.Dataset | None:
    """Load all OISSS files for one year, crop to Peru, return (time, lat, lon)."""
    src_dir = OISSS_SRC / str(year)
    files = sorted(src_dir.glob("OISSS_L4_multimission_global_7d_v*.nc"))
    if not files:
        print(f"  {year}: no source files found")
        return None

    parts = []
    for f in files:
        date_str = f.stem.split("_")[-1]           # YYYY-MM-DD
        date = pd.Timestamp(date_str)
        try:
            ds = xr.open_dataset(f)
            # crop before loading into memory
            sss = (ds["sss"]
                   .sel(latitude=slice(LAT_MIN, LAT_MAX),
                        longitude=slice(LON_MIN, LON_MAX))
                   .squeeze("time", drop=True)
                   .expand_dims(time=[date.to_datetime64()]))
            parts.append(sss.load())
            ds.close()
        except Exception as e:
            print(f"  skipping {f.name}: {e}")

    if not parts:
        return None

    da = xr.concat(parts, dim="time").sortby("time")
    da = da.rename({"latitude": "lat", "longitude": "lon"})
    return da.to_dataset(name="sss")


def main():
    # -- Step 1: per-year cropped SSS --
    print("=== Step 1: per-year cropped SSS ===")
    for year in tqdm(SSS_YEARS, desc="Years"):
        out = FEATURES / f"sss_weekly_{year}.nc"
        if out.exists():
            print(f"  {year}: already exists, skipping")
            continue
        ds = load_year(year)
        if ds is None:
            continue
        ds.to_netcdf(out)
        print(f"  {year}: {ds.sizes['time']} steps -> {out.name}")

    # -- Step 2: DOY climatology over all available years --
    print("\n=== Step 2: DOY climatology ===")
    clim_file = FEATURES / "sss_climatology_doy.nc"
    if clim_file.exists():
        print(f"  Already exists, skipping: {clim_file.name}")
    else:
        yearly = []
        for year in CLIM_YEARS:
            f = FEATURES / f"sss_weekly_{year}.nc"
            if f.exists():
                yearly.append(xr.open_dataset(f))
        if not yearly:
            print("  No yearly files found - aborting climatology step")
            return
        print(f"  Building climatology from {len(yearly)} years "
              f"({CLIM_YEARS[0]}-{CLIM_YEARS[-1]})")
        ds_all = xr.concat(yearly, dim="time").sortby("time")
        ds_clim = (ds_all["sss"]
                   .groupby("time.dayofyear")
                   .mean(dim="time")
                   .to_dataset(name="sss_clim"))
        ds_clim.to_netcdf(clim_file)
        print(f"  Saved -> {clim_file.name}")
        for ds in yearly:
            ds.close()

    # -- Step 3: anomalies per year --
    print("\n=== Step 3: anomalies per year ===")
    ds_clim = xr.open_dataset(clim_file)

    for year in tqdm(SSS_YEARS, desc="Anomalies"):
        out = FEATURES / f"sss_anomaly_weekly_{year}.nc"
        if out.exists():
            print(f"  {year}: anomaly exists, skipping")
            continue
        f = FEATURES / f"sss_weekly_{year}.nc"
        if not f.exists():
            print(f"  {year}: raw file missing, skipping")
            continue
        ds_year = xr.open_dataset(f)
        doy = ds_year["time"].dt.dayofyear
        clim_aligned = ds_clim["sss_clim"].sel(dayofyear=doy).drop_vars("dayofyear")
        anom = (ds_year["sss"] - clim_aligned).to_dataset(name="sss_anomaly")
        anom.to_netcdf(out)
        ds_year.close()
        print(f"  {year}: anomaly saved -> {out.name}")

    ds_clim.close()


if __name__ == "__main__":
    main()
