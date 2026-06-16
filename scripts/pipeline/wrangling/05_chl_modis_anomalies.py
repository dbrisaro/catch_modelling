"""
Step 04 - Chlorophyll-a Data Preparation (MODIS AQUA)

Processes daily MODIS AQUA chlorophyll-a files into per-year netCDF files,
computes a day-of-year climatology, and derives daily anomalies.

Inputs:
  CHL_SRC/AQUA_MODIS.{YYYYMMDD}.*.nc

Outputs:
  FEATURES/chl_daily_{year}.nc
  FEATURES/chl_climatology_doy.nc
  FEATURES/chl_anomaly_daily_{year}.nc

Skip logic: daily and anomaly files are skipped per year if they already exist.
"""
import xarray as xr
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from config import CHL_SRC, FEATURES


def main():
    all_files = sorted(CHL_SRC.glob("AQUA_MODIS.*.nc"))
    print(f"Total CHL files: {len(all_files)}")

    files_by_year = {}
    for f in all_files:
        year = f.name.split(".")[1][:4]
        files_by_year.setdefault(year, []).append(f)

    print(f"Years found: {sorted(files_by_year.keys())}")

    # --- Step 1: build per-year daily netCDF ---
    for year, files in tqdm(sorted(files_by_year.items()), desc="Years"):
        out = FEATURES / f"chl_daily_{year}.nc"
        if out.exists():
            print(f"{year}: daily file exists, skipping")
            continue

        daily_arrays = []
        for f in sorted(files):
            date_str = f.name.split(".")[1]
            date = pd.Timestamp(date_str)
            try:
                ds = xr.open_dataset(f)
                chl = ds["chlor_a"].expand_dims(time=[date])
                daily_arrays.append(chl)
                ds.close()
            except Exception as e:
                print(f"  Skipping {f.name}: {e}")
                continue

        if not daily_arrays:
            print(f"{year}: no valid files, skipping")
            continue

        ds_year = xr.concat(daily_arrays, dim="time").to_dataset(name="chlor_a")
        ds_year = ds_year.sortby("time")
        ds_year.to_netcdf(out)
        print(f"{year}: {ds_year.sizes['time']} days saved -> {out.name}")

    # --- Step 2: climatology ---
    clim_file = FEATURES / "chl_climatology_doy.nc"
    available = [y for y in files_by_year if (FEATURES / f"chl_daily_{y}.nc").exists()]

    if not clim_file.exists():
        print(f"\nComputing climatology from years: {sorted(available)}")
        datasets = [xr.open_dataset(FEATURES / f"chl_daily_{y}.nc") for y in sorted(available)]
        ds_all = xr.concat(datasets, dim="time").sortby("time")
        ds_clim = ds_all.groupby("time.dayofyear").mean(dim="time").rename({"chlor_a": "chl_clim"})
        ds_clim.to_netcdf(clim_file)
        print(f"Climatology saved -> {clim_file.name}")
    else:
        print(f"\nClimatology already exists, skipping")

    # --- Step 3: anomalies per year ---
    ds_clim = xr.open_dataset(clim_file)

    for year in tqdm(sorted(available), desc="Anomalies"):
        out = FEATURES / f"chl_anomaly_daily_{year}.nc"
        if out.exists():
            print(f"{year}: anomaly exists, skipping")
            continue
        ds_year = xr.open_dataset(FEATURES / f"chl_daily_{year}.nc")
        doy = ds_year["time"].dt.dayofyear
        clim_aligned = ds_clim["chl_clim"].sel(dayofyear=doy).drop_vars("dayofyear")
        anom = (ds_year["chlor_a"] - clim_aligned).to_dataset(name="chl_anomaly")
        anom.to_netcdf(out)
        ds_year.close()
        print(f"{year}: anomaly saved -> {out.name}")

    ds_clim.close()


if __name__ == "__main__":
    main()
