"""
Step 04b - SST Data Preparation (MODIS AQUA)

Processes daily MODIS AQUA sea surface temperature files into per-year
netCDF files, computes a day-of-year climatology, and derives daily anomalies.

Inputs:
  SST_SRC/AQUA_MODIS.{YYYYMMDD}.*.nc

Outputs:
  FEATURES/sst_daily_{year}.nc
  FEATURES/sst_climatology_doy.nc
  FEATURES/sst_anomaly_daily_{year}.nc

Skip logic: daily and anomaly files are skipped per year if they already exist.
"""
import xarray as xr
import pandas as pd
from tqdm import tqdm

from config import SST_SRC, FEATURES, SST_YEARS

CLIM_YEARS = list(range(2005, 2025))   # canonical 20-year climatology baseline


def main():
    all_files = sorted(SST_SRC.glob("AQUA_MODIS.*.nc"))
    print(f"Total SST files: {len(all_files)}")

    files_by_year = {}
    for f in all_files:
        year = int(f.name.split(".")[1][:4])
        files_by_year.setdefault(year, []).append(f)

    print(f"Years found: {sorted(files_by_year.keys())}")

    # --- Step 1: per-year daily netCDF ---
    for year in tqdm(SST_YEARS, desc="Years"):
        out = FEATURES / f"sst_daily_{year}.nc"
        if out.exists():
            print(f"{year}: already exists, skipping")
            continue

        files = files_by_year.get(year, [])
        if not files:
            print(f"{year}: no source files found")
            continue

        daily_arrays = []
        for f in sorted(files):
            date_str = f.name.split(".")[1]
            date = pd.Timestamp(date_str)
            try:
                ds = xr.open_dataset(f)
                sst = ds["sst"].expand_dims(time=[date])
                daily_arrays.append(sst)
                ds.close()
            except Exception as e:
                print(f"  Skipping {f.name}: {e}")

        if not daily_arrays:
            print(f"{year}: no valid files")
            continue

        ds_year = xr.concat(daily_arrays, dim="time").to_dataset(name="sst")
        ds_year = ds_year.sortby("time")
        ds_year.to_netcdf(out)
        print(f"{year}: {ds_year.sizes['time']} days -> {out.name}")

    # --- Step 2: day-of-year climatology (2005-2024 only) ---
    clim_file = FEATURES / "sst_climatology_doy.nc"
    if not clim_file.exists():
        yearly = [xr.open_dataset(FEATURES / f"sst_daily_{y}.nc")
                  for y in CLIM_YEARS if (FEATURES / f"sst_daily_{y}.nc").exists()]
        print(f"  Building climatology from {len(yearly)} years: {CLIM_YEARS[0]}-{CLIM_YEARS[-1]}")
        ds_all = xr.concat(yearly, dim="time").sortby("time")
        ds_clim = ds_all.groupby("time.dayofyear").mean(dim="time").rename({"sst": "sst_clim"})
        ds_clim.to_netcdf(clim_file)
        print(f"\nClimatology saved -> {clim_file.name}")
    else:
        print(f"\nSST climatology already exists, skipping")

    # --- Step 3: anomalies per year ---
    ds_clim = xr.open_dataset(clim_file)

    for year in tqdm(SST_YEARS, desc="Anomalies"):
        out = FEATURES / f"sst_anomaly_daily_{year}.nc"
        if out.exists():
            print(f"{year}: anomaly exists, skipping")
            continue

        daily_file = FEATURES / f"sst_daily_{year}.nc"
        if not daily_file.exists():
            print(f"{year}: daily file missing, skipping")
            continue

        ds_year = xr.open_dataset(daily_file)
        doy = ds_year["time"].dt.dayofyear
        clim_aligned = ds_clim["sst_clim"].sel(dayofyear=doy).drop_vars("dayofyear")
        anom = (ds_year["sst"] - clim_aligned).to_dataset(name="sst_anomaly")
        anom.to_netcdf(out)
        ds_year.close()
        print(f"{year}: anomaly saved -> {out.name}")

    ds_clim.close()


if __name__ == "__main__":
    main()
