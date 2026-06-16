"""
Step 00 - HYCOM Daily Resampling

Resamples raw HYCOM 3-hourly netCDF files to daily means, one output file
per year for temperature and salinity.

Inputs:
  HYCOM_SRC/hycom_*.nc          (3-hourly, multiple per year)

Outputs:
  FEATURES/hycom_water_temp_daily_{year}.nc
  FEATURES/hycom_salinity_daily_{year}.nc

Skip logic: each yearly output file is skipped if it already exists.
"""
from collections import defaultdict
from pathlib import Path

import xarray as xr
from tqdm import tqdm

from config import HYCOM_SRC, FEATURES


def main():
    all_files = sorted(HYCOM_SRC.rglob("hycom_*.nc"))
    print(f"Found {len(all_files)} HYCOM source files")

    files_by_year = defaultdict(list)
    for f in all_files:
        year = f.stem.split("_")[1][:4]
        files_by_year[year].append(f)

    for year, files in tqdm(sorted(files_by_year.items()), desc="Resampling years"):
        temp_out = FEATURES / f"hycom_water_temp_daily_{year}.nc"
        salt_out = FEATURES / f"hycom_salinity_daily_{year}.nc"

        if temp_out.exists() and salt_out.exists():
            print(f"{year}: already exists, skipping")
            continue

        try:
            datasets = []
            for f in sorted(files):
                ds = xr.open_dataset(f)
                if "water_temp" not in ds or "salinity" not in ds:
                    ds.close()
                    continue
                datasets.append(ds)

            if not datasets:
                print(f"{year}: no valid files found")
                continue

            ds_year = xr.concat(datasets, dim="time").sortby("time")
            ds_daily = ds_year.resample(time="1D").mean(dim="time", skipna=True)

            ds_daily[["water_temp"]].to_netcdf(temp_out)
            ds_daily[["salinity"]].to_netcdf(salt_out)

            for ds in datasets:
                ds.close()

            print(f"{year}: {ds_daily.sizes['time']} days saved")

        except Exception as e:
            print(f"{year}: ERROR - {e}")
            continue


if __name__ == "__main__":
    main()
