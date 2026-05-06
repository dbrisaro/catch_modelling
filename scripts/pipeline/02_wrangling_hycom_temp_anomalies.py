"""
Step 01 - HYCOM Water Temperature Anomalies

Computes a day-of-year climatology across all years and saves per-year
daily anomaly files (water_temp - climatology[doy]).

Inputs:
  FEATURES/hycom_water_temp_daily_{year}.nc   (from step 00)

Outputs:
  FEATURES/hycom_water_temp_climatology.nc
  FEATURES/hycom_water_temp_anomaly_daily_{year}.nc

Skip logic: individual anomaly files are skipped if they already exist.
  Climatology is always recomputed from all available yearly files.
"""
import xarray as xr
from pathlib import Path

from config import FEATURES, YEARS


def main():
    available = [y for y in YEARS if (FEATURES / f"hycom_water_temp_daily_{y}.nc").exists()]
    print(f"Loading temperature files for years: {available}")

    datasets = []
    for year in available:
        ds = xr.open_dataset(FEATURES / f"hycom_water_temp_daily_{year}.nc")
        datasets.append(ds)
        print(f"  {year}: {ds.sizes['time']} days")

    ds_all = xr.concat(datasets, dim="time").sortby("time")
    ds_all = ds_all.squeeze("depth", drop=True)
    print(f"\nTotal: {ds_all.sizes['time']} days")

    clim_path = FEATURES / "hycom_water_temp_climatology.nc"
    climatology = ds_all["water_temp"].groupby("time.dayofyear").mean("time")
    climatology.to_dataset(name="water_temp_climatology").to_netcdf(clim_path)
    print(f"Saved climatology -> {clim_path.name}")

    anomalies = ds_all["water_temp"].groupby("time.dayofyear") - climatology
    anomalies.name = "water_temp_anomaly"

    for year in available:
        out = FEATURES / f"hycom_water_temp_anomaly_daily_{year}.nc"
        if out.exists():
            print(f"{year}: anomaly already exists, skipping")
            continue
        anom_year = anomalies.sel(time=anomalies.time.dt.year == year)
        anom_year.to_dataset(name="water_temp_anomaly").to_netcdf(out)
        print(f"{year}: anomaly saved -> {out.name}  ({anom_year.sizes['time']} days)")


if __name__ == "__main__":
    main()
