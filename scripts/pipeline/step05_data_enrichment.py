"""
Step 05 - Data Enrichment - Calas x Oceanographic Matching

Matches each daily fishing event to the nearest HYCOM and MODIS grid point
in space and time, attaching temperature, salinity, chlorophyll, and SST vars.

Inputs:
  OUTPUTS/calas_daily_with_coordinates.csv       (from step 03)
  FEATURES/hycom_water_temp_daily_{year}.nc       (from step 00)
  FEATURES/hycom_salinity_daily_{year}.nc         (from step 00)
  FEATURES/hycom_water_temp_anomaly_daily_{year}.nc (from step 01)
  FEATURES/chl_daily_{year}.nc                   (from step 04)
  FEATURES/chl_anomaly_daily_{year}.nc           (from step 04)
  FEATURES/sst_daily_{year}.nc                   (from step 04b)
  FEATURES/sst_anomaly_daily_{year}.nc           (from step 04b)

Outputs:
  OUTPUTS/calas_enriched.csv

Skip logic: skipped entirely if output already exists.
"""
import numpy as np
import pandas as pd
import xarray as xr

from config import FEATURES, OUTPUTS


def match_nearest(df, ds, var):
    """Vectorized nearest-neighbor match of df lat/lon/date against an xarray dataset."""
    dates = xr.DataArray(df["date"].values, dims="points")
    lats  = xr.DataArray(df["lat"].values,  dims="points")
    lons  = xr.DataArray(df["lon"].values,  dims="points")
    vals  = ds[var].sel(time=dates, lat=lats, lon=lons, method="nearest").values
    return vals.astype(float)


def main():
    out = OUTPUTS / "calas_enriched.csv"
    if out.exists():
        print(f"Output already exists: {out} -- skipping")
        return

    df_calas = pd.read_csv(OUTPUTS / "calas_daily_with_coordinates.csv")
    df_calas["date"] = pd.to_datetime(df_calas["date"])
    df_calas["year"] = df_calas["date"].dt.year

    years = sorted(df_calas["year"].unique())
    all_results = []

    for year in years:
        df_year = df_calas[df_calas["year"] == year].copy()

        # HYCOM temperature and salinity
        temp_f = FEATURES / f"hycom_water_temp_daily_{year}.nc"
        salt_f = FEATURES / f"hycom_salinity_daily_{year}.nc"
        if not temp_f.exists() or not salt_f.exists():
            print(f"{year}: no HYCOM files, skipping")
            continue

        print(f"{year}: HYCOM temp/sal...", end=" ", flush=True)
        ds_t = xr.open_dataset(temp_f)
        ds_s = xr.open_dataset(salt_f)
        df_year["hycom_temp"] = match_nearest(df_year, ds_t, "water_temp")
        df_year["hycom_sal"]  = match_nearest(df_year, ds_s, "salinity")
        ds_t.close(); ds_s.close()
        print("done")

        # HYCOM temperature anomaly
        anom_f = FEATURES / f"hycom_water_temp_anomaly_daily_{year}.nc"
        if anom_f.exists():
            print(f"{year}: HYCOM temp anom...", end=" ", flush=True)
            ds = xr.open_dataset(anom_f)
            df_year["hycom_temp_anom"] = match_nearest(df_year, ds, "water_temp_anomaly")
            ds.close()
            print("done")
        else:
            df_year["hycom_temp_anom"] = np.nan

        # Chlorophyll-a
        chl_f = FEATURES / f"chl_daily_{year}.nc"
        if chl_f.exists():
            print(f"{year}: chl...", end=" ", flush=True)
            ds = xr.open_dataset(chl_f)
            df_year["chl"] = match_nearest(df_year, ds, "chlor_a")
            ds.close()
            print("done")
        else:
            df_year["chl"] = np.nan

        # CHL anomaly
        chl_anom_f = FEATURES / f"chl_anomaly_daily_{year}.nc"
        if chl_anom_f.exists():
            print(f"{year}: chl anom...", end=" ", flush=True)
            ds = xr.open_dataset(chl_anom_f)
            df_year["chl_anom"] = match_nearest(df_year, ds, "chl_anomaly")
            ds.close()
            print("done")
        else:
            df_year["chl_anom"] = np.nan

        # MODIS SST
        sst_f = FEATURES / f"sst_daily_{year}.nc"
        if sst_f.exists():
            print(f"{year}: modis sst...", end=" ", flush=True)
            ds = xr.open_dataset(sst_f)
            df_year["modis_sst"] = match_nearest(df_year, ds, "sst")
            ds.close()
            print("done")
        else:
            df_year["modis_sst"] = np.nan

        # MODIS SST anomaly
        sst_anom_f = FEATURES / f"sst_anomaly_daily_{year}.nc"
        if sst_anom_f.exists():
            print(f"{year}: modis sst anom...", end=" ", flush=True)
            ds = xr.open_dataset(sst_anom_f)
            df_year["modis_sst_anom"] = match_nearest(df_year, ds, "sst_anomaly")
            ds.close()
            print("done")
        else:
            df_year["modis_sst_anom"] = np.nan

        df_year["hycom_sal_anom_ref35"] = df_year["hycom_sal"] - 35.1

        all_results.append(df_year)
        print(f"{year}: COMPLETE ({len(df_year):,} rows)\n")

    df_final = pd.concat(all_results, ignore_index=True)
    print(f"Total: {len(df_final):,} rows")
    df_final.to_csv(out, index=False)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
