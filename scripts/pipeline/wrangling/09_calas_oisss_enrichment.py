"""
Step 07b - Enriquecimiento calas x OISSS SSS

Agrega la anomalia de salinidad superficial OISSS a cada cala en
calas_enriched.csv, usando el mismo match nearest-neighbor que el paso 07.

Inputs:
  OUTPUTS/calas_enriched.csv             (de step 07)
  FEATURES/sss_anomaly_weekly_{year}.nc  (de step 03b)

Outputs:
  OUTPUTS/calas_enriched.csv  (misma tabla + columna oisss_sss_anom)

Skip logic: skipped si la columna oisss_sss_anom ya existe en calas_enriched.csv
"""
import numpy as np
import pandas as pd
import xarray as xr

from config import FEATURES, OUTPUTS


def match_nearest(df, ds, var):
    """Identico al step 07: nearest-neighbor en tiempo, lat y lon."""
    da   = ds[var].squeeze()
    data = da.values                          # (T, lat, lon)

    t_arr   = ds["time"].values.astype("datetime64[ns]").astype(np.int64)
    lat_arr = da["lat"].values.astype(np.float64)
    lon_arr = da["lon"].values.astype(np.float64)

    d_pts  = df["date"].values.astype("datetime64[ns]").astype(np.int64)
    la_pts = df["lat"].values.astype(np.float64)
    lo_pts = df["lon"].values.astype(np.float64)

    t_idx   = np.abs(t_arr[:, None]   - d_pts[None, :]).argmin(axis=0)
    lat_idx = np.abs(lat_arr[:, None] - la_pts[None, :]).argmin(axis=0)
    lon_idx = np.abs(lon_arr[:, None] - lo_pts[None, :]).argmin(axis=0)

    return data[t_idx, lat_idx, lon_idx].astype(float)


def main():
    out = OUTPUTS / "calas_enriched.csv"

    df = pd.read_csv(out, low_memory=False)

    if "oisss_sss_anom" in df.columns:
        print("oisss_sss_anom ya existe en calas_enriched.csv -- skipping")
        return

    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["oisss_sss_anom"] = np.nan

    for year in sorted(df["year"].unique()):
        f = FEATURES / f"sss_anomaly_weekly_{year}.nc"
        if not f.exists():
            print(f"{year}: no OISSS file, skipping")
            continue

        idx = df["year"] == year
        print(f"{year}: OISSS SSS anom...", end=" ", flush=True)
        ds = xr.open_dataset(f)
        df.loc[idx, "oisss_sss_anom"] = match_nearest(df[idx], ds, "sss_anomaly")
        ds.close()
        print(f"done  ({idx.sum():,} rows)")

    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")
    n_valid = df["oisss_sss_anom"].notna().sum()
    print(f"oisss_sss_anom: {n_valid:,} / {len(df):,} filas con dato")


if __name__ == "__main__":
    main()
