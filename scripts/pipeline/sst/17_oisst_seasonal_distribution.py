"""
17_oisst_seasonal_distribution.py

Loads NOAA OISST v2.1 monthly SST for the Peru coastal shelf, recomputes
anomalies with the 2005-2024 climatology (matching CLIM_YEARS in the pipeline),
and plots the distribution of T1 and T2 seasonal means per region.

Seasons:
  T1 : Apr-Jul (months 4-7)
  T2 : Nov-Dec (months 11-12)

Regions : Norte (lat > -7.1), Centro Norte (-11 to -7.1), Centro Sur (-15.8 to -11)
          lon -82 to -74

Climatology : 2005-2024 monthly means

Outputs:
  FEATURES/oisst_seasonal_means_clim2005_2024.csv
  outputs/sst/17_oisst_seasonal_distribution.png
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FEATURES, PLOTS

OUT_DIR = PLOTS / "sst"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = FEATURES / "oisst_seasonal_means_clim2005_2024.csv"
OUT_FIG = OUT_DIR  / "17_oisst_seasonal_distribution.png"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CLIM_START, CLIM_END = 2005, 2024
T1_MONTHS = [4, 5, 6, 7]   # Apr-Jul
T2_MONTHS = [11, 12]        # Nov-Dec

LAT_S, LAT_N = -16.0, -4.0
LON_W, LON_E = -83.0, -73.0

REGIONS = [
    dict(name="Norte",        lat_min=-7.1,  lat_max=None),
    dict(name="Centro Norte", lat_min=-11.0, lat_max=-7.1),
    dict(name="Centro Sur",   lat_min=-15.8, lat_max=-11.0),
]

OISST_URL = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.mon.mean.nc"

COL_T1 = "#2166ac"
COL_T2 = "#e07b39"

EVENT_YEARS_T1 = {2015: "2015-16", 2023: "2023-24"}
EVENT_YEARS_T2 = {2015: "2015-16", 2023: "2023-24"}

# ---------------------------------------------------------------------------
# Load and process
# ---------------------------------------------------------------------------

def load_raw_oisst():
    """
    Open OISST from PSL THREDDS (lazy), subset to Peru shelf, download.
    Returns xr.DataArray with dim (time, lat, lon), lon in -180..180.
    """
    print("Opening OISST via OPeNDAP (lazy)...")
    ds = xr.open_dataset(OISST_URL, engine="netcdf4")

    lon_w360 = LON_W + 360
    lon_e360 = LON_E + 360

    sst = ds["sst"].sel(
        lat=slice(LAT_S, LAT_N),
        lon=slice(lon_w360, lon_e360),
    )
    print("  Downloading subset...")
    sst = sst.load()
    sst = sst.assign_coords(lon=(sst.lon.values - 360))
    sst = sst.where(sst < 1e6)
    print(f"  Shape: {sst.shape}  "
          f"time: {str(sst.time.values[0])[:7]} to {str(sst.time.values[-1])[:7]}")
    ds.close()
    return sst


def region_spatial_mean(sst, lat_min, lat_max):
    """Spatial mean over lat/lon band. lat ascending in the data."""
    lo = lat_min
    hi = lat_max if lat_max is not None else float(sst.lat.max())
    sub = sst.sel(lat=slice(lo, hi))
    return sub.mean(dim=["lat", "lon"], skipna=True)


def anomaly_clim2005_2024(sst_1d):
    """Anomaly relative to 2005-2024 monthly climatology."""
    clim_mask = sst_1d.time.dt.year.isin(range(CLIM_START, CLIM_END + 1))
    clim = sst_1d.sel(time=clim_mask).groupby("time.month").mean("time")
    return sst_1d.groupby("time.month") - clim


def seasonal_means_from_monthly(anom_1d):
    """
    Compute annual T1 and T2 seasonal means from a monthly anomaly series.
    T2 uses the calendar year (Nov-Dec of year Y → T2 Y).
    """
    df = anom_1d.to_series().reset_index()
    df.columns = ["time", "anom"]
    df["year"]  = df["time"].dt.year
    df["month"] = df["time"].dt.month

    t1 = (df[df["month"].isin(T1_MONTHS)]
            .groupby("year")["anom"].mean())
    t2 = (df[df["month"].isin(T2_MONTHS)]
            .groupby("year")["anom"].mean())

    years = sorted(t1.index.intersection(t2.index))
    return (np.array(years),
            t1.loc[years].values,
            t2.loc[years].values)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_distributions(all_data, out_path):
    """
    3-panel figure, one per region.
    Each panel overlays the T1 (blue) and T2 (orange) distributions
    as histograms + KDE curves. Rug plot marks individual years;
    key El Nino events are labelled.
    Vertical lines show p90 and p95 per season.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    for ax, row in zip(axes, all_data):
        region = row["name"]
        years  = row["years"]
        t1     = row["t1"]
        t2     = row["t2"]

        xmin = min(t1.min(), t2.min()) - 0.3
        xmax = max(t1.max(), t2.max()) + 0.3
        xgrid = np.linspace(xmin, xmax, 300)

        trans = ax.get_xaxis_transform()  # x=data, y=axes [0,1]

        for vals, color, label in [
            (t1, COL_T1, "T1 (Apr-Jul)"),
            (t2, COL_T2, "T2 (Nov-Dec)"),
        ]:
            # histogram
            ax.hist(vals, bins=10, density=True, color=color,
                    alpha=0.22, edgecolor="none")

            # KDE
            kde = gaussian_kde(vals, bw_method="scott")
            ax.plot(xgrid, kde(xgrid), color=color, lw=2.0, label=label)

            # p90 and p95 vertical lines — labelled at top, rotated
            for pct, ls, lw, ya in [(90, "--", 1.2, 0.94), (95, ":", 1.4, 0.78)]:
                v = float(np.percentile(vals, pct))
                ax.axvline(v, color=color, ls=ls, lw=lw, alpha=0.75)
                ax.text(v - 0.06, ya, f"p{pct}", fontsize=6.5, color=color,
                        rotation=90, ha="right", va="top", transform=trans)

            # rug at bottom — plain tick marks only
            for val in vals:
                ax.plot(val, -0.03, "|", color=color, alpha=0.55, ms=5,
                        transform=trans, clip_on=False)

        # annotate key El Niño events once (using T1 values to avoid duplication)
        year_list = list(years)
        for yr, lbl in EVENT_YEARS_T1.items():
            if yr in year_list:
                xval = t1[year_list.index(yr)]
                ax.text(xval, 0.10, lbl, fontsize=6.5, color=COL_T1,
                        ha="center", va="bottom", transform=trans,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="none", alpha=0.7))

        ax.set_title(region, fontsize=10)
        ax.set_xlabel("SST anomaly (°C)", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("density", fontsize=9)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axvline(0, color="#999999", lw=0.8, ls="-", alpha=0.5)

    n_years = len(all_data[0]["years"])
    y_range = f"{all_data[0]['years'][0]}-{all_data[0]['years'][-1]}"
    fig.suptitle(
        f"OISST seasonal anomaly distribution  |  clim {CLIM_START}-{CLIM_END}  |  "
        f"N={n_years} years ({y_range})  |  "
        f"dashed=p90, dotted=p95",
        fontsize=8.5, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sst = load_raw_oisst()

    all_data = []
    csv_rows = []

    for reg in REGIONS:
        name = reg["name"]
        print(f"\n{name}: computing spatial mean and anomaly...")
        sst_1d = region_spatial_mean(sst, reg["lat_min"], reg["lat_max"])
        anom   = anomaly_clim2005_2024(sst_1d)
        years, t1, t2 = seasonal_means_from_monthly(anom)

        print(f"  N years: {len(years)}  ({years[0]}-{years[-1]})")
        print(f"  T1 — mean={t1.mean():.3f}  p90={np.percentile(t1,90):.3f}  "
              f"p95={np.percentile(t1,95):.3f}  p99={np.percentile(t1,99):.3f}")
        print(f"  T2 — mean={t2.mean():.3f}  p90={np.percentile(t2,90):.3f}  "
              f"p95={np.percentile(t2,95):.3f}  p99={np.percentile(t2,99):.3f}")

        all_data.append(dict(name=name, years=years, t1=t1, t2=t2))

        for yr, v1, v2 in zip(years, t1, t2):
            csv_rows.append(dict(year=yr, region=name, t1=v1, t2=v2))

    pd.DataFrame(csv_rows).to_csv(OUT_CSV, index=False)
    print(f"\nSaved -> {OUT_CSV}")

    plot_distributions(all_data, OUT_FIG)


if __name__ == "__main__":
    main()
