"""
Step 12b - Spatial homogeneity of SSS signal along the Peruvian coast

Analogous to 12_analysis_sst_spatial_homogeneity.py but using OISSS
sea-surface salinity anomalies instead of MODIS SST.

Validates whether salinity anomalies (El Nino freshening signal) are
spatially coherent across the fishing corridor, justifying the use of a
single area-wide SSS anomaly index as a parametric insurance trigger.

Method:
  1. Load sss_anomaly_weekly_{year}.nc for all SSS_YEARS (2012-2024).
  2. Restrict to coastal fishing zone: lat -16 to -6, lon -82 to -74.
  3. Resample weekly data to monthly mean.
  4. Average over longitude to get a single monthly anomaly per 1-degree
     latitude band (10 bands from -16 to -6).
  5. Plot a stacked ridge plot (joy plot): monthly SSS anomaly time series
     per latitude band, filled orange/red (saltier) / blue (fresher).
  6. Compute pairwise Pearson correlations between band time series.
  7. If correlations are consistently high (> 0.7), document justification
     for area-wide SSS average as the trigger index.

Inputs:
  FEATURES/sss_anomaly_weekly_{year}.nc   (variable: sss_anomaly, coords: lat/lon)

Outputs:
  PLOTS/12b_analysis_sss_ridge.png         -- ridge plot by latitude band
  PLOTS/12b_analysis_sss_correlations.png  -- pairwise correlation heatmap

Skip logic: skipped if both outputs already exist.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from pathlib import Path

from config import FEATURES, PLOTS, SSS_YEARS

# ── domain ────────────────────────────────────────────────────────────────────
LAT_S    = -16.0
LAT_N    =  -6.0
LON_W    = -82.0
LON_E    = -74.0
BAND_DEG =   1.0    # width of each latitude band in degrees

CLIM_YEARS = list(range(2015, 2025))   # years used for correlation analysis


# ── helpers ───────────────────────────────────────────────────────────────────
def load_monthly_bands():
    """
    Load weekly SSS anomaly files year by year, restrict to coastal domain,
    resample to monthly mean, then compute the mean over longitude for each
    1-deg latitude band.

    Returns a DataFrame: index = month (Period), columns = band center labels.
    """
    monthly_list = []

    for year in SSS_YEARS:
        f = FEATURES / f"sss_anomaly_weekly_{year}.nc"
        if not f.exists():
            print(f"  {year}: file not found, skipping")
            continue

        ds = xr.open_dataset(f)
        da = ds["sss_anomaly"]

        # Rename coords if needed
        if "latitude" in da.coords:
            da = da.rename({"latitude": "lat", "longitude": "lon"})

        # Spatial subset - try both slice orders
        da_sub = da.sel(lat=slice(LAT_N, LAT_S), lon=slice(LON_W, LON_E))
        if len(da_sub.lat) == 0:
            da_sub = da.sel(lat=slice(LAT_S, LAT_N), lon=slice(LON_W, LON_E))

        # Resample weekly -> monthly mean
        monthly = da_sub.resample(time="1MS").mean(dim="time")

        # Average over longitude -> monthly mean per latitude pixel
        monthly_lat = monthly.mean(dim="lon").compute()   # (months, lat)

        monthly_list.append(monthly_lat)
        print(f"  {year}: {len(da_sub.lat)} lat x {len(da_sub.lon)} lon pixels, "
              f"{len(monthly_lat.time)} monthly steps")
        ds.close()

    if not monthly_list:
        raise FileNotFoundError(
            f"No sss_anomaly_weekly_*.nc files found in {FEATURES}")

    # Concatenate all years along time
    combined = xr.concat(monthly_list, dim="time")

    print(f"  Total monthly time steps: {len(combined.time)}")

    # Define latitude bands
    band_edges   = np.arange(LAT_S, LAT_N + BAND_DEG, BAND_DEG)  # e.g. -16,-15,...,-6
    band_centers = band_edges[:-1] + BAND_DEG / 2

    records = {}
    for lo, hi in zip(band_edges[:-1], band_edges[1:]):
        mask      = (combined.lat >= lo) & (combined.lat < hi)
        band_data = combined.isel(lat=mask).mean(dim="lat")
        label     = f"{int(abs(lo))}-{int(abs(hi))}S"
        records[label] = band_data.values

    # Build DataFrame with time index
    times = pd.to_datetime(combined.time.values).to_period("M")
    df    = pd.DataFrame(records, index=times)

    # Sort columns south -> north (ascending lat edge = decreasing abs value)
    df = df[sorted(df.columns, key=lambda s: int(s.split("-")[0]))]

    print(f"  Bands: {list(df.columns)}")
    return df


# ── ridge plot ────────────────────────────────────────────────────────────────
def plot_ridge(df, outpath):
    bands   = list(df.columns)
    n_bands = len(bands)
    times   = df.index.to_timestamp()

    fig, axes = plt.subplots(
        n_bands, 1, figsize=(14, 1.1 * n_bands),
        sharex=True,
        gridspec_kw={"hspace": 0.05},
    )

    vmax = df.abs().quantile(0.97).max()

    for ax, band in zip(axes, bands):
        y = df[band].values
        t = times

        # Fill positive (saltier / positive anomaly) and negative (fresher / El Nino)
        ax.fill_between(t, 0, y,
                        where=(y >= 0), color="#c0392b", alpha=0.75, lw=0)
        ax.fill_between(t, 0, y,
                        where=(y < 0),  color="#2980b9", alpha=0.75, lw=0)
        ax.plot(t, y, color="black", lw=0.4, alpha=0.6)
        ax.axhline(0, color="black", lw=0.6, alpha=0.4)
        ax.set_ylim(-vmax * 1.4, vmax * 1.4)
        ax.set_yticks([])

        # Band label on the right
        ax.set_ylabel(band, fontsize=8, rotation=0,
                      labelpad=30, va="center")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        # Light horizontal guide
        ax.axhspan(-vmax * 0.25, vmax * 0.25, color="grey", alpha=0.05, zorder=0)

    # X axis formatting on bottom panel
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[7]))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), fontsize=8, rotation=30)

    # Color bar
    sm = plt.cm.ScalarMappable(
        cmap=mcolors.LinearSegmentedColormap.from_list(
            "rw_wb", ["#2980b9", "white", "#c0392b"]),
        norm=mcolors.Normalize(-vmax, vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical",
                        fraction=0.015, pad=0.01, shrink=0.8)
    cbar.set_label("SSS anomaly (PSU)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {outpath}")


# ── correlation heatmap ───────────────────────────────────────────────────────
def plot_correlations(df, outpath):
    # Restrict to CLIM_YEARS for correlation analysis
    clim_mask = df.index.year.isin(CLIM_YEARS)
    df_clim   = df[clim_mask]

    corr = df_clim.dropna().corr(method="pearson")

    bands = list(corr.columns)
    n     = len(bands)

    fig, ax = plt.subplots(figsize=(7, 6))

    vmin, vmax = 0.0, 1.0
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=vmin, vmax=vmax,
                   aspect="equal")
    plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            r   = corr.iloc[i, j]
            col = "white" if r < 0.5 else "black"
            ax.text(j, i, f"{r:.2f}", ha="center", va="center",
                    fontsize=7.5, color=col)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(bands, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(bands, fontsize=8)
    clim_str = f"{min(CLIM_YEARS)}-{max(CLIM_YEARS)}"
    ax.set_title(
        f"Correlaciones Pearson entre bandas de latitud\n"
        f"Anomalia SSS mensual  |  zona costera Peru  |  {clim_str}",
        fontsize=9, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Summary stats
    lower_tri = corr.values[np.tril_indices(n, k=-1)]
    ax.text(0.02, -0.18,
            f"r mediana = {np.median(lower_tri):.2f}   "
            f"r minimo = {lower_tri.min():.2f}   "
            f"r maximo = {lower_tri.max():.2f}",
            transform=ax.transAxes, fontsize=8, color="#333333")

    plt.tight_layout()
    plt.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {outpath}")

    # Print summary
    print("\nPairwise Pearson correlations (lower triangle):")
    print(corr.round(2).to_string())
    print(f"\nSummary: median r={np.median(lower_tri):.2f}, "
          f"min={lower_tri.min():.2f}, max={lower_tri.max():.2f}")
    pct_high = (lower_tri > 0.7).mean() * 100
    print(f"Fraction of pairs with r > 0.7: {pct_high:.0f}%")

    return corr, lower_tri


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    out_ridge = PLOTS / "12b_analysis_sss_ridge.png"
    out_corr  = PLOTS / "12b_analysis_sss_correlations.png"

    if out_ridge.exists() and out_corr.exists():
        print("step12b outputs exist -- skipping")
        return

    print("Loading SSS anomaly data...")
    df = load_monthly_bands()

    if not out_ridge.exists():
        print("\nPlotting ridge plot...")
        plot_ridge(df, out_ridge)

    if not out_corr.exists():
        print("\nPlotting correlation heatmap...")
        plot_correlations(df, out_corr)


if __name__ == "__main__":
    main()
