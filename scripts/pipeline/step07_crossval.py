"""
Step 07 - HYCOM vs MODIS Cross-Sensor Validation

Inputs:
  FEATURES/hycom_water_temp_daily_{year}.nc   (from step 00)
  FEATURES/sst_daily_{year}.nc                (from step 04b)

Outputs:
  PLOTS/step07_crossval.png

Skip logic: skipped if PLOTS/step07_crossval.png already exists.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.spatial import cKDTree

from config import FEATURES, PLOTS

PROJ = ccrs.PlateCarree()
EXT  = [-90.0, -70.0, -20.0, 0.0]
TAB  = plt.cm.tab10.colors


def add_gridlines(ax):
    gl = ax.gridlines(crs=PROJ, draw_labels=True,
                      linewidth=0.4, color="grey", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}


def base_map(ax):
    ax.set_extent(EXT, crs=PROJ)
    ax.add_feature(cfeature.LAND, facecolor="wheat", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":", zorder=3)
    add_gridlines(ax)


def step0_crossval(SAMPLE_YEAR=2021):
    hf = FEATURES / f"hycom_water_temp_daily_{SAMPLE_YEAR}.nc"
    mf = FEATURES / f"sst_daily_{SAMPLE_YEAR}.nc"
    if not hf.exists() or not mf.exists():
        print("Cross-sensor validation: missing files, skipping")
        return

    ds_h = xr.open_dataset(hf)
    ds_m = xr.open_dataset(mf)

    if "depth" in ds_h.dims:
        ds_h = ds_h.isel(depth=0)

    h_lats = ds_h["lat"].values
    h_lons_raw = ds_h["lon"].values
    h_lons = np.where(h_lons_raw > 180, h_lons_raw - 360, h_lons_raw)
    h_times = pd.to_datetime(ds_h["time"].values)
    m_lats  = ds_m["lat"].values
    m_lons  = ds_m["lon"].values
    m_times = pd.to_datetime(ds_m["time"].values)

    hlon2, hlat2 = np.meshgrid(h_lons, h_lats)
    tree = cKDTree(np.column_stack([hlon2.ravel(), hlat2.ravel()]))

    common_days = sorted(set(h_times) & set(m_times))
    print(f"Cross-sensor: {len(common_days)} common days in {SAMPLE_YEAR}")

    n_cells = len(h_lats) * len(h_lons)
    cell_h  = [[] for _ in range(n_cells)]
    cell_m  = [[] for _ in range(n_cells)]
    pairs_h, pairs_m = [], []

    for day in common_days:
        hidx = np.where(h_times == day)[0][0]
        midx = np.where(m_times == day)[0][0]
        sst_slice = ds_m["sst"].isel(time=midx).values
        valid = np.argwhere(np.isfinite(sst_slice))
        if len(valid) == 0:
            continue
        m_pts = np.column_stack([m_lons[valid[:, 1]], m_lats[valid[:, 0]]])
        _, nn_idx = tree.query(m_pts)
        h_arr = ds_h["water_temp"].isel(time=hidx).values
        for k in range(len(valid)):
            mv = float(sst_slice[valid[k, 0], valid[k, 1]])
            ri = nn_idx[k] // len(h_lons)
            ci = nn_idx[k] % len(h_lons)
            hv = float(h_arr[ri, ci])
            if np.isfinite(hv):
                cell_h[nn_idx[k]].append(hv)
                cell_m[nn_idx[k]].append(mv)
                pairs_h.append(hv)
                pairs_m.append(mv)

    ds_h.close(); ds_m.close()

    pairs_h = np.array(pairs_h); pairs_m = np.array(pairs_m)
    bias    = pairs_h - pairs_m
    r_global  = np.corrcoef(pairs_h, pairs_m)[0, 1]
    rmse_global = np.sqrt(np.mean(bias**2))
    print(f"r={r_global:.3f}  RMSE={rmse_global:.3f} C  bias={bias.mean():+.3f} C")

    map_lons, map_lats, map_r, map_bias = [], [], [], []
    for ci in range(n_cells):
        if len(cell_h[ci]) >= 3:
            r_val = np.corrcoef(cell_h[ci], cell_m[ci])[0, 1]
            b_val = np.mean(np.array(cell_h[ci]) - np.array(cell_m[ci]))
            row = ci // len(h_lons); col = ci % len(h_lons)
            map_lons.append(hlon2[row, col]); map_lats.append(hlat2[row, col])
            map_r.append(r_val);              map_bias.append(b_val)

    map_lons = np.array(map_lons); map_lats = np.array(map_lats)
    map_r    = np.array(map_r);    map_bias  = np.array(map_bias)

    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    lim = (min(pairs_m.min(), pairs_h.min()) - 1, max(pairs_m.max(), pairs_h.max()) + 1)
    ax1.hexbin(pairs_m, pairs_h, gridsize=60, cmap="YlOrBr", mincnt=1)
    ax1.plot(lim, lim, "--", color=TAB[2], lw=1.5, label="1:1")
    ax1.set_xlabel("MODIS SST (C)"); ax1.set_ylabel("HYCOM Temp (C)")
    ax1.set_title(f"r={r_global:.3f}  RMSE={rmse_global:.2f}C", loc="left")
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    ax1.legend(frameon=False)

    PERU_EXT = [-83, -74, -17, -4]  # tight Peru fishing domain

    ax2 = fig.add_subplot(1, 3, 2, projection=PROJ)
    ax2.set_extent(PERU_EXT, crs=PROJ)
    ax2.add_feature(cfeature.LAND, facecolor="wheat", zorder=2)
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=3)
    ax2.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":", zorder=3)
    add_gridlines(ax2)
    sc2 = ax2.scatter(map_lons, map_lats, c=map_r, cmap="YlGn",
                      vmin=0, vmax=1, s=14, transform=PROJ, zorder=4)
    plt.colorbar(sc2, ax=ax2, label="Pearson r", shrink=0.8)
    ax2.set_title("Pearson r per HYCOM cell", loc="left")

    ax3 = fig.add_subplot(1, 3, 3, projection=PROJ)
    ax3.set_extent(PERU_EXT, crs=PROJ)
    ax3.add_feature(cfeature.LAND, facecolor="wheat", zorder=2)
    ax3.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=3)
    ax3.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":", zorder=3)
    add_gridlines(ax3)
    vlim = np.percentile(np.abs(map_bias), 95)
    sc3 = ax3.scatter(map_lons, map_lats, c=map_bias, cmap="PuOr_r",
                      vmin=-vlim, vmax=vlim, s=14, transform=PROJ, zorder=4)
    plt.colorbar(sc3, ax=ax3, label="Bias C (HYCOM - MODIS)", shrink=0.8)
    ax3.set_title("Mean bias per HYCOM cell", loc="left")

    plt.tight_layout()
    plt.savefig(PLOTS / "step07_crossval.png", dpi=120, bbox_inches="tight")
    plt.close()


def main():
    if (PLOTS / "step07_crossval.png").exists():
        print("step07_crossval.png exists -- skipping")
        return

    step0_crossval()
    print(f"step07_crossval.png saved to {PLOTS}")


if __name__ == "__main__":
    main()
