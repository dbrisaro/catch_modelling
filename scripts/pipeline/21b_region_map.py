"""
21b_region_map.py

Standalone map of the three fishing zones (North, North Central, South Central)
in English. Does not modify any existing figure.

Output: outputs/21b_region_map.png
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUTS, PLOTS

ZONE_BANDS = [
    # (label, lat_south, lat_north, color)
    ("North",         -7.1,  -5.0,  "#d6604d"),
    ("North Central", -11.0, -7.1,  "#2166ac"),
    ("South Central", -15.8, -11.0, "#4dac26"),
]


def build_fishing_polygon_coords(lat_min=-15.8, lat_max=None):
    df = (pd.read_csv(OUTPUTS / "calas_all_data.csv",
                      usecols=["latitud", "longitud"], low_memory=False)
          .rename(columns={"latitud": "lat", "longitud": "lon"})
          .dropna())
    df = df[df["lat"] > lat_min]
    if lat_max is not None:
        df = df[df["lat"] <= lat_max]
    if df.empty:
        return None, None, None
    band_lo_edges = np.arange(int(np.floor(df["lat"].min())),
                              int(np.ceil(df["lat"].max())), 1.0)
    west_lons, east_lons, band_lats = [], [], []
    for lo in band_lo_edges:
        band = df[(df["lat"] >= lo) & (df["lat"] < lo + 1.0)]
        if len(band) < 20:
            continue
        west_lons.append(np.percentile(band["lon"], 5))
        east_lons.append(np.percentile(band["lon"], 95))
        band_lats.append(lo + 0.5)
    return np.array(west_lons), np.array(east_lons), np.array(band_lats)


def main():
    fig = plt.figure(figsize=(4, 7))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ax.set_extent([-82, -74, -17, -4], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN,     facecolor="#d6eaf8", zorder=0)
    ax.add_feature(cfeature.LAND,      facecolor="#f0ede8", zorder=7)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7,       zorder=8)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.4, linestyle=":", zorder=8)

    lon_w, lon_e = -82, -74
    # label x positions: North gets pushed right since its band is narrow
    label_x = {"North": -78.5, "North Central": -81.5, "South Central": -81.5}
    for label, lat_s, lat_n, color in ZONE_BANDS:
        rect = mpatches.Rectangle(
            (lon_w, lat_s), lon_e - lon_w, lat_n - lat_s,
            transform=ccrs.PlateCarree(),
            color=color, alpha=0.35, zorder=2, linewidth=0,
        )
        ax.add_patch(rect)
        ax.text(label_x.get(label, -81.5), (lat_s + lat_n) / 2, label,
                transform=ccrs.PlateCarree(),
                fontsize=9, va="center", color=color,
                fontweight="bold", zorder=9)

    for lat in [-7.1, -11.0, -15.8]:
        ax.plot([lon_w, lon_e], [lat, lat],
                transform=ccrs.PlateCarree(),
                color="black", lw=0.8, ls="--", alpha=0.5, zorder=9)

    west_lons, east_lons, band_lats = build_fishing_polygon_coords()
    if west_lons is not None:
        lat_full  = np.concatenate([[band_lats[0] - 0.5], band_lats, [band_lats[-1] + 0.5]])
        west_full = np.concatenate([[west_lons[0]], west_lons, [west_lons[-1]]])
        east_full = np.concatenate([[east_lons[0]], east_lons, [east_lons[-1]]])
        poly_lons = np.concatenate([west_full, east_full[::-1], [west_full[0]]])
        poly_lats = np.concatenate([lat_full,  lat_full[::-1],  [lat_full[0]]])
        # filled polygon with shadow effect
        ax.fill(poly_lons, poly_lats, transform=ccrs.PlateCarree(),
                color="#222222", alpha=0.18, zorder=5)
        ax.plot(poly_lons, poly_lats, transform=ccrs.PlateCarree(),
                color="black", lw=1.4, ls="--", zorder=6,
                path_effects=[pe.withSimplePatchShadow(offset=(2, -2),
                                                       shadow_rgbFace=(0, 0, 0),
                                                       alpha=0.25)])

    # lon/lat tick grid
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="grey", alpha=0.4,
                      x_inline=False, y_inline=False)
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlocator = plt.FixedLocator([-82, -80, -78, -76, -74])
    gl.ylocator = plt.FixedLocator([-16, -14, -12, -10, -8, -6, -4])
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}

    ax.set_title("Fishing zones evaluated", fontsize=10, pad=6, loc="left")

    out = PLOTS / "21b_region_map.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
