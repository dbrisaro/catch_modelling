"""
Step 16b - SSS spatial comparison: log(catch) ~ SSS_anom OISSS

Analogo exacto a la Part A de 16_pricing_trigger_design.py pero con
oisss_sss_anom como predictor en lugar de modis_sst_anom.

OLS log(catch) ~ SSS_anom a nivel empresa x temporada para:
  All (Norte+Centro), Norte, Centro Norte, Centro Sur, Centro

Inputs:
  OUTPUTS/calas_enriched.csv   (requiere columna oisss_sss_anom, de step 07b)

Outputs:
  PLOTS/16b_analysis_sss_spatial_comparison.png

Skip logic: skipped si el output ya existe.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import statsmodels.formula.api as smf
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from config import OUTPUTS, PLOTS

SCOPES = [
    ("All\n(Norte+Centro)", -15.8, None),
    ("Norte",               -7.1,  None),
    ("Centro Norte",        -11.0, -7.1),
    ("Centro Sur",          -15.8, -11.0),
    ("Centro",              -15.8, -7.1),
]

SCOPE_COLORS = {
    "All (Norte+Centro)": "#555555",
    "Norte":              "#d6604d",
    "Centro Norte":       "#2166ac",
    "Centro Sur":         "#4dac26",
    "Centro":             "#92c5de",
}

ZONE_BANDS = [
    ("Norte",        -7.1,  -5.0,  "#d6604d"),
    ("Centro Norte", -11.0, -7.1,  "#2166ac"),
    ("Centro Sur",   -15.8, -11.0, "#4dac26"),
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
    actual_lat_min = df["lat"].min()
    actual_lat_max = df["lat"].max()
    band_lo_edges  = np.arange(int(np.floor(actual_lat_min)),
                               int(np.ceil(actual_lat_max)), 1.0)
    west_lons, east_lons, band_lats = [], [], []
    for lo in band_lo_edges:
        band = df[(df["lat"] >= lo) & (df["lat"] < lo + 1.0)]
        if len(band) < 20:
            continue
        west_lons.append(np.percentile(band["lon"], 5))
        east_lons.append(np.percentile(band["lon"], 95))
        band_lats.append(lo + 0.5)
    return np.array(west_lons), np.array(east_lons), np.array(band_lats)


def plot_zone_map(ax):
    ax.set_extent([-82, -74, -17, -4], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,      facecolor="#f0ede8", zorder=1)
    ax.add_feature(cfeature.OCEAN,     facecolor="#d6eaf8", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=3)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.4, linestyle=":", zorder=3)

    lon_w, lon_e = -82, -74
    for label, lat_s, lat_n, color in ZONE_BANDS:
        rect = mpatches.Rectangle(
            (lon_w, lat_s), lon_e - lon_w, lat_n - lat_s,
            transform=ccrs.PlateCarree(),
            color=color, alpha=0.35, zorder=2, linewidth=0,
        )
        ax.add_patch(rect)
        ax.text(-81.5, (lat_s + lat_n) / 2, label,
                transform=ccrs.PlateCarree(),
                fontsize=7, va="center", color=color,
                fontweight="bold", zorder=4)

    for lat in [-7.1, -11.0, -15.8]:
        ax.plot([lon_w, lon_e], [lat, lat],
                transform=ccrs.PlateCarree(),
                color="black", lw=0.8, ls="--", alpha=0.5, zorder=4)
        ax.text(-73.8, lat, f"{abs(lat):.1f}°S",
                transform=ccrs.PlateCarree(),
                fontsize=6.5, va="center", color="#333333", zorder=5)

    west_lons, east_lons, band_lats = build_fishing_polygon_coords()
    if west_lons is not None:
        actual_lat_min = band_lats[0] - 0.5
        actual_lat_max = band_lats[-1] + 0.5
        lat_full  = np.concatenate([[actual_lat_min], band_lats, [actual_lat_max]])
        west_full = np.concatenate([[west_lons[0]], west_lons, [west_lons[-1]]])
        east_full = np.concatenate([[east_lons[0]], east_lons, [east_lons[-1]]])
        poly_lons = np.concatenate([west_full, east_full[::-1], [west_full[0]]])
        poly_lats = np.concatenate([lat_full,  lat_full[::-1],  [lat_full[0]]])
        ax.plot(poly_lons, poly_lats, transform=ccrs.PlateCarree(),
                color="black", lw=1.2, ls="--", zorder=5)

    ax.gridlines(draw_labels=False, linewidth=0.3, color="grey", alpha=0.4)
    ax.set_title("Zonas espaciales\nevaluadas", fontsize=8, pad=4)


def run_ols_seasonal(sub):
    agg = sub.groupby(["company", "season"]).agg(
        total_catch=("catch_tm", "sum"),
        mean_sss=("oisss_sss_anom", "mean"),
    ).reset_index().dropna(subset=["mean_sss"])
    agg = agg[agg["total_catch"] > 0]
    if len(agg) < 10:
        return None, agg
    m = smf.ols("np.log(total_catch) ~ mean_sss", data=agg).fit()
    return m, agg


def main():
    out = PLOTS / "16b_analysis_sss_spatial_comparison.png"
    if out.exists():
        print("16b output exists -- skipping")
        return

    df = pd.read_csv(OUTPUTS / "calas_enriched.csv", low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["catch_tm"] > 0].dropna(
        subset=["oisss_sss_anom", "catch_tm", "lat", "season", "company"])

    n_scopes = len(SCOPES)
    fig = plt.figure(figsize=(22, 5.5))
    gs  = gridspec.GridSpec(1, n_scopes + 1,
                            wspace=0.35,
                            width_ratios=[1.4] + [1] * n_scopes)

    ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    plot_zone_map(ax_map)

    print("\nSpatial comparison SSS (empresa x temporada):")
    for col, (name, lat_min, lat_max) in enumerate(SCOPES):
        sub = df[df["lat"] > lat_min].copy()
        if lat_max is not None:
            sub = sub[sub["lat"] <= lat_max]

        m, agg = run_ols_seasonal(sub)
        short = name.replace("\n", " ")
        col_c = SCOPE_COLORS[short]

        ax_sc = fig.add_subplot(gs[0, col + 1])
        ax_sc.scatter(agg["mean_sss"], np.log(agg["total_catch"]),
                      color=col_c, alpha=0.6, s=20, zorder=3)

        if m is not None:
            x_line = np.linspace(agg["mean_sss"].min() - 0.05,
                                 agg["mean_sss"].max() + 0.05, 100)
            ax_sc.plot(x_line,
                       m.params["Intercept"] + m.params["mean_sss"] * x_line,
                       color="black", lw=1.5, ls="--", alpha=0.8)
            beta = m.params["mean_sss"]
            p    = m.pvalues["mean_sss"]
            r2   = m.rsquared
            ci   = m.conf_int().loc["mean_sss"]
            pstr = "***" if p < 0.001 else ("**" if p < 0.01
                   else ("*" if p < 0.05 else "ns"))
            ax_sc.text(0.04, 0.03,
                       f"beta = {beta:+.3f}{pstr}\n"
                       f"95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]\n"
                       f"R² = {r2:.3f}  N = {len(agg):,}",
                       transform=ax_sc.transAxes, va="bottom", fontsize=8)
            print(f"  {short:25s}  beta={beta:+.3f}  R²={r2:.3f}  "
                  f"p={p:.4f} {pstr}  N={len(agg)}")
        else:
            print(f"  {short:25s}  insuficientes datos")

        ax_sc.axvline(0, color="grey", lw=0.7, ls=":", alpha=0.5)
        ax_sc.set_title(short, fontsize=9, color=col_c, fontweight="bold")
        ax_sc.set_xlabel("Anomalia SSS promedio (PSU)", fontsize=7.5)
        if col == 0:
            ax_sc.set_ylabel("log(captura empresa x temporada)", fontsize=7.5)
        ax_sc.spines["top"].set_visible(False)
        ax_sc.spines["right"].set_visible(False)

    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
