"""
Step 15c - SSS anomaly timeseries + distribucion estacional

Analogo a 15_pricing_sst_timeseries.png pero usando anomalia SSS OISSS.

Panel izquierdo: serie temporal de anomalia SSS diaria (area) con puntos
  estacionales T1/T2 superpuestos y lineas de percentiles.
Panel derecho: histograma + KDE de las anomalias estacionales con percentiles.

Usa el poligono de pesca de la region Centro (Norte + Centro Norte + Centro Sur)
para filtrar los pixels OISSS, igual que el plot SST.

Inputs:
  FEATURES/sss_anomaly_weekly_{year}.nc   (de step 03b)
  OUTPUTS/calas_all_data.csv

Outputs:
  PLOTS/15c_pricing_sss_timeseries.png

Skip logic: skipped si el output ya existe.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.path import Path as MplPath
from scipy.stats import gaussian_kde

from config import FEATURES, OUTPUTS, PLOTS, SSS_YEARS

LON_W, LON_E   = -83.0, -74.0
LAT_MIN        = -15.8
LAT_MAX        = None    # todo el corredor (Centro = Norte + Centro)

T1_MONTHS   = [4, 5, 6, 7]   # Abr - Jul
T2_MONTHS   = [11, 12]       # Nov - Dic
T1_N_MONTHS = 4
T2_N_MONTHS = 2

CLIM_YEARS = list(range(2015, 2025))   # periodo de referencia percentiles


# ---------------------------------------------------------------------------
# Fishing polygon
# ---------------------------------------------------------------------------

def build_fishing_polygon(lat_min, lat_max):
    df = (pd.read_csv(OUTPUTS / "calas_all_data.csv",
                      usecols=["latitud", "longitud"], low_memory=False)
          .rename(columns={"latitud": "lat", "longitud": "lon"})
          .dropna())
    df = df[df["lat"] > lat_min]
    if lat_max is not None:
        df = df[df["lat"] <= lat_max]
    if df.empty:
        return None

    actual_min = df["lat"].min()
    actual_max = df["lat"].max()
    band_edges = np.arange(int(np.floor(actual_min)), int(np.ceil(actual_max)), 1.0)

    west_lons, east_lons, valid_lats = [], [], []
    for lo in band_edges:
        band = df[(df["lat"] >= lo) & (df["lat"] < lo + 1.0)]
        if len(band) < 20:
            continue
        west_lons.append(np.percentile(band["lon"], 5))
        east_lons.append(np.percentile(band["lon"], 95))
        valid_lats.append(lo + 0.5)

    if not valid_lats:
        return None

    valid_lats = np.array(valid_lats)
    west_lons  = np.array(west_lons)
    east_lons  = np.array(east_lons)
    lat_full   = np.concatenate([[actual_min], valid_lats, [actual_max]])
    west_full  = np.concatenate([[west_lons[0]], west_lons, [west_lons[-1]]])
    east_full  = np.concatenate([[east_lons[0]], east_lons, [east_lons[-1]]])
    poly_lons  = np.concatenate([west_full, east_full[::-1], [west_full[0]]])
    poly_lats  = np.concatenate([lat_full,  lat_full[::-1],  [lat_full[0]]])

    verts = list(zip(poly_lons, poly_lats))
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
    return MplPath(verts, codes), actual_min, actual_max


# ---------------------------------------------------------------------------
# Load SSS regional timeseries
# ---------------------------------------------------------------------------

def load_sss_timeseries():
    result = build_fishing_polygon(LAT_MIN, LAT_MAX)
    if result is None:
        raise RuntimeError("No fishing polygon built")
    polygon, actual_min, actual_max = result
    lat_lo = actual_min - 0.1
    lat_hi = actual_max + 0.1

    parts = []
    for yr in SSS_YEARS:
        f = FEATURES / f"sss_anomaly_weekly_{yr}.nc"
        if not f.exists():
            continue
        ds    = xr.open_dataset(f)
        lat_v = ds["lat"].values
        lon_v = ds["lon"].values
        lat_m = (lat_v > lat_lo) & (lat_v <= lat_hi)
        lon_m = (lon_v >= LON_W) & (lon_v <= LON_E)
        lat_s = lat_v[lat_m]
        lon_s = lon_v[lon_m]
        G_lon, G_lat = np.meshgrid(lon_s, lat_s)
        pts     = np.column_stack([G_lon.ravel(), G_lat.ravel()])
        mask_2d = polygon.contains_points(pts).reshape(lat_s.size, lon_s.size)
        mask_da = xr.DataArray(mask_2d, dims=["lat", "lon"],
                               coords={"lat": lat_s, "lon": lon_s})
        s = (ds["sss_anomaly"].isel(lat=lat_m, lon=lon_m)
             .where(mask_da).mean(dim=["lat", "lon"]).to_series())
        s.index = pd.to_datetime(s.index)
        ds.close()
        parts.append(s.dropna())

    return pd.concat(parts).sort_index().dropna()


# ---------------------------------------------------------------------------
# Seasonal means
# ---------------------------------------------------------------------------

def seasonal_means(monthly):
    rows = []
    for year in sorted(monthly.index.year.unique()):
        yr_mon = monthly[monthly.index.year == year]
        t1_mon = yr_mon[yr_mon.index.month.isin(T1_MONTHS)]
        t2_mon = yr_mon[yr_mon.index.month.isin(T2_MONTHS)]
        if len(t1_mon) == T1_N_MONTHS:
            rows.append({"year": year, "tipo": "T1",
                         "date": pd.Timestamp(f"{year}-05-15"),
                         "sss": float(t1_mon.mean())})
        if len(t2_mon) == T2_N_MONTHS:
            rows.append({"year": year, "tipo": "T2",
                         "date": pd.Timestamp(f"{year}-11-15"),
                         "sss": float(t2_mon.mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out = PLOTS / "15c_pricing_sss_timeseries.png"
    if out.exists():
        print("15c output exists -- skipping")
        return

    print("Cargando serie temporal SSS...")
    daily   = load_sss_timeseries()
    monthly = daily.resample("ME").mean().dropna()
    print(f"  {len(daily):,} pasos semanales -> {len(monthly)} medias mensuales")
    print(f"  {monthly.index[0].date()} - {monthly.index[-1].date()}")

    seas = seasonal_means(monthly)

    # percentiles sobre periodo de referencia
    clim_mask = seas["year"].isin(CLIM_YEARS)
    sss_clim  = seas.loc[clim_mask, "sss"].values
    p90 = float(np.percentile(sss_clim, 90))
    p95 = float(np.percentile(sss_clim, 95))
    p99 = float(np.percentile(sss_clim, 99))
    p10 = float(np.percentile(sss_clim, 10))
    p05 = float(np.percentile(sss_clim,  5))
    p01 = float(np.percentile(sss_clim,  1))
    print(f"  Percentiles ({CLIM_YEARS[0]}-{CLIM_YEARS[-1]}):")
    print(f"    p01={p01:.3f}  p05={p05:.3f}  p10={p10:.3f}")
    print(f"    p90={p90:.3f}  p95={p95:.3f}  p99={p99:.3f}")

    # ── figura ────────────────────────────────────────────────────────────────
    fig, (ax_ts, ax_hist) = plt.subplots(
        1, 2, figsize=(16, 5.5),
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.07})

    # -- timeseries panel --
    dates = monthly.index
    vals  = monthly.values

    # filled area: positive (high sal) vs negative (low sal)
    ax_ts.fill_between(dates, 0, vals, where=(vals >= 0),
                       color="#d6604d", alpha=0.25, label="_nolegend_")
    ax_ts.fill_between(dates, 0, vals, where=(vals < 0),
                       color="#4393c3", alpha=0.25, label="_nolegend_")
    ax_ts.plot(dates, vals, color="black", lw=0.4, alpha=0.5, zorder=2)

    # seasonal points
    t1 = seas[seas["tipo"] == "T1"]
    t2 = seas[seas["tipo"] == "T2"]
    ax_ts.scatter(t1["date"], t1["sss"], color="#d6604d", s=40, zorder=5,
                  marker="o", label="T1 (abr-jul)")
    ax_ts.scatter(t2["date"], t2["sss"], color="#4393c3", s=40, zorder=5,
                  marker="s", label="T2 (nov-dic)")

    # annotate low-side extremes (below p05)
    for _, row in seas.iterrows():
        if row["sss"] <= p05:
            ax_ts.annotate(
                f"{int(row['year'])}-{row['tipo']}",
                (row["date"], row["sss"]),
                xytext=(0, 8 if row["sss"] > 0 else -12),
                textcoords="offset points",
                fontsize=7, ha="center", color="#333333",
            )

    # percentile lines (low side only - SSS trigger is inverted)
    for val, ls, col, lbl in [
        (p10, ":", "#74b9ff", f"p10 ({CLIM_YEARS[0]}-{CLIM_YEARS[-1]}) = {p10:.2f} PSU"),
        (p05, ":", "#2196f3", f"p05 ({CLIM_YEARS[0]}-{CLIM_YEARS[-1]}) = {p05:.2f} PSU"),
        (p01, ":", "#0984e3", f"p01 ({CLIM_YEARS[0]}-{CLIM_YEARS[-1]}) = {p01:.2f} PSU"),
    ]:
        ax_ts.axhline(val, color=col, lw=1.2, ls=ls, alpha=0.85, label=lbl)

    ax_ts.axhline(0, color="grey", lw=0.7, ls=":", alpha=0.5)
    ax_ts.set_ylabel("SSS anomalia OISSS (PSU)", fontsize=10)
    ax_ts.set_title(
        f"SSS anomalia - Region Centro fishing pixels  (OISSS {SSS_YEARS[0]}-{SSS_YEARS[-1]})",
        fontsize=11, loc="left")
    ax_ts.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=2)
    ax_ts.spines["top"].set_visible(False)
    ax_ts.spines["right"].set_visible(False)

    # -- histogram panel --
    all_vals = sss_clim
    ax_hist.hist(all_vals, bins=14, orientation="horizontal",
                 color="#aaaaaa", alpha=0.7, density=True)

    kde_y = np.linspace(all_vals.min() - 0.05, all_vals.max() + 0.05, 300)
    kde   = gaussian_kde(all_vals, bw_method=0.4)
    ax_hist.plot(kde(kde_y), kde_y, color="black", lw=1.5)

    for val, ls, col in [
        (p10, ":", "#74b9ff"), (p05, ":", "#2196f3"), (p01, ":", "#0984e3"),
    ]:
        ax_hist.axhline(val, color=col, lw=1.2, ls=ls, alpha=0.85)

    ax_hist.axhline(0, color="grey", lw=0.7, ls=":", alpha=0.5)
    ax_hist.set_xlabel("Densidad", fontsize=9)
    ax_hist.set_title(
        f"Distribucion\nanomalias de temporada ({CLIM_YEARS[0]}-{CLIM_YEARS[-1]})",
        fontsize=9)
    ax_hist.yaxis.set_label_position("right")
    ax_hist.yaxis.tick_right()

    # sync y limits
    y_all = np.concatenate([vals, sss_clim])
    ypad  = (y_all.max() - y_all.min()) * 0.08
    ylim  = (y_all.min() - ypad, y_all.max() + ypad)
    ax_ts.set_ylim(ylim)
    ax_hist.set_ylim(ylim)

    # legend for percentiles on histogram
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="#74b9ff", ls=":", lw=1.2, label=f"p10={p10:.2f} PSU"),
        Line2D([0], [0], color="#2196f3", ls=":", lw=1.2, label=f"p05={p05:.2f} PSU"),
        Line2D([0], [0], color="#0984e3", ls=":", lw=1.2, label=f"p01={p01:.2f} PSU"),
    ]
    ax_hist.legend(handles=legend_handles, fontsize=7, frameon=False,
                   loc="upper right")
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["left"].set_visible(False)

    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
