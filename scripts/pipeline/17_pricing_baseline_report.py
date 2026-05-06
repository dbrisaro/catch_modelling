"""
Report - Baseline catch and SST trigger percentiles by region

Outputs:
  OUTPUTS/report_baseline.md

For each region (Norte, Centro Norte, Centro Sur, Centro):
  - Mean seasonal catch T1 and T2 per year (from catch_summary_by_region.csv)
  - SST anomaly seasonal distribution (spatial mean over fishing polygon,
    2004-2025, climatology 2005-2024) -> p80, p90, p95

SST is computed only within the fishing corridor:
  For each 1-degree lat band in the region, the fishing polygon spans the
  5th-95th percentile of observed cala longitudes (same method as step13).
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.path import Path

from config import FEATURES, OUTPUTS, PLOTS

# ── regions ───────────────────────────────────────────────────────────────────
REGIONS = [
    ("Norte",        -7.1,  None),
    ("Centro Norte", -11.0, -7.1),
    ("Centro Sur",   -15.8, -11.0),
    ("Centro",       -15.8, -7.1),
]

SST_YEARS    = list(range(2005, 2025))   # same as climatology baseline
LON_W, LON_E = -85.0, -70.0     # broad lon bbox for SST slice

T1_DOY_START, T1_DOY_END = 91, 212    # Apr 1 - Jul 31
T2_DOY_START, T2_DOY_END = 305, 365   # Nov 1 - Dec 31
MIN_DAYS_T1, MIN_DAYS_T2 = 30, 15


# ── fishing polygon per region ────────────────────────────────────────────────
def build_fishing_polygon(lat_min, lat_max):
    """
    Fishing corridor for a region: 5th-95th percentile of observed cala
    longitudes per 1-degree lat band. Returns matplotlib.path.Path.
    lat_min: southern boundary (e.g. -11.0)
    lat_max: northern boundary (e.g. -7.1); None = use calas north limit
    """
    df = pd.read_csv(
        OUTPUTS / "calas_all_data.csv",
        usecols=["latitud", "longitud"], low_memory=False,
    ).rename(columns={"latitud": "lat", "longitud": "lon"}).dropna()

    df = df[df["lat"] > lat_min]
    if lat_max is not None:
        df = df[df["lat"] <= lat_max]

    if df.empty:
        return None

    actual_lat_min = df["lat"].min()
    actual_lat_max = df["lat"].max()

    band_lo_edges = np.arange(int(np.floor(actual_lat_min)),
                              int(np.ceil(actual_lat_max)), 1.0)

    west_lons, east_lons, valid_lats = [], [], []
    for lo in band_lo_edges:
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

    lat_full  = np.concatenate([[actual_lat_min], valid_lats, [actual_lat_max]])
    west_full = np.concatenate([[west_lons[0]], west_lons, [west_lons[-1]]])
    east_full = np.concatenate([[east_lons[0]], east_lons, [east_lons[-1]]])

    poly_lons = np.concatenate([west_full, east_full[::-1], [west_full[0]]])
    poly_lats = np.concatenate([lat_full,  lat_full[::-1],  [lat_full[0]]])

    verts = list(zip(poly_lons, poly_lats))
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
    return Path(verts, codes), actual_lat_min, actual_lat_max


def polygon_spatial_mean(ds, var, fishing_path, reg_lat_min, reg_lat_max):
    """Spatial mean of var within fishing polygon for dates in ds."""
    lat_vals = ds["lat"].values
    lon_vals = ds["lon"].values

    lat_m = (lat_vals > reg_lat_min) & (lat_vals <= reg_lat_max)
    lon_m = (lon_vals >= LON_W) & (lon_vals <= LON_E)
    lat_sub = lat_vals[lat_m]
    lon_sub = lon_vals[lon_m]

    LON_2D, LAT_2D = np.meshgrid(lon_sub, lat_sub)
    points  = np.column_stack([LON_2D.ravel(), LAT_2D.ravel()])
    mask_2d = fishing_path.contains_points(points).reshape(lat_sub.size, lon_sub.size)

    mask_da = xr.DataArray(mask_2d, dims=["lat", "lon"],
                           coords={"lat": lat_sub, "lon": lon_sub})
    series = (ds[var].isel(lat=lat_m, lon=lon_m)
              .where(mask_da).mean(dim=["lat", "lon"]).to_series())
    series.index = pd.to_datetime(series.index)
    return series.dropna()


# ── SST daily series per region ───────────────────────────────────────────────
MIN_DAYS_YEAR = 200   # minimum valid days to consider a year complete


def compute_sst_daily(region_name, lat_min, lat_max):
    """
    Returns (all_daily_vals, year_df) where all_daily_vals is a pd.Series of
    all daily spatial-mean SST anomaly values from complete years (>= MIN_DAYS_YEAR),
    and year_df is a per-year summary DataFrame.
    """
    result = build_fishing_polygon(lat_min, lat_max)
    if result is None:
        print(f"  {region_name}: no fishing polygon found")
        return pd.Series(dtype=float), pd.DataFrame()
    fishing_path, actual_lat_min, actual_lat_max = result

    reg_lat_min = actual_lat_min - 0.1
    reg_lat_max = actual_lat_max + 0.1
    print(f"  {region_name}: lat {actual_lat_min:.1f} to {actual_lat_max:.1f}")

    parts = []
    for yr in SST_YEARS:
        f = FEATURES / f"sst_anomaly_daily_{yr}.nc"
        if not f.exists():
            continue
        ds = xr.open_dataset(f)
        s = polygon_spatial_mean(ds, "sst_anomaly", fishing_path,
                                 reg_lat_min, reg_lat_max)
        ds.close()
        parts.append(s)

    if not parts:
        return pd.Series(dtype=float), pd.DataFrame()

    daily = pd.concat(parts).sort_index().dropna()

    complete_days = []
    for yr in SST_YEARS:
        yr_vals = daily[daily.index.year == yr]
        if len(yr_vals) >= MIN_DAYS_YEAR:
            complete_days.append(yr_vals)

    if not complete_days:
        return pd.Series(dtype=float), pd.DataFrame()

    all_vals = pd.concat(complete_days)
    monthly  = all_vals.resample("MS").mean().dropna()

    # seasonal means: mean of monthly means within T1 and T2 per year
    seasonal_vals = []
    T1_MONTHS_SET = {4, 5, 6, 7}
    T2_MONTHS_SET = {11, 12}
    for yr in SST_YEARS:
        yr_mon = monthly[monthly.index.year == yr]
        t1_mon = yr_mon[yr_mon.index.month.isin(T1_MONTHS_SET)]
        t2_mon = yr_mon[yr_mon.index.month.isin(T2_MONTHS_SET)]
        if len(t1_mon) == 4:
            seasonal_vals.append(float(t1_mon.mean()))
        if len(t2_mon) == 2:
            seasonal_vals.append(float(t2_mon.mean()))

    seasonal = pd.Series(seasonal_vals)
    return all_vals, monthly, seasonal


# ── catch table ───────────────────────────────────────────────────────────────
def load_catch():
    df = pd.read_csv(OUTPUTS / "catch_summary_by_region.csv")
    df["temporada_label"] = df["temporada"].map({"1ra": "T1", "2da": "T2"})
    return df


# ── markdown helpers ──────────────────────────────────────────────────────────
def md_table(headers, rows, fmts=None):
    lines = ["| " + " | ".join(headers) + " |"]
    lines += ["| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        if fmts:
            cells = []
            for v, f in zip(row, fmts):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    cells.append("-")
                elif f == "int":
                    cells.append(f"{int(v):,}")
                elif f == "float1":
                    cells.append(f"{v:.1f}")
                elif f == "float3":
                    cells.append(f"{v:.3f}")
                else:
                    cells.append(str(v))
        else:
            cells = [str(c) for c in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    out = PLOTS / "report_baseline.md"
    catch_df = load_catch()

    reg_names = [r[0] for r in REGIONS]

    lines = [
        "# Baseline Capture and SST Trigger Report",
        "",
        f"Climatologia SST: 2005-2024 (MODIS AQUA).  "
        f"SST calculada dentro del corredor de pesca (p5-p95 longitud por banda de 1 grado lat).  ",
        f"Percentiles de SST calculados sobre la distribucion completa de valores diarios, anios completos 2005-2024.",
        "",
    ]

    # ── Section 1: catch summary per region ──────────────────────────────────
    lines += ["## 1. Captura media por region (tm, 2015-2025)", ""]

    T1_MONTHS, T2_MONTHS = 4, 2  # T1: Apr-Jul, T2: Nov-Dic
    def _fmt3(v): return f"{v:.3f}" if v is not None else "-"

    # Compute SST percentiles per region (daily, monthly, seasonal mean-of-monthly-means)
    sst_daily    = {}
    sst_monthly  = {}
    sst_seasonal = {}
    all_daily_pooled    = []
    all_monthly_pooled  = []
    all_seasonal_pooled = []
    for reg_name, lat_min, lat_max in REGIONS:
        print(f"\nSST {reg_name}...")
        daily_vals, monthly_vals, seasonal_vals = compute_sst_daily(reg_name, lat_min, lat_max)

        if len(daily_vals) == 0:
            sst_daily[reg_name]    = (None, None, None, None)
            sst_monthly[reg_name]  = (None, None, None, None)
            sst_seasonal[reg_name] = (None, None, None, None)
        else:
            def _pcts(v):
                return tuple(float(np.percentile(v, p)) for p in (80, 90, 95, 99))
            sst_daily[reg_name]    = _pcts(daily_vals)
            sst_monthly[reg_name]  = _pcts(monthly_vals)
            sst_seasonal[reg_name] = _pcts(seasonal_vals) if len(seasonal_vals) > 0 else (None, None, None, None)
            if reg_name != "Centro":
                all_daily_pooled.append(daily_vals)
                all_monthly_pooled.append(monthly_vals)
                all_seasonal_pooled.append(seasonal_vals)

    def _pool_pcts(series_list):
        if not series_list:
            return (None, None, None, None)
        v = pd.concat([pd.Series(s) if not isinstance(s, pd.Series) else s for s in series_list])
        return tuple(float(np.percentile(v, p)) for p in (80, 90, 95, 99))

    sst_total_daily    = _pool_pcts(all_daily_pooled)
    sst_total_monthly  = _pool_pcts(all_monthly_pooled)
    sst_total_seasonal = _pool_pcts(all_seasonal_pooled)

    headers = ["Region",
               "Media mensual T1 (tm)", "Media mensual T2 (tm)", "Media anual (tm)",
               "p80 SST (C)", "p90 SST (C)", "p95 SST (C)"]
    rows = []
    total_t1, total_t2 = 0.0, 0.0

    for reg_name, lat_min, lat_max in REGIONS:
        if reg_name == "Centro":
            continue
        reg_catch = catch_df[catch_df["region"] == reg_name].copy()
        if reg_catch.empty:
            continue
        t1 = reg_catch[reg_catch["temporada"] == "1ra"]["captura_tm"]
        t2 = reg_catch[reg_catch["temporada"] == "2da"]["captura_tm"]
        m1 = t1.mean()
        m2 = t2.mean()
        p80, p90, p95, p99 = sst_daily.get(reg_name, (None, None, None, None))
        rows.append([reg_name, m1 / T1_MONTHS, m2 / T2_MONTHS, m1 + m2,
                     _fmt3(p80), _fmt3(p90), _fmt3(p95), _fmt3(p99)])
        total_t1 += m1
        total_t2 += m2
    rows.append(["**Total**",
                 total_t1 / T1_MONTHS, total_t2 / T2_MONTHS, total_t1 + total_t2,
                 _fmt3(sst_total_daily[0]), _fmt3(sst_total_daily[1]),
                 _fmt3(sst_total_daily[2]), _fmt3(sst_total_daily[3])])

    headers = ["Region",
               "Media mensual T1 (tm)", "Media mensual T2 (tm)", "Media anual (tm)",
               "p80 SST (C)", "p90 SST (C)", "p95 SST (C)", "p99 SST (C)"]
    fmts = ["str", "int", "int", "int", "str", "str", "str", "str"]
    lines += [md_table(headers, rows, fmts), ""]

    # ── SST monthly percentiles ───────────────────────────────────────────────
    lines += ["## 2. Percentiles de anomalia SST mensual por region (2005-2024)", ""]
    lines += [
        "Misma metodologia pero usando medias mensuales de SST anomalia (en vez de valores diarios).",
        "",
    ]

    mon_rows = []
    for reg_name, lat_min, lat_max in REGIONS:
        if reg_name == "Centro":
            continue
        p80, p90, p95, p99 = sst_monthly.get(reg_name, (None, None, None, None))
        mon_rows.append([reg_name, _fmt3(p80), _fmt3(p90), _fmt3(p95), _fmt3(p99)])
    mon_rows.append(["**Total**",
                     _fmt3(sst_total_monthly[0]), _fmt3(sst_total_monthly[1]),
                     _fmt3(sst_total_monthly[2]), _fmt3(sst_total_monthly[3])])

    lines += [md_table(["Region", "p80 (C)", "p90 (C)", "p95 (C)", "p99 (C)"], mon_rows), ""]

    # ── SST seasonal percentiles (mean of monthly means per season) ───────────
    lines += ["## 3. Percentiles de anomalia SST estacional (media de medias mensuales, 2005-2024)", ""]
    lines += [
        "Para cada temporada y anio: media de las medias mensuales (T1 = media(Abr,May,Jun,Jul), "
        "T2 = media(Nov,Dic)). Pooled T1+T2, N~40.",
        "",
    ]

    sea_rows = []
    for reg_name, lat_min, lat_max in REGIONS:
        if reg_name == "Centro":
            continue
        p80, p90, p95, p99 = sst_seasonal.get(reg_name, (None, None, None, None))
        sea_rows.append([reg_name, _fmt3(p80), _fmt3(p90), _fmt3(p95), _fmt3(p99)])
    sea_rows.append(["**Total**",
                     _fmt3(sst_total_seasonal[0]), _fmt3(sst_total_seasonal[1]),
                     _fmt3(sst_total_seasonal[2]), _fmt3(sst_total_seasonal[3])])

    lines += [md_table(["Region", "p80 (C)", "p90 (C)", "p95 (C)", "p99 (C)"], sea_rows), ""]

    # ── Section 4: trigger thresholds for Centro Norte pricing ───────────────
    lines += ["## 4. Triggers de pricing - Centro Norte (2005-2024)", ""]
    lines += [
        "Basado en la distribucion de medias estacionales (T1+T2 pooled, N~40) de Centro Norte. "
        "Estos son los valores usados como lineas de referencia en step 15.",
        "",
    ]
    p80_cn, p90_cn, p95_cn, p99_cn = sst_seasonal.get("Centro Norte", (None, None, None, None))
    trig_rows = [
        ["p80", _fmt3(p80_cn)],
        ["p90", _fmt3(p90_cn)],
        ["p95", _fmt3(p95_cn)],
        ["p99", _fmt3(p99_cn)],
    ]
    lines += [md_table(["Trigger", "SST anomalia (C)"], trig_rows), ""]

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
