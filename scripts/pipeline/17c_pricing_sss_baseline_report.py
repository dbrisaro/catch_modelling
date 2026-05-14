"""
Report - Baseline catch and SSS trigger percentiles by region

Outputs:
  OUTPUTS/report_baseline_sss.md

For each region (Norte, Centro Norte, Centro Sur, Centro):
  - Mean seasonal catch T1 and T2 per year (from catch_summary_by_region.csv)
  - SSS anomaly seasonal distribution (spatial mean over fishing polygon,
    2012-2024, climatology 2015-2024) -> p20, p10, p05, p01

SSS trigger logic is inverted relative to SST:
  Low SSS (fresh water, El Nino) -> bad for anchoveta -> payout
  Percentiles reported: p20, p10, p05, p01 (low-side tail).

SSS is computed only within the fishing corridor:
  For each 1-degree lat band in the region, the fishing polygon spans the
  5th-95th percentile of observed cala longitudes (same method as steps 15b, 13b).
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.path import Path as MplPath

from config import FEATURES, OUTPUTS, SSS_YEARS

# -- regions ------------------------------------------------------------------
REGIONS = [
    ("Norte",        -7.1,  None),
    ("Centro Norte", -11.0, -7.1),
    ("Centro Sur",   -15.8, -11.0),
    ("Centro",       -15.8, -7.1),
]

CLIM_YEARS   = list(range(2015, 2025))   # reference period for percentiles
LON_W, LON_E = -83.0, -74.0

T1_MONTHS = {4, 5, 6, 7}   # Apr - Jul
T2_MONTHS = {11, 12}        # Nov - Dic

T1_N_MONTHS = 4
T2_N_MONTHS = 2

OUTPUT_FILE  = OUTPUTS / "report_baseline_sss.md"


# -- fishing polygon ----------------------------------------------------------
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
    actual_lat_min = df["lat"].min()
    actual_lat_max = df["lat"].max()
    band_edges = np.arange(int(np.floor(actual_lat_min)), int(np.ceil(actual_lat_max)), 1.0)
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
    west_lons = np.array(west_lons)
    east_lons = np.array(east_lons)
    lat_full = np.concatenate([[actual_lat_min], valid_lats, [actual_lat_max]])
    west_full = np.concatenate([[west_lons[0]], west_lons, [west_lons[-1]]])
    east_full = np.concatenate([[east_lons[0]], east_lons, [east_lons[-1]]])
    poly_lons = np.concatenate([west_full, east_full[::-1], [west_full[0]]])
    poly_lats = np.concatenate([lat_full, lat_full[::-1], [lat_full[0]]])
    verts = list(zip(poly_lons, poly_lats))
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
    return MplPath(verts, codes), actual_lat_min, actual_lat_max


# -- SSS loading per region ---------------------------------------------------
def load_sss_weekly(lat_min, lat_max):
    """
    Load SSS anomaly weekly series for a region, masked to fishing polygon.
    Returns a pd.Series indexed by date (all available years from SSS_YEARS).
    """
    result = build_fishing_polygon(lat_min, lat_max)
    if result is None:
        return pd.Series(dtype=float)
    polygon, actual_lat_min, actual_lat_max = result
    lat_lo = actual_lat_min - 0.1
    lat_hi = actual_lat_max + 0.1

    parts = []
    for yr in SSS_YEARS:
        f = FEATURES / f"sss_anomaly_weekly_{yr}.nc"
        if not f.exists():
            continue
        ds = xr.open_dataset(f)
        lat_v = ds["lat"].values
        lon_v = ds["lon"].values
        lat_m = (lat_v > lat_lo) & (lat_v <= lat_hi)
        lon_m = (lon_v >= LON_W) & (lon_v <= LON_E)
        lat_s = lat_v[lat_m]
        lon_s = lon_v[lon_m]
        G_lon, G_lat = np.meshgrid(lon_s, lat_s)
        pts = np.column_stack([G_lon.ravel(), G_lat.ravel()])
        mask_2d = polygon.contains_points(pts).reshape(lat_s.size, lon_s.size)
        mask_da = xr.DataArray(mask_2d, dims=["lat", "lon"],
                               coords={"lat": lat_s, "lon": lon_s})
        s = (ds["sss_anomaly"].isel(lat=lat_m, lon=lon_m)
             .where(mask_da).mean(dim=["lat", "lon"]).to_series())
        s.index = pd.to_datetime(s.index)
        ds.close()
        parts.append(s.dropna())

    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts).sort_index().dropna()


def compute_sss_series(region_name, lat_min, lat_max):
    """
    Returns (all_vals, monthly_vals, seasonal_vals) where each is a pd.Series
    or list of floats. Filters to CLIM_YEARS for percentile computation.
    seasonal_vals: list of seasonal means (T1 and T2 pooled, mean of monthly means).
    """
    weekly = load_sss_weekly(lat_min, lat_max)
    if weekly.empty:
        print(f"  {region_name}: no SSS data found")
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    print(f"  {region_name}: {len(weekly):,} weekly obs  "
          f"{weekly.index[0].date()} - {weekly.index[-1].date()}")

    # Resample to monthly means
    monthly = weekly.resample("MS").mean().dropna()

    # Filter to climatology reference years
    clim_weekly  = weekly[weekly.index.year.isin(CLIM_YEARS)]
    clim_monthly = monthly[monthly.index.year.isin(CLIM_YEARS)]

    # Seasonal means: mean of monthly means for T1 and T2 per year
    seasonal_vals = []
    for yr in CLIM_YEARS:
        yr_mon = clim_monthly[clim_monthly.index.year == yr]
        t1_mon = yr_mon[yr_mon.index.month.isin(T1_MONTHS)]
        t2_mon = yr_mon[yr_mon.index.month.isin(T2_MONTHS)]
        if len(t1_mon) == T1_N_MONTHS:
            seasonal_vals.append(float(t1_mon.mean()))
        if len(t2_mon) == T2_N_MONTHS:
            seasonal_vals.append(float(t2_mon.mean()))

    return clim_weekly, clim_monthly, pd.Series(seasonal_vals)


# -- catch table --------------------------------------------------------------
def load_catch():
    df = pd.read_csv(OUTPUTS / "catch_summary_by_region.csv")
    return df


# -- markdown helpers ---------------------------------------------------------
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


# -- main ---------------------------------------------------------------------
def main():
    catch_df = load_catch()

    lines = [
        "# Baseline Capture and SSS Trigger Report",
        "",
        f"Climatologia SSS: 2015-2024 (OISSS).  "
        f"SSS calculada dentro del corredor de pesca (p5-p95 longitud por banda de 1 grado lat).  ",
        f"Trigger invertido: SSS baja (agua fresca, El Nino) -> pago.  "
        f"Percentiles reportados: cola baja (p20, p10, p05, p01).",
        "",
    ]

    # Compute SSS percentiles per region
    sss_weekly_pcts   = {}
    sss_monthly_pcts  = {}
    sss_seasonal_pcts = {}
    all_weekly_pooled   = []
    all_monthly_pooled  = []
    all_seasonal_pooled = []

    for reg_name, lat_min, lat_max in REGIONS:
        print(f"\nSSS {reg_name}...")
        w_vals, m_vals, s_vals = compute_sss_series(reg_name, lat_min, lat_max)

        def _pcts_low(v):
            if len(v) == 0:
                return (None, None, None, None)
            return tuple(float(np.percentile(v, p)) for p in (20, 10, 5, 1))

        sss_weekly_pcts[reg_name]   = _pcts_low(w_vals)
        sss_monthly_pcts[reg_name]  = _pcts_low(m_vals)
        sss_seasonal_pcts[reg_name] = _pcts_low(s_vals)

        if reg_name != "Centro":
            if len(w_vals) > 0:
                all_weekly_pooled.append(w_vals)
            if len(m_vals) > 0:
                all_monthly_pooled.append(m_vals)
            if len(s_vals) > 0:
                all_seasonal_pooled.append(s_vals)

    def _pool_pcts_low(series_list):
        if not series_list:
            return (None, None, None, None)
        v = pd.concat([pd.Series(s) if not isinstance(s, pd.Series) else s
                       for s in series_list])
        return tuple(float(np.percentile(v, p)) for p in (20, 10, 5, 1))

    sss_total_weekly   = _pool_pcts_low(all_weekly_pooled)
    sss_total_monthly  = _pool_pcts_low(all_monthly_pooled)
    sss_total_seasonal = _pool_pcts_low(all_seasonal_pooled)

    def _fmt3(v):
        return f"{v:.3f}" if v is not None else "-"

    # ── Section 1: catch summary per region with SSS percentiles (weekly) ─────
    lines += ["## 1. Captura media por region (tm, 2015-2025)", ""]
    lines += [
        "Captura media mensual T1 (Abr-Jul) y T2 (Nov-Dic) a partir de catch_summary_by_region.csv.  "
        "Percentiles de SSS anomalia calculados sobre valores semanales dentro del corredor de pesca, "
        "periodo 2015-2024.",
        "",
    ]

    headers = [
        "Region",
        "Media mensual T1 (tm)", "Media mensual T2 (tm)", "Media anual (tm)",
        "p20 SSS (PSU)", "p10 SSS (PSU)", "p05 SSS (PSU)", "p01 SSS (PSU)",
    ]
    fmts = ["str", "int", "int", "int", "str", "str", "str", "str"]
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
        m1 = float(t1.mean()) if len(t1) else 0.0
        m2 = float(t2.mean()) if len(t2) else 0.0
        p20, p10, p05, p01 = sss_weekly_pcts.get(reg_name, (None, None, None, None))
        rows.append([reg_name, m1 / T1_N_MONTHS, m2 / T2_N_MONTHS, m1 + m2,
                     _fmt3(p20), _fmt3(p10), _fmt3(p05), _fmt3(p01)])
        total_t1 += m1
        total_t2 += m2

    rows.append([
        "**Total**",
        total_t1 / T1_N_MONTHS, total_t2 / T2_N_MONTHS, total_t1 + total_t2,
        _fmt3(sss_total_weekly[0]), _fmt3(sss_total_weekly[1]),
        _fmt3(sss_total_weekly[2]), _fmt3(sss_total_weekly[3]),
    ])
    lines += [md_table(headers, rows, fmts), ""]

    # ── Section 2: SSS monthly percentiles ────────────────────────────────────
    lines += ["## 2. Percentiles de anomalia SSS mensual por region (2015-2024)", ""]
    lines += [
        "Misma metodologia pero usando medias mensuales de SSS anomalia "
        "(en vez de valores semanales).",
        "",
    ]

    mon_rows = []
    for reg_name, lat_min, lat_max in REGIONS:
        if reg_name == "Centro":
            continue
        p20, p10, p05, p01 = sss_monthly_pcts.get(reg_name, (None, None, None, None))
        mon_rows.append([reg_name, _fmt3(p20), _fmt3(p10), _fmt3(p05), _fmt3(p01)])
    mon_rows.append([
        "**Total**",
        _fmt3(sss_total_monthly[0]), _fmt3(sss_total_monthly[1]),
        _fmt3(sss_total_monthly[2]), _fmt3(sss_total_monthly[3]),
    ])
    lines += [md_table(
        ["Region", "p20 (PSU)", "p10 (PSU)", "p05 (PSU)", "p01 (PSU)"],
        mon_rows,
    ), ""]

    # ── Section 3: SSS seasonal percentiles (mean of monthly means) ───────────
    lines += ["## 3. Percentiles de anomalia SSS estacional (media de medias mensuales, 2015-2024)", ""]
    lines += [
        "Para cada temporada y anio: media de las medias mensuales "
        "(T1 = media(Abr,May,Jun,Jul), T2 = media(Nov,Dic)). "
        "Pooled T1+T2.",
        "",
    ]

    sea_rows = []
    for reg_name, lat_min, lat_max in REGIONS:
        if reg_name == "Centro":
            continue
        p20, p10, p05, p01 = sss_seasonal_pcts.get(reg_name, (None, None, None, None))
        sea_rows.append([reg_name, _fmt3(p20), _fmt3(p10), _fmt3(p05), _fmt3(p01)])
    sea_rows.append([
        "**Total**",
        _fmt3(sss_total_seasonal[0]), _fmt3(sss_total_seasonal[1]),
        _fmt3(sss_total_seasonal[2]), _fmt3(sss_total_seasonal[3]),
    ])
    lines += [md_table(
        ["Region", "p20 (PSU)", "p10 (PSU)", "p05 (PSU)", "p01 (PSU)"],
        sea_rows,
    ), ""]

    # ── Section 4: trigger thresholds for Centro Norte ────────────────────────
    lines += ["## 4. Triggers de pricing - Centro Norte (2015-2024)", ""]
    lines += [
        "Basado en la distribucion de medias estacionales (T1+T2 pooled) de Centro Norte. "
        "Logica de disparo invertida: SSS baja (agua fresca, El Nino) activa el pago. "
        "Estos valores se usan como lineas de referencia en step 15b/16c.",
        "",
    ]
    p20_cn, p10_cn, p05_cn, p01_cn = sss_seasonal_pcts.get(
        "Centro Norte", (None, None, None, None))
    trig_rows = [
        ["p20", _fmt3(p20_cn)],
        ["p10", _fmt3(p10_cn)],
        ["p05", _fmt3(p05_cn)],
        ["p01", _fmt3(p01_cn)],
    ]
    lines += [md_table(["Trigger", "SSS anomalia (PSU)"], trig_rows), ""]

    OUTPUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
