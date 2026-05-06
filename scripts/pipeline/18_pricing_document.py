"""
18_pricing_document.py

Genera un documento de referencia para el diseno de seguro parametrico
de anchoveta. Calcula desde los datos:

  - Produccion nacional media y maximos anuales (Todo el mar, 2015-2025)
  - Caida de captura estimada por anomalia SST (region total, beta conservador)
  - Propuesta preliminar Centro Norte:
      * Baseline de captura
      * Gatillos SST p90 / p95 / p99 (2005-2024, medias estacionales)
      * Activaciones historicas con estimacion de perdida
      * AAL para dos escenarios de ramp

Betas: limite inferior del IC95% (escenario conservador, figura 16).
  All (Norte+Centro): -0.586
  Centro Norte:       -0.518

Output:
  PLOTS/report_pricing_document.md
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.path import Path

from config import FEATURES, OUTPUTS, PLOTS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BETA_ALL = -0.586      # All (Norte+Centro), CI lower bound [-1.013, -0.586]
BETA_CN  = -0.518      # Centro Norte,        CI lower bound [-0.915, -0.518]

PRICE_PER_TON = 1_000  # soles / tonelada

CLIM_YEARS  = list(range(2005, 2025))
T1_MONTHS   = [4, 5, 6, 7]
T2_MONTHS   = [11, 12]
T1_N_MONTHS = 4
T2_N_MONTHS = 2

CN_LAT_MIN, CN_LAT_MAX = -11.0, -7.1
LON_W, LON_E            = -82.0, -74.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pct_drop(beta, sst):
    """% caida de captura relativa a condicion normal (SST=0)."""
    return (1.0 - np.exp(beta * sst)) * 100.0


def ramp_frac(sst, entry, exhaustion):
    if sst <= entry:
        return 0.0
    if sst >= exhaustion:
        return 1.0
    return (sst - entry) / (exhaustion - entry)


def fmt_M(v):
    return f"{v/1e6:.3f}"


def fmt_pct(v):
    return f"{v:.1f}%"


def md_table(headers, rows):
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines = ["| " + " | ".join(headers) + " |", sep]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. SST fishing polygon and seasonal means for Centro Norte
# ---------------------------------------------------------------------------

def build_cn_polygon():
    df = (pd.read_csv(OUTPUTS / "calas_all_data.csv",
                      usecols=["latitud", "longitud"], low_memory=False)
          .rename(columns={"latitud": "lat", "longitud": "lon"})
          .dropna())
    df = df[(df["lat"] >= CN_LAT_MIN) & (df["lat"] <= CN_LAT_MAX)]

    band_edges = np.arange(int(np.floor(CN_LAT_MIN)), int(np.ceil(CN_LAT_MAX)), 1.0)
    west_lons, east_lons, valid_lats = [], [], []
    for lo in band_edges:
        band = df[(df["lat"] >= lo) & (df["lat"] < lo + 1.0)]
        if len(band) < 20:
            continue
        west_lons.append(np.percentile(band["lon"], 5))
        east_lons.append(np.percentile(band["lon"], 95))
        valid_lats.append(lo + 0.5)

    valid_lats = np.array(valid_lats)
    west_lons  = np.array(west_lons)
    east_lons  = np.array(east_lons)
    lat_full   = np.concatenate([[CN_LAT_MIN], valid_lats, [CN_LAT_MAX]])
    west_full  = np.concatenate([[west_lons[0]], west_lons, [west_lons[-1]]])
    east_full  = np.concatenate([[east_lons[0]], east_lons, [east_lons[-1]]])
    poly_lons  = np.concatenate([west_full, east_full[::-1], [west_full[0]]])
    poly_lats  = np.concatenate([lat_full,  lat_full[::-1],  [lat_full[0]]])
    verts = list(zip(poly_lons, poly_lats))
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
    return Path(verts, codes)


def load_cn_sst_seasonal():
    """Seasonal SST means for Centro Norte, 2005-2024.
    Returns DataFrame with columns: year, tipo (T1/T2), sst."""
    polygon = build_cn_polygon()
    parts = []
    for yr in CLIM_YEARS:
        f = FEATURES / f"sst_anomaly_daily_{yr}.nc"
        if not f.exists():
            continue
        ds = xr.open_dataset(f)
        lat_v = ds["lat"].values
        lon_v = ds["lon"].values
        lat_m = (lat_v >= CN_LAT_MIN) & (lat_v <= CN_LAT_MAX)
        lon_m = (lon_v >= LON_W) & (lon_v <= LON_E)
        lat_s = lat_v[lat_m]
        lon_s = lon_v[lon_m]
        G_lon, G_lat = np.meshgrid(lon_s, lat_s)
        pts     = np.column_stack([G_lon.ravel(), G_lat.ravel()])
        mask_2d = polygon.contains_points(pts).reshape(lat_s.size, lon_s.size)
        mask_da = xr.DataArray(mask_2d, dims=["lat", "lon"],
                               coords={"lat": lat_s, "lon": lon_s})
        s = (ds["sst_anomaly"].isel(lat=lat_m, lon=lon_m)
             .where(mask_da).mean(dim=["lat", "lon"]).to_series())
        s.index = pd.to_datetime(s.index)
        ds.close()
        parts.append(s.dropna())

    daily   = pd.concat(parts).sort_index()
    monthly = daily.resample("ME").mean()

    rows = []
    for yr in CLIM_YEARS:
        yr_mon = monthly[monthly.index.year == yr]
        t1_mon = yr_mon[yr_mon.index.month.isin(T1_MONTHS)]
        t2_mon = yr_mon[yr_mon.index.month.isin(T2_MONTHS)]
        if len(t1_mon) == T1_N_MONTHS:
            rows.append({"year": yr, "tipo": "T1", "sst": float(t1_mon.mean())})
        if len(t2_mon) == T2_N_MONTHS:
            rows.append({"year": yr, "tipo": "T2", "sst": float(t2_mon.mean())})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Catch data
# ---------------------------------------------------------------------------

def load_production():
    """Returns annual national totals and Centro Norte seasonal baselines."""
    df = pd.read_csv(OUTPUTS / "catch_summary_by_region.csv")

    # --- National annual totals from 'Todo el mar' ---
    national = df[df["region"] == "Todo el mar"].copy()
    annual = (national.groupby("anio")["captura_tm"].sum()
              .reset_index().rename(columns={"anio": "year", "captura_tm": "total_tm"}))

    # --- Centro Norte baselines ---
    cn = df[df["region"] == "Centro Norte"].copy()
    cn_t1_mean = cn[cn["temporada"] == "1ra"]["captura_tm"].mean()
    cn_t2_mean = cn[cn["temporada"] == "2da"]["captura_tm"].mean()

    return annual, cn_t1_mean, cn_t2_mean


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------

def main():
    out = PLOTS / "report_pricing_document.md"

    print("Loading catch data...")
    annual, baseline_t1, baseline_t2 = load_production()
    baseline_annual = baseline_t1 + baseline_t2

    print("Loading Centro Norte SST seasonal means (2005-2024)...")
    seas = load_cn_sst_seasonal()
    all_sst = seas["sst"].values
    p90 = float(np.percentile(all_sst, 90))
    p95 = float(np.percentile(all_sst, 95))
    p99 = float(np.percentile(all_sst, 99))
    print(f"  p90={p90:.3f}  p95={p95:.3f}  p99={p99:.3f}  N={len(all_sst)}")

    lines = []
    lines += ["# Seguro parametrico de anchoveta - Documento de referencia", ""]

    # -----------------------------------------------------------------------
    # Seccion 1: Produccion nacional
    # -----------------------------------------------------------------------
    lines += ["## 1. Produccion nacional de anchoveta", ""]

    mean_M = annual["total_tm"].mean() / 1e6
    max_M  = annual["total_tm"].max()  / 1e6
    max_yr = int(annual.loc[annual["total_tm"].idxmax(), "year"])

    lines += [
        f"Periodo: {int(annual['year'].min())}-{int(annual['year'].max())} "
        f"(datos de capturas declaradas por temporada).",
        "",
        f"- **Media anual: {mean_M:.2f} M tm**",
        f"- **Maximo anual: {max_M:.2f} M tm ({max_yr})**",
        "",
    ]

    yr_rows = []
    for _, r in annual.sort_values("year").iterrows():
        yr_rows.append([str(int(r["year"])), f"{r['total_tm']/1e6:.3f}"])
    lines += [md_table(["Ano", "Captura total (M tm)"], yr_rows), ""]

    # -----------------------------------------------------------------------
    # Seccion 2: Sensibilidad captura a SST - region total
    # -----------------------------------------------------------------------
    lines += ["## 2. Sensibilidad de la captura a la SST - Region total", ""]
    lines += [
        f"Beta conservador (All Norte+Centro, limite inferior IC95%): **{BETA_ALL}**  ",
        f"Interpretacion: por cada grado Celsius de anomalia positiva de SST, "
        f"la captura cae un {pct_drop(BETA_ALL, 1.0):.1f}% respecto al nivel normal.",
        "",
        md_table(
            ["Anomalia SST (C)", "Caida estimada de captura (%)"],
            [["1.0", f"{pct_drop(BETA_ALL, 1.0):.1f}"],
             ["2.5", f"{pct_drop(BETA_ALL, 2.5):.1f}"]],
        ),
        "",
        "_Caida relativa al nivel esperado con SST anomalia = 0. "
        "Modelo: log(captura empresa x temporada) ~ beta x SST._",
        "",
    ]

    # -----------------------------------------------------------------------
    # Seccion 3: Propuesta preliminar Centro Norte
    # -----------------------------------------------------------------------
    lines += ["## 3. Propuesta preliminar - Region Centro Norte", ""]

    # --- 3.1 Baseline ---
    max_payout_soles = baseline_annual * PRICE_PER_TON
    lines += [
        "### 3.1 Baseline de captura",
        "",
        f"| | Captura media (tm) | Valor referencia (M soles) |",
        f"| --- | --- | --- |",
        f"| T1 (abr-jul) | {baseline_t1:,.0f} | {baseline_t1*PRICE_PER_TON/1e6:.2f} |",
        f"| T2 (nov-dic) | {baseline_t2:,.0f} | {baseline_t2*PRICE_PER_TON/1e6:.2f} |",
        f"| **Anual** | **{baseline_annual:,.0f}** | **{max_payout_soles/1e6:.2f}** |",
        "",
        f"Precio de referencia: 1,000 soles/tm. "
        f"Pago maximo (cobertura total): **{max_payout_soles/1e6:.2f} M soles**.",
        "",
    ]

    # --- 3.2 Gatillos SST ---
    lines += [
        "### 3.2 Gatillos SST (2005-2024, medias estacionales T1+T2 pooled)",
        "",
        md_table(
            ["Percentil", "SST anomalia (C)"],
            [["p90", f"{p90:.3f}"], ["p95", f"{p95:.3f}"], ["p99", f"{p99:.3f}"]],
        ),
        "",
        f"N = {len(all_sst)} temporadas (T1+T2 combinadas, 2005-2024). "
        f"SST calculada como media espacial dentro del poligono de pesca "
        f"(p5-p95 longitudes por banda de 1 grado lat).",
        "",
        f"Beta conservador Centro Norte: **{BETA_CN}**. "
        f"Caida de captura estimada para cada gatillo:",
        "",
        md_table(
            ["Gatillo", "SST anomalia (C)", "Caida estimada de captura (%)"],
            [
                ["p90", f"{p90:.3f}", f"{pct_drop(BETA_CN, p90):.1f}"],
                ["p95", f"{p95:.3f}", f"{pct_drop(BETA_CN, p95):.1f}"],
                ["p99", f"{p99:.3f}", f"{pct_drop(BETA_CN, p99):.1f}"],
            ],
        ),
        "",
    ]

    # --- 3.3 Activaciones historicas ---
    lines += ["### 3.3 Activaciones historicas (2005-2024)", ""]

    events_p95 = seas[seas["sst"] >= p95]
    events_p99 = seas[seas["sst"] >= p99]

    def _vez(n):
        return "vez" if n == 1 else "veces"

    lines += [
        f"- Gatillo **p95 ({p95:.2f}C):** activado **{len(events_p95)} {_vez(len(events_p95))}** "
        f"en 20 anios ({len(events_p95)/20*100:.0f}% de los anos).",
        f"- Gatillo **p99 ({p99:.2f}C):** activado **{len(events_p99)} {_vez(len(events_p99))}** "
        f"en 20 anios ({len(events_p99)/20*100:.0f}% de los anos).",
        "",
    ]

    events_p90 = seas[seas["sst"] >= p90].sort_values("sst", ascending=False)

    trig_rows = []
    for _, row in events_p90.iterrows():
        season_id  = f"{int(row['year'])}-{row['tipo']}"
        sst_val    = row["sst"]
        base_s     = baseline_t1 if row["tipo"] == "T1" else baseline_t2
        loss_frac  = max(0.0, 1.0 - np.exp(BETA_CN * sst_val)) if sst_val > 0 else 0.0
        loss_tm    = base_s * loss_frac
        loss_M_sol = loss_tm * PRICE_PER_TON / 1e6

        if sst_val >= p99:
            tag = "p90 p95 p99"
        elif sst_val >= p95:
            tag = "p90 p95"
        else:
            tag = "p90"

        trig_rows.append([
            season_id, f"{sst_val:.2f}", tag,
            f"{loss_frac*100:.0f}%",
            f"{loss_tm:,.0f}", f"{loss_M_sol:.2f}",
        ])

    lines += [
        md_table(
            ["Temporada", "SST (C)", "Gatillos activos",
             "Caida captura (%)", "Caida captura (tm)", "Perdida estimada (M soles)"],
            trig_rows,
        ),
        "",
        f"_Beta Centro Norte conservador: {BETA_CN}. "
        f"Caida relativa al baseline de cada temporada (T1: {baseline_t1:,.0f} tm, "
        f"T2: {baseline_t2:,.0f} tm)._",
        "",
    ]

    # --- 3.4 AAL escenarios ---
    lines += ["### 3.4 Escenarios de cobertura - Ramp lineal", ""]
    lines += [
        f"Estructura: ramp lineal sobre SST anomalia estacional. "
        f"Pago por temporada = baseline_temporada x frac x 1,000 soles/tm. "
        f"Pago anual maximo = {max_payout_soles/1e6:.2f} M soles.",
        "",
    ]

    years_t1 = dict(zip(seas[seas["tipo"] == "T1"]["year"],
                        seas[seas["tipo"] == "T1"]["sst"]))
    years_t2 = dict(zip(seas[seas["tipo"] == "T2"]["year"],
                        seas[seas["tipo"] == "T2"]["sst"]))
    common_years = sorted(set(years_t1) & set(years_t2))

    for esc_lbl, entry_lbl, entry_val in [("A", "p90", p90), ("B", "p95", p95)]:
        payouts = []
        n_triggered = 0
        for yr in common_years:
            sst_t1 = years_t1[yr]
            sst_t2 = years_t2[yr]
            pay_t1 = baseline_t1 * ramp_frac(sst_t1, entry_val, p99) * PRICE_PER_TON
            pay_t2 = baseline_t2 * ramp_frac(sst_t2, entry_val, p99) * PRICE_PER_TON
            annual_pay = min(pay_t1 + pay_t2, max_payout_soles)
            payouts.append(annual_pay)
            if annual_pay > 0:
                n_triggered += 1

        aal        = float(np.mean(payouts))
        aal_pct    = aal / max_payout_soles * 100
        rol        = aal_pct

        lines += [
            f"**Escenario {esc_lbl}: entrada {entry_lbl} ({entry_val:.2f}C) - salida p99 ({p99:.2f}C)**",
            "",
            f"| Metrica | Valor |",
            f"| --- | --- |",
            f"| AAL historico (2005-2024) | **{aal/1e6:.3f} M soles** |",
            f"| AAL como % del valor asegurado maximo | **{aal_pct:.2f}%** |",
            f"| Tasa de prima de riesgo (RoL) | **{rol:.2f}%** |",
            f"| Anos con pago | {n_triggered} de {len(common_years)} |",
            f"| Anos sin pago | {len(common_years)-n_triggered} de {len(common_years)} |",
            "",
        ]

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
