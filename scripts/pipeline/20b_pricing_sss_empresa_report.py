"""
20b_pricing_sss_empresa_report.py

Reporte de pricing parametrico para una empresa - indicador SSS (salinidad).

Para cada region (Norte, Centro Norte, Centro Sur):
  - Baseline de captura de la empresa (media por temporada, 2015-2025)
  - Serie historica de SSS estacional (media espacial dentro del poligono de
    pesca, 2012-2024, archivos OISSS semanales)
  - Triggers: p20/p10/p05/p01 calculados por region sobre 2015-2024 (periodo fijo)
  - Beta conservador: limite superior IC95% cargado de betas_by_region_sss.csv
  - Activaciones historicas y estimacion de perdidas
  - AAL para escenarios A (entrada p10) y B (entrada p20)

Inputs:
  OUTPUTS/calas_enriched.csv
  OUTPUTS/betas_by_region_sss.csv
  FEATURES/sss_anomaly_weekly_{year}.nc  (2012-2024)
  OUTPUTS/calas_all_data.csv            (para polygon de pesca)

Output:
  PLOTS/report_pricing_sss_{empresa_slug}.md
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.path import Path as MplPath
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FEATURES, OUTPUTS, PLOTS, SSS_YEARS

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

EMPRESA       = "PESQUERA EXALMAR S.A.A."
PRICE_PER_TON = 1_000   # soles / tonelada

CLIM_YEARS = list(range(2015, 2025))   # periodo de referencia para triggers
T1_MONTHS  = {4, 5, 6, 7}
T2_MONTHS  = {11, 12}

LON_W, LON_E = -85.0, -70.0

REGIONS = [
    ("Norte",        -7.1,   None),
    ("Centro Norte", -11.0,  -7.1),
    ("Centro Sur",   -15.8, -11.0),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pct_drop_sss(beta, sss):
    """
    beta > 0 for SSS relationship (lower SSS -> lower catch).
    Loss when sss < 0 (fresher than normal).
    """
    return max(0.0, (1.0 - np.exp(beta * sss))) * 100.0


def ramp_frac(sss, entry, exhaustion):
    """
    Inverted trigger: SSS more negative = more stress = higher payout.
    entry      = p10 (e.g. -0.109) - SSS must be at least this low to trigger
    exhaustion = p01 (e.g. -0.176) - SSS at or below this yields full payout
    """
    if sss >= entry:       # not fresh enough to trigger
        return 0.0
    if sss <= exhaustion:  # fully exhausted
        return 1.0
    return (entry - sss) / (entry - exhaustion)


def md_table(headers, rows):
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines = ["| " + " | ".join(headers) + " |", sep]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def empresa_slug(name):
    return (name.lower()
                .replace(".", "")
                .replace(" ", "_")
                .replace("s_a_a", "saa")
                .replace("s_a_c", "sac")
                .strip("_"))


# ---------------------------------------------------------------------------
# Fishing polygon per region
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

    actual_lat_min = df["lat"].min()
    actual_lat_max = df["lat"].max()

    band_edges = np.arange(int(np.floor(actual_lat_min)),
                           int(np.ceil(actual_lat_max)), 1.0)
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

    lat_full  = np.concatenate([[actual_lat_min], valid_lats, [actual_lat_max]])
    west_full = np.concatenate([[west_lons[0]], west_lons, [west_lons[-1]]])
    east_full = np.concatenate([[east_lons[0]], east_lons, [east_lons[-1]]])

    poly_lons = np.concatenate([west_full, east_full[::-1], [west_full[0]]])
    poly_lats = np.concatenate([lat_full,  lat_full[::-1],  [lat_full[0]]])

    verts = list(zip(poly_lons, poly_lats))
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
    return MplPath(verts, codes), actual_lat_min, actual_lat_max


# ---------------------------------------------------------------------------
# SSS seasonal means per region
# ---------------------------------------------------------------------------

def load_sss_seasonal(reg_name, lat_min, lat_max, years=None):
    """
    Returns DataFrame: year, tipo (T1/T2), sss
    SSS = spatial mean inside fishing polygon, seasonal mean of monthly means.
    Weekly files are resampled to monthly before seasonal aggregation.
    years: list of years to load (default: SSS_YEARS)
    """
    if years is None:
        years = SSS_YEARS

    result = build_fishing_polygon(lat_min, lat_max)
    if result is None:
        print(f"  {reg_name}: poligono de pesca no disponible")
        return pd.DataFrame(columns=["year", "tipo", "sss"])

    polygon, actual_lat_min, actual_lat_max = result
    reg_lat_min = actual_lat_min - 0.1
    reg_lat_max = actual_lat_max + 0.1

    parts = []
    for yr in years:
        f = FEATURES / f"sss_anomaly_weekly_{yr}.nc"
        if not f.exists():
            continue
        ds = xr.open_dataset(f)
        lat_v = ds["lat"].values
        lon_v = ds["lon"].values
        lat_m = (lat_v > reg_lat_min) & (lat_v <= reg_lat_max)
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

    if not parts:
        return pd.DataFrame(columns=["year", "tipo", "sss"])

    weekly  = pd.concat(parts).sort_index()
    monthly = weekly.resample("ME").mean()

    rows = []
    for yr in years:
        yr_mon = monthly[monthly.index.year == yr]
        t1_mon = yr_mon[yr_mon.index.month.isin(T1_MONTHS)]
        t2_mon = yr_mon[yr_mon.index.month.isin(T2_MONTHS)]
        if len(t1_mon) == 4:
            rows.append({"year": yr, "tipo": "T1", "sss": float(t1_mon.mean())})
        if len(t2_mon) == 2:
            rows.append({"year": yr, "tipo": "T2", "sss": float(t2_mon.mean())})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Company catch baselines
# ---------------------------------------------------------------------------

def load_empresa_baselines(empresa_name):
    """
    Returns dict: {region_name: {T1: mean_tm, T2: mean_tm}}
    """
    df = pd.read_csv(OUTPUTS / "calas_enriched.csv", low_memory=False)
    df = df[(df["company"] == empresa_name) & (df["catch_tm"] > 0)].copy()

    baselines = {}
    for reg_name, lat_min, lat_max in REGIONS:
        sub = df[df["lat"] > lat_min].copy()
        if lat_max is not None:
            sub = sub[sub["lat"] <= lat_max]

        agg = sub.groupby("season")["catch_tm"].sum().reset_index()
        agg["tipo"] = agg["season"].apply(
            lambda s: "T1" if "1ra" in s else "T2"
        )
        t1_mean = agg[agg["tipo"] == "T1"]["catch_tm"].mean()
        t2_mean = agg[agg["tipo"] == "T2"]["catch_tm"].mean()

        baselines[reg_name] = {
            "T1": float(t1_mean) if not np.isnan(t1_mean) else 0.0,
            "T2": float(t2_mean) if not np.isnan(t2_mean) else 0.0,
        }

    return baselines


def load_empresa_annual_max(empresa_name):
    """
    Returns (max_tm, max_year) using calas_all_data (full historical record).
    Matches empresa name as substring to handle slight naming variants.
    """
    df = pd.read_csv(OUTPUTS / "calas_all_data.csv", low_memory=False)
    df["declarado_tm"] = pd.to_numeric(df["declarado_tm"], errors="coerce")
    df["fecha_cala"]   = pd.to_datetime(df["fecha_cala"], errors="coerce")
    df["anio"]         = df["fecha_cala"].dt.year

    slug = empresa_name.lower().split()[1] if len(empresa_name.split()) > 1 else empresa_name.lower()
    mask = df["empresa"].str.lower().str.contains(slug, na=False)
    ex   = df[mask & (df["declarado_tm"] > 0)].copy()

    # exclude current incomplete year
    current_year = pd.Timestamp.now().year
    ex = ex[ex["anio"] < current_year]

    anual = ex.groupby("anio")["declarado_tm"].sum()
    if anual.empty:
        return None, None
    max_year = int(anual.idxmax())
    max_tm   = float(anual.max())
    return max_tm, max_year


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # -- load betas --
    betas_df = pd.read_csv(OUTPUTS / "betas_by_region_sss.csv")
    betas = {r["region"]: r for r in betas_df.to_dict("records")}

    # -- load company baselines --
    print(f"\nCargando baselines de {EMPRESA}...")
    baselines = load_empresa_baselines(EMPRESA)
    for reg_name, v in baselines.items():
        anual = v["T1"] + v["T2"]
        print(f"  {reg_name:15s}: T1={v['T1']:>10,.0f}  T2={v['T2']:>10,.0f}  Anual={anual:>10,.0f}")

    max_tm, max_year = load_empresa_annual_max(EMPRESA)
    if max_tm:
        print(f"\n  Maximo historico: {max_tm:,.0f} tm en {max_year}")

    # -- load SSS per region and compute triggers on CLIM_YEARS --
    sss_by_region      = {}
    triggers_by_region = {}
    seas_clim_by_region = {}
    for reg_name, lat_min, lat_max in REGIONS:
        print(f"\nCargando SSS {reg_name}...")
        seas_full = load_sss_seasonal(reg_name, lat_min, lat_max, years=SSS_YEARS)
        sss_by_region[reg_name] = seas_full

        if not seas_full.empty:
            seas_clim = seas_full[seas_full["year"].isin(CLIM_YEARS)]
            seas_clim_by_region[reg_name] = seas_clim
            if not seas_clim.empty:
                rp20 = float(np.percentile(seas_clim["sss"], 20))
                rp10 = float(np.percentile(seas_clim["sss"], 10))
                rp05 = float(np.percentile(seas_clim["sss"], 5))
                rp01 = float(np.percentile(seas_clim["sss"], 1))
                triggers_by_region[reg_name] = {
                    "p20": rp20, "p10": rp10, "p05": rp05, "p01": rp01
                }
                print(f"  {len(seas_full)} temporadas totales ({len(seas_clim)} para triggers). "
                      f"p20={rp20:.3f}  p10={rp10:.3f}  p05={rp05:.3f}  p01={rp01:.3f}")

    # -- build document --
    slug = empresa_slug(EMPRESA)
    out  = PLOTS / f"report_pricing_sss_{slug}.md"
    lines = []

    empresa_display = EMPRESA.replace("S.A.A.", "S.A.A").replace("S.A.C.", "S.A.C")

    lines += [
        f"# Seguro parametrico de anchoveta (SSS) - {empresa_display}",
        "",
        f"Empresa: **{empresa_display}**  ",
        f"Triggers: calculados por region sobre SSS estacional 2015-2024 (OISSS). "
        f"Betas: limite superior IC95% por region (nivel empresa x temporada).  ",
        f"Precio de referencia: S/. {PRICE_PER_TON:,}/tm.",
        "",
    ]

    # -----------------------------------------------------------------------
    # 1. Baselines por region
    # -----------------------------------------------------------------------
    lines += ["## 1. Baseline de captura por region", ""]

    total_t1 = sum(v["T1"] for v in baselines.values())
    total_t2 = sum(v["T2"] for v in baselines.values())
    total_anual = total_t1 + total_t2

    bl_rows = []
    for reg_name, v in baselines.items():
        anual = v["T1"] + v["T2"]
        bl_rows.append([
            reg_name,
            f"{v['T1']:,.0f}",
            f"{v['T2']:,.0f}",
            f"{anual:,.0f}",
            f"{anual * PRICE_PER_TON / 1e6:.1f}",
        ])
    bl_rows.append([
        "**Total**",
        f"**{total_t1:,.0f}**",
        f"**{total_t2:,.0f}**",
        f"**{total_anual:,.0f}**",
        f"**{total_anual * PRICE_PER_TON / 1e6:.1f}**",
    ])
    max_note = ""
    if max_tm and max_year:
        max_note = f"Maximo historico: **{max_tm:,.0f} tm ({max_year})**. "

    lines += [
        md_table(
            ["Region", "T1 media (tm)", "T2 media (tm)", "Anual media (tm)", "Valor asegurado (M soles)"],
            bl_rows,
        ),
        "",
        f"Periodo de referencia: 2015-2025. Precio: S/. {PRICE_PER_TON:,}/tm. {max_note}",
        "",
    ]

    # -----------------------------------------------------------------------
    # 2. Betas por region
    # -----------------------------------------------------------------------
    lines += ["## 2. Sensibilidad SSS por region (beta conservador)", ""]
    lines += [
        "Beta conservador = limite superior IC95% (el mas cercano a cero). "
        "Modelo: log(captura empresa x temporada) ~ anomalia SSS.",
        "",
    ]

    beta_rows = []
    for reg_name, _, __ in REGIONS:
        b = betas.get(reg_name)
        if b is None:
            continue
        t = triggers_by_region.get(reg_name, {})
        beta_rows.append([
            reg_name,
            f"{b['beta']:+.3f}",
            f"[{b['ci_lower']:+.3f}, {b['ci_upper']:+.3f}]",
            f"**{b['beta_conserv']:+.3f}**",
            f"{pct_drop_sss(b['beta_conserv'], -1.0):.1f}",
            f"{pct_drop_sss(b['beta_conserv'], t.get('p10', 0)):.1f}" if t else "-",
            f"{pct_drop_sss(b['beta_conserv'], t.get('p01', 0)):.1f}" if t else "-",
        ])
    lines += [
        md_table(
            ["Region", "Beta", "IC 95%", "Beta conservador",
             "Caida -1.0 psu (%)", "Caida p10 regional (%)", "Caida p01 regional (%)"],
            beta_rows,
        ),
        "",
    ]

    # -----------------------------------------------------------------------
    # 3. Por region: triggers, activaciones, AAL
    # -----------------------------------------------------------------------
    lines += ["## 3. Analisis por region", ""]

    # tabla resumen de triggers por region
    trig_summary_rows = []
    for reg_name, _, __ in REGIONS:
        t = triggers_by_region.get(reg_name)
        if t:
            trig_summary_rows.append([
                reg_name,
                f"{t['p20']:.3f}",
                f"{t['p10']:.3f}",
                f"{t['p05']:.3f}",
                f"{t['p01']:.3f}",
            ])
    lines += [
        "Triggers calculados sobre la distribucion de SSS estacional de cada region (2015-2024). "
        "Valores negativos indican anomalia de salinidad baja (agua mas fresca = estres para anchoveta):",
        "",
        md_table(
            ["Region", "p20 (psu)", "p10 (psu)", "p05 (psu)", "p01 (psu)"],
            trig_summary_rows,
        ),
        "",
    ]

    total_aal_A = 0.0
    total_aal_B = 0.0
    total_max_payout = 0.0

    def _vez(n): return "vez" if n == 1 else "veces"

    for reg_name, lat_min, lat_max in REGIONS:
        b = betas.get(reg_name)
        if b is None:
            continue
        beta_c = b["beta_conserv"]
        bl     = baselines[reg_name]
        seas   = sss_by_region.get(reg_name, pd.DataFrame())
        t      = triggers_by_region.get(reg_name)
        max_payout = (bl["T1"] + bl["T2"]) * PRICE_PER_TON
        total_max_payout += max_payout

        lines += [f"### 3.{REGIONS.index((reg_name, lat_min, lat_max))+1}. {reg_name}", ""]

        if seas.empty or t is None:
            lines += ["_No hay datos SSS disponibles para esta region._", ""]
            continue

        p20 = t["p20"]
        p10 = t["p10"]
        p05 = t["p05"]
        p01 = t["p01"]

        # activations (low SSS = stress = trigger)
        events_p10 = seas[seas["sss"] <= p10].sort_values("sss", ascending=True)
        events_p20 = seas[seas["sss"] <= p20]
        events_p05 = seas[seas["sss"] <= p05]
        events_p01 = seas[seas["sss"] <= p01]
        n_seas     = len(seas)
        n_years    = len(seas["year"].unique())

        lines += [
            f"Beta conservador: **{beta_c:+.3f}**  |  "
            f"Valor asegurado maximo: **{max_payout/1e6:.1f} M soles**  |  "
            f"Triggers: p20={p20:.3f} psu  p10={p10:.3f} psu  p05={p05:.3f} psu  p01={p01:.3f} psu",
            "",
            f"Activaciones historicas (2012-2024, {n_seas} temporadas en {n_years} anios):",
            "",
            f"- p20 ({p20:.3f} psu): activado **{len(events_p20)} {_vez(len(events_p20))}** "
            f"({len(events_p20)/n_seas*100:.0f}% de las temporadas)",
            f"- p10 ({p10:.3f} psu): activado **{len(events_p10)} {_vez(len(events_p10))}** "
            f"({len(events_p10)/n_seas*100:.0f}% de las temporadas)",
            f"- p05 ({p05:.3f} psu): activado **{len(events_p05)} {_vez(len(events_p05))}** "
            f"({len(events_p05)/n_seas*100:.0f}% de las temporadas)",
            f"- p01 ({p01:.3f} psu): activado **{len(events_p01)} {_vez(len(events_p01))}** "
            f"({len(events_p01)/n_seas*100:.0f}% de las temporadas)",
            "",
        ]

        # table of p10 and below events (sorted most anomalous first)
        events_display = seas[seas["sss"] <= p20].sort_values("sss", ascending=True)
        if not events_display.empty:
            trig_rows = []
            for _, row in events_display.iterrows():
                sss_val   = row["sss"]
                base_s    = bl["T1"] if row["tipo"] == "T1" else bl["T2"]
                loss_frac = pct_drop_sss(beta_c, sss_val) / 100.0
                loss_tm   = base_s * loss_frac
                loss_sol  = loss_tm * PRICE_PER_TON / 1e6

                if sss_val <= p01:
                    tag = "p20 p10 p05 p01"
                elif sss_val <= p05:
                    tag = "p20 p10 p05"
                elif sss_val <= p10:
                    tag = "p20 p10"
                else:
                    tag = "p20"

                trig_rows.append([
                    f"{int(row['year'])}-{row['tipo']}",
                    f"{sss_val:.3f}",
                    tag,
                    f"{loss_frac*100:.0f}%",
                    f"{loss_tm:,.0f}",
                    f"{loss_sol:.2f}",
                ])

            lines += [
                md_table(
                    ["Temporada", "SSS (psu)", "Gatillos activos",
                     "Caida captura (%)", "Caida captura (tm)", "Perdida estimada (M soles)"],
                    trig_rows,
                ),
                "",
                f"_Beta conservador: {beta_c:+.3f}. "
                f"Baseline: T1={bl['T1']:,.0f} tm, T2={bl['T2']:,.0f} tm._",
                "",
            ]

        # AAL scenarios
        years_t1 = {r["year"]: r["sss"] for _, r in seas[seas["tipo"]=="T1"].iterrows()}
        years_t2 = {r["year"]: r["sss"] for _, r in seas[seas["tipo"]=="T2"].iterrows()}
        common   = sorted(set(years_t1) & set(years_t2))

        lines += ["**Escenarios de cobertura (ramp lineal):**", ""]

        aal_vals = {}
        for esc, entry_lbl, entry_val in [("A", "p10", p10), ("B", "p20", p20)]:
            payouts = []
            n_trig  = 0
            for yr in common:
                pay_t1 = max_payout * ramp_frac(years_t1[yr], entry_val, p01)
                pay_t2 = max_payout * ramp_frac(years_t2[yr], entry_val, p01)
                ann    = min(pay_t1 + pay_t2, max_payout)
                payouts.append(ann)
                if ann > 0:
                    n_trig += 1
            aal     = float(np.mean(payouts))
            aal_pct = aal / max_payout * 100 if max_payout > 0 else 0.0
            aal_vals[esc] = aal

            lines += [
                f"Escenario {esc}: entrada {entry_lbl} ({entry_val:.3f} psu) - salida p01 ({p01:.3f} psu)",
                "",
                md_table(
                    ["Metrica", "Valor"],
                    [
                        ["AAL historico (2012-2024)", f"**{aal/1e6:.3f} M soles**"],
                        ["AAL como % del valor asegurado", f"**{aal_pct:.2f}%**"],
                        ["Tasa de prima de riesgo (RoL)", f"**{aal_pct:.2f}%**"],
                        ["Anos con pago", f"{n_trig} de {len(common)}"],
                        ["Anos sin pago", f"{len(common)-n_trig} de {len(common)}"],
                    ],
                ),
                "",
            ]

        total_aal_A += aal_vals.get("A", 0.0)
        total_aal_B += aal_vals.get("B", 0.0)

    # -----------------------------------------------------------------------
    # 4. Resumen total
    # -----------------------------------------------------------------------
    lines += ["## 4. Resumen consolidado", ""]

    total_aal_A_pct = total_aal_A / total_max_payout * 100 if total_max_payout > 0 else 0.0
    total_aal_B_pct = total_aal_B / total_max_payout * 100 if total_max_payout > 0 else 0.0

    lines += [
        md_table(
            ["Escenario", "Valor asegurado maximo", "AAL historico", "Tasa de prima (RoL)"],
            [
                ["A (entrada p10)",
                 f"{total_max_payout/1e6:.1f} M soles",
                 f"{total_aal_A/1e6:.3f} M soles",
                 f"{total_aal_A_pct:.2f}%"],
                ["B (entrada p20)",
                 f"{total_max_payout/1e6:.1f} M soles",
                 f"{total_aal_B/1e6:.3f} M soles",
                 f"{total_aal_B_pct:.2f}%"],
            ],
        ),
        "",
        f"_Suma de las tres regiones. "
        f"Valor asegurado = captura media anual empresa x S/. {PRICE_PER_TON:,}/tm._",
        "",
    ]

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
