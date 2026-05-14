"""
24_pricing_sss_reinsurance.py

Analisis de reaseguro SSS - portafolio agregado Norte + Centro Norte + Centro Sur.

Analogo a 21_pricing_reinsurance.py pero usa SSS (salinidad semanal) en lugar de SST.

Diferencias clave respecto al script 21 (SST):
  - SSS: media mensual de valores semanales en ventana T1=[Apr-Jul] y T2=[Nov-Dic]
  - Triggers: percentiles BAJOS de SSS estacional pooled T1+T2, climatologia 2015-2024
  - Trigger direction: invertida. Pago cuando SSS es baja (agua dulce = El Nino)
  - Bootstrap: Normal fit a T1 y T2 por separado, 4000 anos sinteticos
  - Ramp: lineal por temporada, cap en baseline anual regional
  - Agregado: suma regional, cap en programa total

Outputs:
  outputs/21_reinsurance_sss_aep.png
  outputs/21_reinsurance_sss_analysis.md
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from matplotlib.path import Path as MplPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FEATURES, OUTPUTS, PLOTS, SSS_YEARS

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

TOTAL_PROGRAM_TON = None   # None = baseline total agregado

PRICE_USD_TON = 300
LOSS_RATIO    = 0.65

N_BOOTSTRAP = 4000
CLIM_YEARS  = list(range(2015, 2025))

LON_W, LON_E = -83.0, -74.0

T1_MONTHS = [4, 5, 6, 7]    # Abr-Jul
T2_MONTHS = [11, 12]         # Nov-Dic

REGIONS = [
    dict(name="North",         lat_min=-7.1,  lat_max=None,  col="#d6604d"),
    dict(name="North Central", lat_min=-11.0, lat_max=-7.1,  col="#2166ac"),
    dict(name="South Central", lat_min=-15.8, lat_max=-11.0, col="#4dac26"),
]

SCENARIOS = [
    dict(key="p20", label="Entry p20", col="#1B5E20", marker="s"),
    dict(key="p10", label="Entry p10", col="#e07b39", marker="o"),
    dict(key="p05", label="Entry p05", col="#c0392b", marker="^"),
]

# ---------------------------------------------------------------------------
# Fishing polygon  (identico al resto de scripts SSS)
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
    west_lons = np.array(west_lons)
    east_lons = np.array(east_lons)
    lat_full = np.concatenate([[actual_min], valid_lats, [actual_max]])
    west_full = np.concatenate([[west_lons[0]], west_lons, [west_lons[-1]]])
    east_full = np.concatenate([[east_lons[0]], east_lons, [east_lons[-1]]])
    poly_lons = np.concatenate([west_full, east_full[::-1], [west_full[0]]])
    poly_lats = np.concatenate([lat_full, lat_full[::-1], [lat_full[0]]])
    verts = list(zip(poly_lons, poly_lats))
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
    return MplPath(verts, codes), actual_min, actual_max


# ---------------------------------------------------------------------------
# SSS loading
# ---------------------------------------------------------------------------

def load_sss_region(lat_min, lat_max):
    """SSS anomalia semanal espacial-media dentro del poligono de pesca regional."""
    result = build_fishing_polygon(lat_min, lat_max)
    if result is None:
        return pd.Series(dtype=float)
    polygon, actual_min, actual_max = result
    lat_lo = actual_min - 0.1
    lat_hi = actual_max + 0.1

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
        return pd.Series(dtype=float)
    return pd.concat(parts).sort_index().dropna()


# ---------------------------------------------------------------------------
# Seasonal means  (mes-based, no DOY)
# ---------------------------------------------------------------------------

def get_seasonal_means(weekly_series):
    monthly = weekly_series.resample("ME").mean().dropna()
    T1_MONTHS_SET = {4, 5, 6, 7}
    T2_MONTHS_SET = {11, 12}
    years_with_both = []
    t1_vals, t2_vals = [], []
    for yr in sorted(monthly.index.year.unique()):
        yr_mon = monthly[monthly.index.year == yr]
        t1_mon = yr_mon[yr_mon.index.month.isin(T1_MONTHS_SET)]
        t2_mon = yr_mon[yr_mon.index.month.isin(T2_MONTHS_SET)]
        if len(t1_mon) == 4 and len(t2_mon) == 2:
            years_with_both.append(yr)
            t1_vals.append(float(t1_mon.mean()))
            t2_vals.append(float(t2_mon.mean()))
    return np.array(years_with_both), np.array(t1_vals), np.array(t2_vals)


# ---------------------------------------------------------------------------
# Baseline  (identico al script 21)
# ---------------------------------------------------------------------------

_REGION_CSV_NAME = {
    "North":         "Norte",
    "North Central": "Centro Norte",
    "South Central": "Centro Sur",
}

def load_baseline_region(region_name):
    summary = pd.read_csv(OUTPUTS / "catch_summary_by_region.csv")
    csv_name = _REGION_CSV_NAME.get(region_name, region_name)
    sub = summary[summary["region"] == csv_name].copy()

    def tipo(s):
        s = str(s)
        if "1ra" in s or "-I" in s:
            return "T1"
        if "2da" in s or "-II" in s:
            return "T2"
        return "other"

    sub["tipo"] = sub["temporada_key"].apply(tipo)
    t1 = sub[sub["tipo"] == "T1"]["captura_tm"]
    t2 = sub[sub["tipo"] == "T2"]["captura_tm"]
    return float(t1.mean()) if len(t1) else 0.0, float(t2.mean()) if len(t2) else 0.0


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def parametric_bootstrap(t1_obs, t2_obs, n=N_BOOTSTRAP):
    mu1, sd1 = t1_obs.mean(), t1_obs.std(ddof=1)
    mu2, sd2 = t2_obs.mean(), t2_obs.std(ddof=1)
    rng = np.random.default_rng(42)
    return rng.normal(mu1, sd1, n), rng.normal(mu2, sd2, n)


# ---------------------------------------------------------------------------
# Ramp payout - SSS (inverted: pago cuando SSS es baja)
# ---------------------------------------------------------------------------

def ramp_payout_sss(sss_t1, sss_t2, baseline_t1, baseline_t2, entry_sss, exit_sss):
    # entry_sss = p10 (e.g. -0.109), exit_sss = p01 (e.g. -0.176)
    # mas negativo = mayor pago
    annual_baseline = baseline_t1 + baseline_t2
    denom = entry_sss - exit_sss
    if abs(denom) < 1e-9:
        return np.zeros(len(sss_t1))
    frac_t1 = np.clip((entry_sss - sss_t1) / denom, 0.0, 1.0)
    frac_t2 = np.clip((entry_sss - sss_t2) / denom, 0.0, 1.0)
    return np.minimum(annual_baseline * (frac_t1 + frac_t2), annual_baseline)


# ---------------------------------------------------------------------------
# AEP helpers  (identico al script 21)
# ---------------------------------------------------------------------------

def empirical_aep(values):
    sv = np.sort(values)
    n  = len(sv)
    ep = (n + 1 - np.arange(1, n + 1)) / (n + 1)
    return sv, ep


def aep_ci(values, n_chunks=20):
    sv_full, ep_full = empirical_aep(values)
    ep_grid = np.linspace(ep_full.min(), ep_full.max(), 300)
    chunk   = len(values) // n_chunks
    curves  = []
    for i in range(n_chunks):
        sub        = values[i * chunk:(i + 1) * chunk]
        sv_i, ep_i = empirical_aep(sub)
        curves.append(np.interp(ep_grid, ep_i[::-1], sv_i[::-1]))
    median = np.interp(ep_grid, ep_full[::-1], sv_full[::-1])
    ci_lo  = np.percentile(curves,  5, axis=0)
    ci_hi  = np.percentile(curves, 95, axis=0)
    return ep_grid, ci_lo, median, ci_hi


# ---------------------------------------------------------------------------
# AEP plot
# ---------------------------------------------------------------------------

def plot_aep_ramp(scenarios_data, total_baseline, out_path, ann_y=None):
    if ann_y is None:
        ann_y = [0.82, 0.60, 0.38]
    fig, ax = plt.subplots(figsize=(9, 7))
    for sc, ay in zip(scenarios_data, ann_y):
        ep_g, lo_g, med_g, hi_g = aep_ci(sc["pay_synth"])
        rate = sc["aal"] / total_baseline * 100
        ax.fill_betweenx(ep_g[::-1], lo_g[::-1], hi_g[::-1],
                         alpha=0.20, color=sc["col"])
        ax.plot(med_g[::-1], ep_g[::-1], color=sc["col"], lw=2.5, zorder=3,
                label=(f"{sc['label']} ({sc['entry_val']:.3f} PSU)"
                       f"   AAL={sc['aal']/1e6:.3f}M tn  ({rate:.1f}% of max)"))
        ax.scatter(sc["pay_hist"], sc["ep_hist"],
                   color=sc["col"], s=55, zorder=5,
                   edgecolors="white", lw=0.6, marker=sc["marker"],
                   label=f"Historical {sc['key']} ({sc['n_hist']} years)")

        ax.axvline(sc["aal"], color=sc["col"], lw=1.2, ls=":", alpha=0.85, zorder=4)
        ax.annotate(
            f"AAL {sc['key']}\n{sc['aal']/1e6:.3f}M tn\n{rate:.1f}% of max",
            xy=(sc["aal"], ay), xycoords=("data", "axes fraction"),
            xytext=(6, 0), textcoords="offset points",
            fontsize=10, color=sc["col"], va="center",
        )

    ax.axvline(total_baseline, color="#43A047", lw=1.5, ls="--", alpha=0.85,
               zorder=4, label=f"Max payout = baseline ({total_baseline/1e6:.2f}M tn)")

    ax.set_xlim(left=0, right=total_baseline * 1.03)
    ax.set_ylabel("Annual exceedance probability (AEP)", fontsize=9)
    ax.set_xlabel("Aggregate annual payout (tons)", fontsize=9)
    ax.set_title("AEP ramp payout SSS  |  North + North Central + South Central",
                 loc="left", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.2f}M"))
    ax.legend(fontsize=8, frameon=False, loc="upper right",
              bbox_to_anchor=(0.88, 1.0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

def md_table(headers, rows):
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines = ["| " + " | ".join(headers) + " |", sep]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    region_results = []
    for reg in REGIONS:
        name    = reg["name"]
        lat_min = reg["lat_min"]
        lat_max = reg["lat_max"]
        print(f"\n{name}: cargando SSS...")
        weekly = load_sss_region(lat_min, lat_max)
        if weekly.empty:
            print(f"  {name}: sin datos SSS, saltando")
            continue
        print(f"  {len(weekly):,} pasos  {weekly.index[0].date()} - {weekly.index[-1].date()}")

        bl_t1, bl_t2 = load_baseline_region(name)
        bl_annual = bl_t1 + bl_t2
        print(f"  Baseline: T1={bl_t1/1e3:.0f}k  T2={bl_t2/1e3:.0f}k  "
              f"Anual={bl_annual/1e6:.3f}M tn")

        years, t1_obs, t2_obs = get_seasonal_means(weekly)
        print(f"  Anos con T1+T2 completos: {list(years)}")

        # triggers: percentiles BAJOS de SSS pooled (clim_mask)
        clim_mask = np.isin(years, CLIM_YEARS)
        sss_clim  = np.concatenate([t1_obs[clim_mask], t2_obs[clim_mask]])
        triggers  = {lbl: float(np.percentile(sss_clim, p))
                     for lbl, p in [("p20", 20), ("p10", 10), ("p05", 5), ("p01", 1)]}
        print(f"  Triggers SSS: p20={triggers['p20']:.3f}  p10={triggers['p10']:.3f}  "
              f"p05={triggers['p05']:.3f}  p01={triggers['p01']:.3f} PSU")

        t1_synth, t2_synth = parametric_bootstrap(t1_obs[clim_mask],
                                                   t2_obs[clim_mask])

        ramp_synth = {}
        ramp_hist  = {}
        for sc in SCENARIOS:
            entry = triggers[sc["key"]]
            exit_ = triggers["p01"]
            ramp_synth[sc["key"]] = ramp_payout_sss(
                t1_synth, t2_synth, bl_t1, bl_t2, entry, exit_)
            t1_h = t1_obs[clim_mask]
            t2_h = t2_obs[clim_mask]
            ramp_hist[sc["key"]] = ramp_payout_sss(
                t1_h, t2_h, bl_t1, bl_t2, entry, exit_)

        region_results.append(dict(
            name=name, col=reg["col"],
            bl_t1=bl_t1, bl_t2=bl_t2, bl_annual=bl_annual,
            triggers=triggers,
            t1_obs=t1_obs, t2_obs=t2_obs,
            years=years, clim_mask=clim_mask,
            ramp_synth=ramp_synth, ramp_hist=ramp_hist,
        ))

    if not region_results:
        print("Sin datos para ninguna region.")
        return

    total_baseline = sum(r["bl_annual"] for r in region_results)
    program_ton    = TOTAL_PROGRAM_TON if TOTAL_PROGRAM_TON else total_baseline
    scale          = program_ton / total_baseline
    n_hist         = int(region_results[0]["clim_mask"].sum())

    print(f"\nBaseline total: {total_baseline/1e6:.3f}M tn")
    print(f"Programa:       {program_ton/1e6:.3f}M tn")

    # -- agrega pagos entre regiones --
    scenarios_data = []
    for sc in SCENARIOS:
        pay_synth_agg = sum(r["ramp_synth"][sc["key"]] for r in region_results)
        pay_hist_agg  = sum(r["ramp_hist"][sc["key"]]  for r in region_results)

        pay_synth_agg = np.minimum(pay_synth_agg * scale, program_ton)
        pay_hist_agg  = np.minimum(pay_hist_agg  * scale, program_ton)

        n_h      = len(pay_hist_agg)
        ep_h     = (n_h + 1 - np.arange(1, n_h + 1)) / (n_h + 1)
        h_sorted = np.sort(pay_hist_agg)
        aal      = float(pay_synth_agg.mean())

        # entry value de referencia (North Central)
        cn = next(r for r in region_results if r["name"] == "North Central")
        entry_val = cn["triggers"][sc["key"]]

        print(f"Escenario {sc['key']}: AAL={aal/1e3:,.0f}k tn  "
              f"({aal/program_ton*100:.2f}%)")

        scenarios_data.append(dict(
            key=sc["key"], label=sc["label"],
            entry_val=entry_val,
            col=sc["col"], marker=sc["marker"],
            pay_synth=pay_synth_agg,
            pay_hist=h_sorted, ep_hist=ep_h, n_hist=n_h,
            aal=aal,
        ))

    # -- AEP plot A: verde (p20) + naranja (p10) --
    sc_A = [s for s in scenarios_data if s["key"] in ("p20", "p10")]
    plot_aep_ramp(sc_A, program_ton, PLOTS / "21_reinsurance_sss_aep_A.png",
                  ann_y=[0.76, 0.55])

    # -- AEP plot B: naranja (p10) + rojo (p05) --
    sc_B = [s for s in scenarios_data if s["key"] in ("p10", "p05")]
    plot_aep_ramp(sc_B, program_ton, PLOTS / "21_reinsurance_sss_aep_B.png",
                  ann_y=[0.76, 0.55])

    # -- markdown report --
    out_md = PLOTS / "21_reinsurance_sss_analysis.md"
    lines  = [
        "# Analisis de reaseguro SSS - Portafolio agregado",
        "",
        f"Regiones: Norte + Centro Norte + Centro Sur.  "
        f"SSS OISSS {CLIM_YEARS[0]}-{CLIM_YEARS[-1]}.  "
        f"N = {n_hist} anos por region.  "
        f"Bootstrap parametrico: {N_BOOTSTRAP} anos sinteticos.",
        "",
        f"Programa: **{program_ton/1e6:.3f}M ton** = "
        f"**USD {program_ton*PRICE_USD_TON/1e6:.1f}M**.  "
        f"Precio: USD {PRICE_USD_TON}/tm.  "
        f"Loss ratio objetivo: {LOSS_RATIO*100:.0f}%.  "
        f"Trigger: SSS baja (agua dulce = El Nino).  "
        f"Ventanas: T1=Abr-Jul, T2=Nov-Dic.",
        "",
        "## 1. Baselines regionales",
        "",
    ]

    bl_rows = []
    for r in region_results:
        share = r["bl_annual"] / total_baseline * 100
        bl_rows.append([r["name"],
                        f"{r['bl_t1']/1e3:,.0f}k", f"{r['bl_t2']/1e3:,.0f}k",
                        f"{r['bl_annual']/1e6:.3f}M", f"{share:.1f}%"])
    bl_rows.append(["**Total**",
                    f"**{sum(r['bl_t1'] for r in region_results)/1e3:,.0f}k**",
                    f"**{sum(r['bl_t2'] for r in region_results)/1e3:,.0f}k**",
                    f"**{total_baseline/1e6:.3f}M**", "**100%**"])
    lines += [md_table(["Region", "Baseline T1", "Baseline T2", "Baseline anual", "Share"],
                       bl_rows), ""]

    lines += [f"## 2. Triggers por region (SSS pooled T1+T2, {CLIM_YEARS[0]}-{CLIM_YEARS[-1]})", ""]
    trig_rows = []
    for r in region_results:
        t = r["triggers"]
        trig_rows.append([r["name"],
                          f"{t['p20']:.3f}", f"{t['p10']:.3f}",
                          f"{t['p05']:.3f}", f"{t['p01']:.3f}"])
    lines += [md_table(["Region", "p20 (PSU)", "p10 (PSU)", "p05 (PSU)", "p01 (PSU)"],
                       trig_rows), ""]

    lines += ["## 3. Pricing", ""]
    pr_rows = []
    for sc_d in scenarios_data:
        aal     = sc_d["aal"]
        rol     = aal / program_ton
        pp_usd  = aal * PRICE_USD_TON
        pc_usd  = pp_usd / LOSS_RATIO
        n_pay   = int((sc_d["pay_hist"] > 0).sum())
        pr_rows.append([
            sc_d["label"],
            f"{sc_d['entry_val']:.3f} PSU",
            f"{rol*100:.2f}%",
            f"{aal/1e3:,.0f}k ton",
            f"USD {pp_usd/1e3:,.0f}k",
            f"USD {pc_usd/1e3:,.0f}k",
            f"{n_pay} / {n_hist}",
        ])
    lines += [md_table(["Escenario", "Entry (CN ref.)", "RoL", "Prima pura (ton/ano)",
                         "Prima pura (USD/ano)", "Prima comercial (USD/ano)",
                         "Anos con pago"], pr_rows), ""]

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved -> {out_md}")


if __name__ == "__main__":
    main()
