"""
Step 15b - Bootstrap AEP agregado por SSS (Norte + Centro Norte + Centro Sur)

Analogo a 15_pricing_bootstrap_aep.py pero usa anomalia de salinidad (SSS)
como variable de disparo en lugar de SST.

Logica de disparo invertida respecto a SST:
  - SST alta (calida) -> mal para anchoveta -> pago
  - SSS baja (agua dulce, El Nino) -> mal para anchoveta -> pago
  Entrada: percentil p10/p20 de SSS (cola baja)
  Ramp: frac = clip((entry - sss) / (entry - exit), 0, 1)
        con entry = p20, exit = p01 de la distribucion historica

Inputs:
  FEATURES/sss_anomaly_weekly_{year}.nc      (de step 03b)
  OUTPUTS/catch_summary_by_region.csv
  OUTPUTS/calas_all_data.csv

Outputs:
  PLOTS/15b_pricing_bootstrap_aep_sss.png
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import xarray as xr
from matplotlib.path import Path as MplPath

from config import FEATURES, OUTPUTS, PLOTS, SSS_YEARS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LON_W, LON_E = -83.0, -74.0

T1_DOY_START, T1_DOY_END = 91, 212    # Apr 1 - Jul 31
T2_DOY_START, T2_DOY_END = 305, 365   # Nov 1 - Dec 31

N_BOOTSTRAP = 4000
CLIM_YEARS  = SSS_YEARS                # use full record for climatology

REGIONS = [
    dict(name="Norte",        lat_min=-7.1,  lat_max=None,  col="#d6604d"),
    dict(name="Centro Norte", lat_min=-11.0, lat_max=-7.1,  col="#2166ac"),
    dict(name="Centro Sur",   lat_min=-15.8, lat_max=-11.0, col="#4dac26"),
]

# Escenarios: entry = percentil bajo de SSS (agua fresca = mal)
SCENARIOS = [
    dict(key="p20", label="Entry p20", col="#e07b39", marker="o"),
    dict(key="p10", label="Entry p10", col="#4a7c6f", marker="s"),
]

# ---------------------------------------------------------------------------
# Fishing polygon (identical to script 15)
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

    lat_full  = np.concatenate([[actual_min], valid_lats, [actual_max]])
    west_full = np.concatenate([[west_lons[0]], west_lons, [west_lons[-1]]])
    east_full = np.concatenate([[east_lons[0]], east_lons, [east_lons[-1]]])

    poly_lons = np.concatenate([west_full, east_full[::-1], [west_full[0]]])
    poly_lats = np.concatenate([lat_full,  lat_full[::-1],  [lat_full[0]]])

    verts = list(zip(poly_lons, poly_lats))
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
    return MplPath(verts, codes), actual_min, actual_max


# ---------------------------------------------------------------------------
# SSS loading
# ---------------------------------------------------------------------------

def load_sss_region(lat_min, lat_max):
    """SSS anomalia espacial-media dentro del poligono de pesca regional."""
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
# Baseline catch (identical to script 15)
# ---------------------------------------------------------------------------

def load_baseline_region(region_name):
    summary = pd.read_csv(OUTPUTS / "catch_summary_by_region.csv")
    sub = summary[summary["region"] == region_name].copy()

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
# Seasonal SSS means
# ---------------------------------------------------------------------------

def get_seasonal_means(daily):
    df = daily.rename("sss").reset_index()
    df.columns = ["date", "sss"]
    df["year"] = df["date"].dt.year
    df["doy"]  = df["date"].dt.dayofyear

    t1 = df[df["doy"].between(T1_DOY_START, T1_DOY_END)].groupby("year")["sss"].mean()
    t2 = df[df["doy"].between(T2_DOY_START, T2_DOY_END)].groupby("year")["sss"].mean()

    common = sorted(t1.index.intersection(t2.index))
    return np.array(common), t1.loc[common].values, t2.loc[common].values


# ---------------------------------------------------------------------------
# Bootstrap (identical structure to script 15)
# ---------------------------------------------------------------------------

def parametric_bootstrap(t1_obs, t2_obs, n=N_BOOTSTRAP):
    mu1, sd1 = t1_obs.mean(), t1_obs.std(ddof=1)
    mu2, sd2 = t2_obs.mean(), t2_obs.std(ddof=1)
    rng = np.random.default_rng(42)
    return rng.normal(mu1, sd1, n), rng.normal(mu2, sd2, n)


# ---------------------------------------------------------------------------
# Ramp payout - SSS direction (low SSS = payout)
#
# entry > exit: entry is the "mild fresh" threshold, exit is extreme fresh.
# frac = clip((entry - sss) / (entry - exit), 0, 1)
# ---------------------------------------------------------------------------

def ramp_payout_sss(sss_t1, sss_t2, baseline_t1, baseline_t2, entry_sss, exit_sss):
    annual_baseline = baseline_t1 + baseline_t2
    denom = entry_sss - exit_sss
    if abs(denom) < 1e-9:
        return np.zeros(len(sss_t1))
    frac_t1 = np.clip((entry_sss - sss_t1) / denom, 0.0, 1.0)
    frac_t2 = np.clip((entry_sss - sss_t2) / denom, 0.0, 1.0)
    return np.minimum(annual_baseline * (frac_t1 + frac_t2), annual_baseline)


# ---------------------------------------------------------------------------
# AEP helpers (identical to script 15)
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
# Figure
# ---------------------------------------------------------------------------

def plot_aep_sss(scenarios_data, total_baseline, out_path):
    fig, ax = plt.subplots(figsize=(9, 7))

    ann_y = [0.72, 0.45]
    for sc, ay in zip(scenarios_data, ann_y):
        ep_g, lo_g, med_g, hi_g = aep_ci(sc["pay_synth"])
        rate = sc["aal"] / total_baseline * 100
        ax.fill_betweenx(ep_g[::-1], lo_g[::-1], hi_g[::-1],
                         alpha=0.20, color=sc["col"])
        ax.plot(med_g[::-1], ep_g[::-1], color=sc["col"], lw=2.5, zorder=3,
                label=(f"{sc['label']} ({sc['entry_val']:.3f} PSU)   "
                       f"AAL={sc['aal']:,.0f} tn  ({rate:.1f}% of max)"))
        ax.scatter(sc["pay_hist"], sc["ep_hist"],
                   color=sc["col"], s=55, zorder=5,
                   edgecolors="white", lw=0.6, marker=sc["marker"],
                   label=f"Historical {sc['key']} ({sc['n_hist']} years)")

        ax.axvline(sc["aal"], color=sc["col"], lw=1.2, ls=":", alpha=0.85, zorder=4)
        ax.annotate(
            f"AAL {sc['key']}\n{sc['aal']/1e6:.3f}M tn\n{rate:.1f}% of max",
            xy=(sc["aal"], ay), xycoords=("data", "axes fraction"),
            xytext=(6, 0), textcoords="offset points",
            fontsize=7.5, color=sc["col"], va="center",
        )

    ax.axvline(total_baseline, color="#43A047", lw=1.5, ls="--", alpha=0.85,
               zorder=4, label=f"Max payout = baseline ({total_baseline/1e6:.2f}M tn)")

    ax.set_xlim(left=0, right=total_baseline * 1.03)
    ax.set_ylabel("Annual exceedance probability (AEP)", fontsize=9)
    ax.set_xlabel("Aggregate annual payout (tons)", fontsize=9)
    ax.set_title(
        "AEP ramp payout  |  SSS trigger  |  Norte + Centro Norte + Centro Sur",
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
# Main
# ---------------------------------------------------------------------------

def main():
    out_path = PLOTS / "15b_pricing_bootstrap_aep_sss.png"
    if out_path.exists():
        print("15b output exists -- skipping")
        return

    region_results = []
    for reg in REGIONS:
        name    = reg["name"]
        lat_min = reg["lat_min"]
        lat_max = reg["lat_max"]
        print(f"\n{name}: cargando SSS...")
        sss = load_sss_region(lat_min, lat_max)
        if sss.empty:
            print(f"  {name}: sin datos SSS, saltando")
            continue
        print(f"  {len(sss):,} pasos  {sss.index[0].date()} - {sss.index[-1].date()}")

        bl_t1, bl_t2 = load_baseline_region(name)
        bl_annual = bl_t1 + bl_t2
        print(f"  Baseline: T1={bl_t1/1e3:.0f}k  T2={bl_t2/1e3:.0f}k  Anual={bl_annual/1e6:.3f}M tn")

        years, t1_obs, t2_obs = get_seasonal_means(sss)
        print(f"  Anos con T1+T2: {years}")

        # triggers: percentiles BAJOS de la distribucion de SSS (fresco = malo)
        sss_pooled = np.concatenate([t1_obs, t2_obs])
        triggers = {lbl: float(np.percentile(sss_pooled, p))
                    for lbl, p in [("p01", 1), ("p10", 10), ("p20", 20)]}
        print(f"  Triggers SSS: p20={triggers['p20']:.3f}  "
              f"p10={triggers['p10']:.3f}  p01={triggers['p01']:.3f} PSU")

        t1_synth, t2_synth = parametric_bootstrap(t1_obs, t2_obs)

        ramp_synth = {}
        ramp_hist  = {}
        for sc in SCENARIOS:
            entry = triggers[sc["key"]]
            exit_ = triggers["p01"]
            ramp_synth[sc["key"]] = ramp_payout_sss(
                t1_synth, t2_synth, bl_t1, bl_t2, entry, exit_)
            ramp_hist[sc["key"]] = ramp_payout_sss(
                t1_obs, t2_obs, bl_t1, bl_t2, entry, exit_)

        region_results.append(dict(
            name=name, bl_t1=bl_t1, bl_t2=bl_t2, bl_annual=bl_annual,
            triggers=triggers, t1_obs=t1_obs, t2_obs=t2_obs,
            years=years,
            ramp_synth=ramp_synth, ramp_hist=ramp_hist,
        ))

    if not region_results:
        print("Sin datos para ninguna region.")
        return

    total_baseline = sum(r["bl_annual"] for r in region_results)
    print(f"\nBaseline total agregado: {total_baseline/1e6:.3f}M tn")

    scenarios_data = []
    for sc in SCENARIOS:
        pay_synth_agg = sum(r["ramp_synth"][sc["key"]] for r in region_results)
        pay_hist_agg  = sum(r["ramp_hist"][sc["key"]]  for r in region_results)
        pay_synth_agg = np.minimum(pay_synth_agg, total_baseline)
        pay_hist_agg  = np.minimum(pay_hist_agg,  total_baseline)

        n_h  = len(pay_hist_agg)
        ep_h = (n_h + 1 - np.arange(1, n_h + 1)) / (n_h + 1)
        h_sorted = np.sort(pay_hist_agg)

        aal = float(pay_synth_agg.mean())
        print(f"Escenario {sc['key']}: AAL={aal/1e6:.4f}M tn ({aal/total_baseline*100:.2f}%)")

        cn = next(r for r in region_results if r["name"] == "Centro Norte")
        entry_val = cn["triggers"][sc["key"]]

        scenarios_data.append(dict(
            key=sc["key"],
            label=sc["label"],
            col=sc["col"], marker=sc["marker"],
            pay_synth=pay_synth_agg,
            pay_hist=h_sorted, ep_hist=ep_h, n_hist=n_h,
            aal=aal, entry_val=entry_val,
        ))

    plot_aep_sss(scenarios_data, total_baseline, out_path)


if __name__ == "__main__":
    main()
