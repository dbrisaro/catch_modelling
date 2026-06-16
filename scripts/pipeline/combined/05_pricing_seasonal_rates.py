"""
05_pricing_seasonal_rates.py  —  SST only

Per-season thresholds: p90 and p99 are computed separately from the T1 distribution
and from the T2 distribution.  Each season is also capped at its own seasonal baseline.

Three AEP curves per scenario panel:
  T1   — payout from T1 season only,  capped at bl_t1
  T2   — payout from T2 season only,  capped at bl_t2
  Annual — T1 + T2 combined, capped at total baseline

Output:
  outputs/combined/29_sst_seasonal_rates.png
  outputs/combined/29_seasonal_rates_analysis.md
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FEATURES, OUTPUTS, PLOTS

FLAVOR_DIR = PLOTS / "combined"
FLAVOR_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOSS_RATIO  = 0.65
N_BOOTSTRAP = 4000
CLIM_YEARS  = list(range(2005, 2025))

LON_W, LON_E = -82.0, -74.0
T1_DOY_START, T1_DOY_END = 91,  212   # Apr 1 - Jul 31
T2_DOY_START, T2_DOY_END = 305, 365   # Nov 1 - Dec 31

REGIONS = [
    dict(name="North",         csv="Norte",        lat_min=-7.1,  lat_max=None),
    dict(name="North Central", csv="Centro Norte",  lat_min=-11.0, lat_max=-7.1),
    dict(name="South Central", csv="Centro Sur",    lat_min=-15.8, lat_max=-11.0),
]

SCENARIOS = [
    dict(key="p90", entry_pct=90, exit_pct=99, label="Entry p90 / Exit p99"),
    dict(key="p95", entry_pct=95, exit_pct=99, label="Entry p95 / Exit p99"),
]

COL_T1     = "#2166ac"   # blue
COL_T2     = "#e07b39"   # orange
COL_ANNUAL = "#43A047"   # green

EVENT_LABELS = {2015: "El Niño\n15-16", 2023: "El Niño\n23-24"}


# ---------------------------------------------------------------------------
# Fishing polygon
# ---------------------------------------------------------------------------

def build_polygon(lat_min, lat_max):
    df = (
        pd.read_csv(OUTPUTS / "calas_all_data.csv",
                    usecols=["latitud", "longitud"], low_memory=False)
        .rename(columns={"latitud": "lat", "longitud": "lon"})
        .dropna()
    )
    df = df[df["lat"] > lat_min]
    if lat_max is not None:
        df = df[df["lat"] <= lat_max]
    if df.empty:
        return None
    a_min, a_max = df["lat"].min(), df["lat"].max()
    wl, el, vl = [], [], []
    for lo in np.arange(int(np.floor(a_min)), int(np.ceil(a_max)), 1.0):
        band = df[(df["lat"] >= lo) & (df["lat"] < lo + 1.0)]
        if len(band) < 20:
            continue
        wl.append(np.percentile(band["lon"], 5))
        el.append(np.percentile(band["lon"], 95))
        vl.append(lo + 0.5)
    if not vl:
        return None
    vl = np.array(vl); wl = np.array(wl); el = np.array(el)
    lf = np.concatenate([[a_min], vl, [a_max]])
    wf = np.concatenate([[wl[0]], wl, [wl[-1]]])
    ef = np.concatenate([[el[0]], el, [el[-1]]])
    poly_lon = np.concatenate([wf, ef[::-1], [wf[0]]])
    poly_lat = np.concatenate([lf, lf[::-1], [lf[0]]])
    verts = list(zip(poly_lon, poly_lat))
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
    return MplPath(verts, codes), a_min, a_max


# ---------------------------------------------------------------------------
# SST loading
# ---------------------------------------------------------------------------

def load_sst_region(lat_min, lat_max):
    res = build_polygon(lat_min, lat_max)
    if res is None:
        return pd.Series(dtype=float)
    poly, a_min, a_max = res
    lat_lo, lat_hi = a_min - 0.1, a_max + 0.1
    parts = []
    for yr in CLIM_YEARS:
        f = FEATURES / f"sst_anomaly_daily_{yr}.nc"
        if not f.exists():
            continue
        ds = xr.open_dataset(f)
        lv, lov = ds["lat"].values, ds["lon"].values
        lm  = (lv > lat_lo) & (lv <= lat_hi)
        lom = (lov >= LON_W) & (lov <= LON_E)
        ls, los = lv[lm], lov[lom]
        Glo, Gla = np.meshgrid(los, ls)
        pts = np.column_stack([Glo.ravel(), Gla.ravel()])
        m2d = poly.contains_points(pts).reshape(ls.size, los.size)
        mda = xr.DataArray(m2d, dims=["lat", "lon"], coords={"lat": ls, "lon": los})
        s = (ds["sst_anomaly"].isel(lat=lm, lon=lom)
             .where(mda).mean(dim=["lat", "lon"]).to_series())
        s.index = pd.to_datetime(s.index)
        ds.close()
        parts.append(s.dropna())
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts).sort_index().dropna()


def seasonal_means(daily):
    df = daily.rename("sst").reset_index()
    df.columns = ["date", "sst"]
    df["year"] = df["date"].dt.year
    df["doy"]  = df["date"].dt.dayofyear
    t1 = df[df["doy"].between(T1_DOY_START, T1_DOY_END)].groupby("year")["sst"].mean()
    t2 = df[df["doy"].between(T2_DOY_START, T2_DOY_END)].groupby("year")["sst"].mean()
    common = sorted(t1.index.intersection(t2.index))
    return np.array(common), t1.loc[common].values, t2.loc[common].values


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def load_baseline(csv_name):
    summary = pd.read_csv(OUTPUTS / "catch_summary_by_region.csv")
    sub = summary[summary["region"] == csv_name].copy()
    def tipo(s):
        s = str(s)
        if "1ra" in s or "-I" in s:  return "T1"
        if "2da" in s or "-II" in s: return "T2"
        return "other"
    sub["tipo"] = sub["temporada_key"].apply(tipo)
    t1 = sub[sub["tipo"] == "T1"]["captura_tm"]
    t2 = sub[sub["tipo"] == "T2"]["captura_tm"]
    mean_t1 = float(t1.mean()) if len(t1) else 0.0
    mean_t2 = float(t2.mean()) if len(t2) else 0.0
    max_t1  = float(t1.max())  if len(t1) else 0.0
    max_t2  = float(t2.max())  if len(t2) else 0.0
    return mean_t1, mean_t2, max_t1, max_t2


# ---------------------------------------------------------------------------
# Per-season thresholds
# ---------------------------------------------------------------------------

def season_triggers(obs, entry_pct, exit_pct):
    """Compute entry and exit thresholds from a single season's observations."""
    return (float(np.percentile(obs, entry_pct)),
            float(np.percentile(obs, exit_pct)))


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def parametric_bootstrap(t1_obs, t2_obs, n=N_BOOTSTRAP, seed=42):
    rng = np.random.default_rng(seed)
    return (rng.normal(t1_obs.mean(), t1_obs.std(ddof=1), n),
            rng.normal(t2_obs.mean(), t2_obs.std(ddof=1), n))


# ---------------------------------------------------------------------------
# Ramp: one season at a time, with its own thresholds
# ---------------------------------------------------------------------------

def ramp_one_season(t, bl, entry, exit_):
    """Payout for a single season: ramp from entry to exit, capped at bl."""
    denom = exit_ - entry
    if abs(denom) < 1e-9:
        return np.zeros(len(np.atleast_1d(t)))
    f = np.clip((np.atleast_1d(t) - entry) / denom, 0.0, 1.0)
    return np.minimum(bl * f, bl)


# ---------------------------------------------------------------------------
# AEP helpers
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
        sub = values[i * chunk:(i + 1) * chunk]
        sv_i, ep_i = empirical_aep(sub)
        curves.append(np.interp(ep_grid, ep_i[::-1], sv_i[::-1]))
    median = np.interp(ep_grid, ep_full[::-1], sv_full[::-1])
    ci_lo  = np.percentile(curves,  5, axis=0)
    ci_hi  = np.percentile(curves, 95, axis=0)
    return ep_grid, ci_lo, median, ci_hi


# ---------------------------------------------------------------------------
# Plot — exit threshold comparison
# ---------------------------------------------------------------------------

def plot_exit_comparison(region_results, agg_p99, agg_pmax,
                         total_baseline, bl_t1_total, bl_t2_total,
                         years_hist, out_path):
    """
    2 rows × 3 cols.
      Row 0: exit = p99
      Row 1: exit = max (p100)
      Cols : T1, T2, Annual
      Lines: solid = p90 entry, dashed = p95 entry
    """
    panels = [
        dict(title="T1 season (Apr-Jul)", syn_key="t1_syn",  hist_key="t1_hist",
             col=COL_T1,     cap=bl_t1_total,    cap_label=f"Max = {bl_t1_total/1e6:.2f}M tn"),
        dict(title="T2 season (Nov-Dec)", syn_key="t2_syn",  hist_key="t2_hist",
             col=COL_T2,     cap=bl_t2_total,    cap_label=f"Max = {bl_t2_total/1e6:.2f}M tn"),
        dict(title="Annual (T1 + T2)",   syn_key="ann_syn", hist_key="ann_hist",
             col=COL_ANNUAL, cap=total_baseline, cap_label=f"Max = {total_baseline/1e6:.2f}M tn"),
    ]
    row_configs = [
        dict(agg=agg_p99,  row_label="Exit = p99"),
        dict(agg=agg_pmax, row_label="Exit = max observed"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(17, 10), sharey=True, sharex="col")

    for row_i, rcfg in enumerate(row_configs):
        agg = rcfg["agg"]
        for col_i, panel in enumerate(panels):
            ax = axes[row_i, col_i]

            for sc in SCENARIOS:
                k    = sc["key"]
                syn  = agg[k][panel["syn_key"]]
                hist = agg[k][panel["hist_key"]]
                ls      = "-"  if k == "p90" else "--"
                alpha_b = 0.15 if k == "p90" else 0.08
                marker  = "o"  if k == "p90" else "s"

                ep_g, lo_g, med_g, hi_g = aep_ci(syn)
                aal  = float(syn.mean())
                rate = aal / panel["cap"] / LOSS_RATIO * 100

                ax.fill_betweenx(ep_g[::-1], lo_g[::-1], hi_g[::-1],
                                 alpha=alpha_b, color=panel["col"])
                ax.plot(med_g[::-1], ep_g[::-1], color=panel["col"], lw=2.0, ls=ls,
                        zorder=3,
                        label=f"{sc['label']}   AAL={aal/1e3:.0f}k tn   rate={rate:.1f}%")

                h_sort = np.sort(hist)
                n_h    = len(h_sort)
                ep_h   = (n_h + 1 - np.arange(1, n_h + 1)) / (n_h + 1)
                ax.scatter(h_sort, ep_h, color=panel["col"], s=35, marker=marker,
                           edgecolors="white", lw=0.6, zorder=5, alpha=0.85)

                if k == "p90":
                    asc_idx = np.argsort(hist)
                    labeled = set()
                    for rank, orig_idx in enumerate(asc_idx):
                        yr = years_hist[orig_idx]
                        if yr in EVENT_LABELS and yr not in labeled and h_sort[rank] > 0:
                            labeled.add(yr)
                            ax.annotate(EVENT_LABELS[yr],
                                        xy=(h_sort[rank], ep_h[rank]),
                                        xytext=(7, 2), textcoords="offset points",
                                        fontsize=7, color=panel["col"])

                ax.axvline(aal, color=panel["col"], lw=0.9, ls=":", alpha=0.65, zorder=4)

            ax.axvline(panel["cap"], color=panel["col"], lw=1.2, ls="--", alpha=0.4,
                       label=panel["cap_label"])
            ax.set_xlim(left=0, right=panel["cap"] * 1.05)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.2f}M"))
            ax.legend(fontsize=7.5, frameon=False, loc="upper right")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if row_i == 0:
                ax.set_title(panel["title"], fontsize=11, fontweight="bold")
            if col_i == 0:
                ax.set_ylabel(f"{rcfg['row_label']}\n\nAEP", fontsize=9)
            if row_i == 1:
                ax.set_xlabel("Aggregate payout (tons)", fontsize=9)

    fig.suptitle(
        "SST ramp AEP — exit threshold comparison  |  Norte + Centro Norte + Centro Sur\n"
        f"Cap = mean historical catch  |  Solid = p90 entry, Dashed = p95 entry  |  "
        f"N={len(years_hist)} years, {N_BOOTSTRAP} bootstrap",
        fontsize=9, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Plot — season / annual AEP
# ---------------------------------------------------------------------------

def plot_aep(scenarios, agg, total_baseline, bl_t1_total, bl_t2_total,
             years_hist, out_path, cap_label="mean catch cap"):
    panels = [
        dict(title="T1 season (Apr-Jul)", syn_key="t1_syn",  hist_key="t1_hist",
             col=COL_T1,     max_pay=bl_t1_total,    max_label=f"Max = {bl_t1_total/1e6:.2f}M tn"),
        dict(title="T2 season (Nov-Dec)", syn_key="t2_syn",  hist_key="t2_hist",
             col=COL_T2,     max_pay=bl_t2_total,    max_label=f"Max = {bl_t2_total/1e6:.2f}M tn"),
        dict(title="Annual (T1 + T2)",   syn_key="ann_syn", hist_key="ann_hist",
             col=COL_ANNUAL, max_pay=total_baseline, max_label=f"Max = {total_baseline/1e6:.2f}M tn"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True)

    for ax, panel in zip(axes, panels):
        for sc in scenarios:
            k    = sc["key"]
            syn  = agg[k][panel["syn_key"]]
            hist = agg[k][panel["hist_key"]]
            ls      = "-"  if k == "p90" else "--"
            alpha_b = 0.15 if k == "p90" else 0.08
            marker  = "o"  if k == "p90" else "s"

            ep_g, lo_g, med_g, hi_g = aep_ci(syn)
            aal  = float(syn.mean())
            rate = aal / panel["max_pay"] / LOSS_RATIO * 100

            ax.fill_betweenx(ep_g[::-1], lo_g[::-1], hi_g[::-1],
                             alpha=alpha_b, color=panel["col"])
            ax.plot(med_g[::-1], ep_g[::-1], color=panel["col"], lw=2.2, ls=ls, zorder=3,
                    label=f"{sc['label']}   AAL={aal/1e3:.0f}k tn   rate={rate:.1f}%")

            # Historical scatter
            h_sort = np.sort(hist)
            n_h    = len(h_sort)
            ep_h   = (n_h + 1 - np.arange(1, n_h + 1)) / (n_h + 1)
            ax.scatter(h_sort, ep_h, color=panel["col"], s=40, marker=marker,
                       edgecolors="white", lw=0.6, zorder=5, alpha=0.9)

            # Year labels on p90 curve only
            if k == "p90":
                asc_idx = np.argsort(hist)
                labeled = set()
                for rank, orig_idx in enumerate(asc_idx):
                    yr = years_hist[orig_idx]
                    if yr in EVENT_LABELS and yr not in labeled and h_sort[rank] > 0:
                        labeled.add(yr)
                        ax.annotate(
                            EVENT_LABELS[yr],
                            xy=(h_sort[rank], ep_h[rank]),
                            xytext=(7, 2), textcoords="offset points",
                            fontsize=8, color=panel["col"],
                        )

            # AAL vertical line
            ax.axvline(aal, color=panel["col"], lw=1.0, ls=":", alpha=0.7, zorder=4)

        # Max-payout reference line
        ax.axvline(panel["max_pay"], color=panel["col"], lw=1.2, ls="--", alpha=0.45,
                   label=panel["max_label"])

        ax.set_xlim(left=0, right=panel["max_pay"] * 1.05)
        ax.set_title(panel["title"], fontsize=11)
        ax.set_xlabel("Aggregate payout (tons)", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.2f}M"))
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Annual exceedance probability (AEP)", fontsize=9)
    fig.suptitle(
        f"SST ramp AEP by season  |  Norte + Centro Norte + Centro Sur  |  {cap_label}\n"
        f"Per-season thresholds  |  Solid = p90 entry, Dashed = p95 entry  |  "
        f"N={len(years_hist)} years, {N_BOOTSTRAP} bootstrap",
        fontsize=9, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def md_table(headers, rows):
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines = ["| " + " | ".join(headers) + " |", sep]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def write_report(scenarios, agg, region_results, total_baseline, bl_t1_total, bl_t2_total):
    lines = [
        "# SST pricing: per-season thresholds",
        "",
        "Each season uses its own p90/p99 thresholds computed from that season's observations.",
        "Payout per season: `min(bl_season * f, bl_season)`  where  "
        "`f = clip((sst - entry) / (exit - entry), 0, 1)`",
        "Annual payout = T1 payout + T2 payout.",
        "",
        f"CLIM_YEARS {CLIM_YEARS[0]}-{CLIM_YEARS[-1]}  |  N bootstrap: {N_BOOTSTRAP}  |  "
        f"Loss ratio: {LOSS_RATIO}",
        "",
        f"Total baseline: {total_baseline/1e6:.3f}M tn  "
        f"(T1={bl_t1_total/1e6:.3f}M  T2={bl_t2_total/1e6:.3f}M)",
        "",
        "## Thresholds by region",
        "",
    ]

    trg_rows = []
    for r in region_results:
        for sc in scenarios:
            k = sc["key"]
            trg_rows.append([
                r["name"], sc["label"],
                f"{r['trg_t1'][k]['entry']:.3f}C", f"{r['trg_t1'][k]['exit']:.3f}C",
                f"{r['trg_t2'][k]['entry']:.3f}C", f"{r['trg_t2'][k]['exit']:.3f}C",
            ])
    lines += [md_table(
        ["Region", "Scenario", "T1 entry", "T1 exit", "T2 entry", "T2 exit"],
        trg_rows), ""]

    lines += ["## Rates", ""]
    rate_rows = []
    for sc in scenarios:
        k = sc["key"]
        aal_t1  = float(agg[k]["t1_syn"].mean())
        aal_t2  = float(agg[k]["t2_syn"].mean())
        aal_ann = float(agg[k]["ann_syn"].mean())
        r_t1    = aal_t1  / bl_t1_total   / LOSS_RATIO * 100
        r_t2    = aal_t2  / bl_t2_total   / LOSS_RATIO * 100
        r_ann   = aal_ann / total_baseline / LOSS_RATIO * 100
        rate_rows.append([
            sc["label"],
            f"{aal_t1/1e3:.0f}k tn", f"{r_t1:.2f}%",
            f"{aal_t2/1e3:.0f}k tn", f"{r_t2:.2f}%",
            f"{aal_ann/1e3:.0f}k tn", f"{r_ann:.2f}%",
        ])
    lines += [md_table(
        ["Scenario",
         "AAL T1", "Rate T1",
         "AAL T2", "Rate T2",
         "AAL Annual", "Rate Annual"],
        rate_rows), ""]

    out_md = FLAVOR_DIR / "29_seasonal_rates_analysis.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved -> {out_md}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    region_results = []

    for reg in REGIONS:
        print(f"\n{reg['name']}: loading SST...")
        daily = load_sst_region(reg["lat_min"], reg["lat_max"])
        if daily.empty:
            print("  no data, skipping")
            continue

        bl_t1, bl_t2, max_t1, max_t2 = load_baseline(reg["csv"])
        years, t1_obs, t2_obs = seasonal_means(daily)

        # Per-season thresholds: T1 uses T1 obs, T2 uses T2 obs
        # Computed for SCENARIOS (exit=p99) and for exit=max (p100)
        def trg(obs, sc):
            e, x = season_triggers(obs, sc["entry_pct"], sc["exit_pct"])
            return {"entry": e, "exit": x}

        trg_t1 = {sc["key"]: trg(t1_obs, sc) for sc in SCENARIOS}
        trg_t2 = {sc["key"]: trg(t2_obs, sc) for sc in SCENARIOS}

        # exit=max variants (same entry, exit_pct=100)
        trg_t1_max = {sc["key"]: trg(t1_obs, {**sc, "exit_pct": 100}) for sc in SCENARIOS}
        trg_t2_max = {sc["key"]: trg(t2_obs, {**sc, "exit_pct": 100}) for sc in SCENARIOS}

        for sc in SCENARIOS:
            k = sc["key"]
            print(f"  {k}: T1 entry={trg_t1[k]['entry']:.3f}C exit={trg_t1[k]['exit']:.3f}C  |  "
                  f"T2 entry={trg_t2[k]['entry']:.3f}C exit={trg_t2[k]['exit']:.3f}C")
        print(f"  mean T1={bl_t1/1e3:.0f}k  T2={bl_t2/1e3:.0f}k tn  |  "
              f"max T1={max_t1/1e3:.0f}k  T2={max_t2/1e3:.0f}k tn")

        t1_syn, t2_syn = parametric_bootstrap(t1_obs, t2_obs)

        sc_results      = {}   # exit=p99, mean cap
        sc_results_max  = {}   # exit=p99, max cap
        sc_results_xmax = {}   # exit=max, mean cap

        for sc in SCENARIOS:
            k = sc["key"]
            e1, x1     = trg_t1[k]["entry"],     trg_t1[k]["exit"]
            e2, x2     = trg_t2[k]["entry"],     trg_t2[k]["exit"]
            e1m, x1m   = trg_t1_max[k]["entry"], trg_t1_max[k]["exit"]
            e2m, x2m   = trg_t2_max[k]["entry"], trg_t2_max[k]["exit"]

            # exit=p99, mean cap
            p1s = ramp_one_season(t1_syn, bl_t1, e1, x1)
            p2s = ramp_one_season(t2_syn, bl_t2, e2, x2)
            p1h = ramp_one_season(t1_obs, bl_t1, e1, x1)
            p2h = ramp_one_season(t2_obs, bl_t2, e2, x2)
            sc_results[k] = dict(
                t1_syn=p1s, t2_syn=p2s, ann_syn=p1s+p2s,
                t1_hist=p1h, t2_hist=p2h, ann_hist=p1h+p2h,
            )

            # exit=p99, max cap
            m1s = ramp_one_season(t1_syn, max_t1, e1, x1)
            m2s = ramp_one_season(t2_syn, max_t2, e2, x2)
            m1h = ramp_one_season(t1_obs, max_t1, e1, x1)
            m2h = ramp_one_season(t2_obs, max_t2, e2, x2)
            sc_results_max[k] = dict(
                t1_syn=m1s, t2_syn=m2s, ann_syn=m1s+m2s,
                t1_hist=m1h, t2_hist=m2h, ann_hist=m1h+m2h,
            )

            # exit=max, mean cap
            x1s = ramp_one_season(t1_syn, bl_t1, e1m, x1m)
            x2s = ramp_one_season(t2_syn, bl_t2, e2m, x2m)
            x1h = ramp_one_season(t1_obs, bl_t1, e1m, x1m)
            x2h = ramp_one_season(t2_obs, bl_t2, e2m, x2m)
            sc_results_xmax[k] = dict(
                t1_syn=x1s, t2_syn=x2s, ann_syn=x1s+x2s,
                t1_hist=x1h, t2_hist=x2h, ann_hist=x1h+x2h,
            )

        region_results.append(dict(
            name=reg["name"], bl_t1=bl_t1, bl_t2=bl_t2,
            max_t1=max_t1, max_t2=max_t2,
            years=years, trg_t1=trg_t1, trg_t2=trg_t2,
            sc_results=sc_results, sc_results_max=sc_results_max,
            sc_results_xmax=sc_results_xmax,
        ))

    if not region_results:
        print("No data.")
        return

    total_baseline = sum(r["bl_t1"] + r["bl_t2"] for r in region_results)
    bl_t1_total    = sum(r["bl_t1"] for r in region_results)
    bl_t2_total    = sum(r["bl_t2"] for r in region_results)
    max_t1_total   = sum(r["max_t1"] for r in region_results)
    max_t2_total   = sum(r["max_t2"] for r in region_results)
    total_max      = max_t1_total + max_t2_total
    years_hist     = region_results[0]["years"]
    print(f"\nMean cap: {total_baseline/1e6:.3f}M tn  (T1={bl_t1_total/1e6:.3f}M  T2={bl_t2_total/1e6:.3f}M)")
    print(f"Max cap:  {total_max/1e6:.3f}M tn  (T1={max_t1_total/1e6:.3f}M  T2={max_t2_total/1e6:.3f}M)")

    def aggregate(results_key, cap_t1, cap_t2, cap_ann):
        agg = {}
        for sc in SCENARIOS:
            k = sc["key"]
            agg[k] = {}
            for key in ("t1_syn", "t2_syn", "ann_syn", "t1_hist", "t2_hist", "ann_hist"):
                total = sum(r[results_key][k][key] for r in region_results)
                if "ann" in key:
                    agg[k][key] = np.minimum(total, cap_ann)
                elif "t1" in key:
                    agg[k][key] = np.minimum(total, cap_t1)
                else:
                    agg[k][key] = np.minimum(total, cap_t2)
        return agg

    agg_mean  = aggregate("sc_results",      bl_t1_total,  bl_t2_total,  total_baseline)
    agg_max   = aggregate("sc_results_max",  max_t1_total, max_t2_total, total_max)
    agg_xmax  = aggregate("sc_results_xmax", bl_t1_total,  bl_t2_total,  total_baseline)

    plot_aep(SCENARIOS, agg_mean, total_baseline, bl_t1_total, bl_t2_total,
             years_hist, FLAVOR_DIR / "29_sst_seasonal_rates.png",
             cap_label="cap = mean historical catch")
    plot_aep(SCENARIOS, agg_max,  total_max,      max_t1_total, max_t2_total,
             years_hist, FLAVOR_DIR / "30_sst_maxcap_rates.png",
             cap_label="cap = max historical catch")
    plot_exit_comparison(region_results, agg_mean, agg_xmax,
                         total_baseline, bl_t1_total, bl_t2_total,
                         years_hist, FLAVOR_DIR / "31_sst_exit_comparison.png")
    write_report(SCENARIOS, agg_mean, region_results, total_baseline, bl_t1_total, bl_t2_total)

    print("\n--- Rates summary (mean cap) ---")
    for sc in SCENARIOS:
        k = sc["key"]
        r_t1  = float(agg_mean[k]["t1_syn"].mean())  / bl_t1_total   / LOSS_RATIO * 100
        r_t2  = float(agg_mean[k]["t2_syn"].mean())  / bl_t2_total   / LOSS_RATIO * 100
        r_ann = float(agg_mean[k]["ann_syn"].mean()) / total_baseline / LOSS_RATIO * 100
        print(f"  {sc['label']}: T1={r_t1:.2f}%  T2={r_t2:.2f}%  Annual={r_ann:.2f}%")

    print("\n--- Rates summary (max cap) ---")
    for sc in SCENARIOS:
        k = sc["key"]
        r_t1  = float(agg_max[k]["t1_syn"].mean())  / max_t1_total / LOSS_RATIO * 100
        r_t2  = float(agg_max[k]["t2_syn"].mean())  / max_t2_total / LOSS_RATIO * 100
        r_ann = float(agg_max[k]["ann_syn"].mean()) / total_max    / LOSS_RATIO * 100
        print(f"  {sc['label']}: T1={r_t1:.2f}%  T2={r_t2:.2f}%  Annual={r_ann:.2f}%")


if __name__ == "__main__":
    main()
