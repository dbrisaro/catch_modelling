"""
Step 14b - OLS: log(catch) ~ SSS anomaly OISSS

Analogo exacto a 14_analysis_sst_catch_ols.py usando oisss_sss_anom.
OLS en 5 niveles de agregacion por region.

Inputs:
  OUTPUTS/calas_enriched.csv    (requiere columna oisss_sss_anom, de step 07b)

Outputs:
  PLOTS/14b_analysis_sss_catch_ols_betas.png
  PLOTS/14b_analysis_sss_catch_ols_betas_all_regions.png

Skip logic: skipped si los outputs ya existen.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from matplotlib.patches import Patch

from config import OUTPUTS, PLOTS

REGIONS = [
    ("Norte",        -7.1,  None),
    ("Centro Norte", -11.0, -7.1),
    ("Centro Sur",   -15.8, -11.0),
]

AGG_LEVELS = [
    ("Calas\nindividuales", None),
    ("Empresa\nx Diario",   "date"),
    ("Empresa\nx Semanal",  "year_week"),
    ("Empresa\nx Mensual",  "year_month"),
    ("Empresa\nx Temporada","season"),
]

COLOR_M1 = "#2166ac"
BAR_W    = 0.6


def run_ols(x, y):
    tmp = pd.DataFrame({"y": y, "x": x}).dropna()
    if len(tmp) < 10:
        return None
    m = smf.ols("y ~ x", data=tmp).fit()
    return {
        "beta": m.params["x"],
        "se":   m.bse["x"],
        "p":    m.pvalues["x"],
        "r2":   m.rsquared,
        "N":    int(m.nobs),
    }


def collect(sub, reg_name):
    betas, ses, pvals, ns, labels = [], [], [], [], []
    for label, tcol in AGG_LEVELS:
        if tcol is None:
            res = run_ols(sub["oisss_sss_anom"], sub["log_catch"])
        else:
            agg = sub.groupby(["company", tcol]).agg(
                total_catch=("catch_tm", "sum"),
                mean_sss_anom=("oisss_sss_anom", "mean"),
            ).reset_index().dropna(subset=["mean_sss_anom"])
            res = run_ols(agg["mean_sss_anom"], np.log(agg["total_catch"]))
        if res is None:
            continue
        betas.append(res["beta"])
        ses.append(res["se"])
        pvals.append(res["p"])
        ns.append(res["N"])
        labels.append(label)
        print(f"  {reg_name:12s} | {label.replace(chr(10),' '):25s} | "
              f"beta={res['beta']:+.3f}  se={res['se']:.3f}  "
              f"p={res['p']:.4f}  R2={res['r2']:.3f}  N={res['N']:,}")
    return betas, ses, pvals, ns, labels


def plot_betas_panel(ax, betas, ses, pvals, ns, labels, ylim, ymax):
    x_pos = np.arange(len(labels))
    for i, (b, se, p, n) in enumerate(zip(betas, ses, pvals, ns)):
        col = COLOR_M1 if p < 0.05 else "#aaaaaa"
        ax.bar(x_pos[i], b, width=BAR_W, color=col, alpha=0.85, zorder=3)
        ax.errorbar(x_pos[i], b, yerr=1.96 * se,
                    fmt="none", color="black", capsize=3, lw=1.0, zorder=4)
        pstr = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        ax.text(x_pos[i],
                b + np.sign(b) * (1.96 * se + ymax * 0.03),
                pstr, ha="center", va="bottom" if b >= 0 else "top", fontsize=7)
        ax.text(x_pos[i], ylim[0] + ymax * 0.02,
                f"{n:,}", ha="center", va="bottom", fontsize=5.5, color="#333333")
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_ylim(ylim)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Beta (OLS coefficient)", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=[
        Patch(color=COLOR_M1, alpha=0.85, label="log(captura) ~ SSS OISSS  (2015-2024)"),
        Patch(color="#aaaaaa", alpha=0.85, label="p >= 0.05"),
    ], frameon=False, fontsize=7, loc="upper right")


def main():
    out     = PLOTS / "14b_analysis_sss_catch_ols_betas.png"
    out_all = PLOTS / "14b_analysis_sss_catch_ols_betas_all_regions.png"

    df = pd.read_csv(OUTPUTS / "calas_enriched.csv", low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["oisss_sss_anom", "catch_tm", "lat", "lon", "season", "company"])
    df = df[df["catch_tm"] > 0]
    df["log_catch"] = np.log(df["catch_tm"])

    iso = df["date"].dt.isocalendar()
    df["year_week"]  = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    if not out.exists():
        panel_data = []
        for reg_name, lat_min, lat_max in REGIONS:
            sub = df[df["lat"] > lat_min].copy()
            if lat_max is not None:
                sub = sub[sub["lat"] <= lat_max]
            print(f"\n--- {reg_name} ---")
            panel_data.append((reg_name, collect(sub, reg_name)))

        all_b, all_s = [], []
        for _, (b, s, *_) in panel_data:
            all_b.extend(b); all_s.extend(s)
        ymax = (np.abs(all_b) + 1.96 * np.array(all_s)).max() * 1.3
        ylim = (-ymax, ymax)

        fig, axes = plt.subplots(1, len(REGIONS), figsize=(8 * len(REGIONS), 6))
        for ax, (reg_name, m1) in zip(axes, panel_data):
            b1, se1, p1, n1, lab1 = m1
            plot_betas_panel(ax, b1, se1, p1, n1, lab1, ylim, ymax)
            ax.set_title(f"Region {reg_name}", loc="left", fontsize=11)

        fig.suptitle(
            "OLS betas  |  log(captura) ~ SSS_anom OISSS  |  por nivel de agregacion  |  "
            "Norte  |  Centro Norte  |  Centro Sur",
            fontsize=10, x=0.01, ha="left")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(out, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"\nSaved -> {out}")
    else:
        print(f"{out.name} exists -- skipping")

    if not out_all.exists():
        print("\n--- All regions combined ---")
        b_all, se_all, p_all, n_all, lab_all = collect(df, "All regions")

        ymax = (np.abs(b_all) + 1.96 * np.array(se_all)).max() * 1.3
        ylim = (-ymax, ymax)

        fig, ax = plt.subplots(figsize=(8, 6))
        plot_betas_panel(ax, b_all, se_all, p_all, n_all, lab_all, ylim, ymax)
        ax.set_title("All regions  (Norte + Centro Norte + Centro Sur)", loc="left", fontsize=11)
        fig.suptitle(
            "OLS betas  |  log(captura) ~ SSS_anom OISSS  |  por nivel de agregacion  |  todas las regiones",
            fontsize=10, x=0.01, ha="left")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(out_all, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"Saved -> {out_all}")
    else:
        print(f"{out_all.name} exists -- skipping")


if __name__ == "__main__":
    main()
