"""
Step 13b - Scatter: SSS anomaly OISSS vs log(catch)

Analogo exacto a 13_analysis_sst_catch_scatter.py usando oisss_sss_anom.
5 paneles por region: calas individuales, empresa x diario/semanal/mensual/temporada.
Regions: Norte (lat > -7.1), Centro Norte (-11 < lat <= -7.1), Centro Sur (-15.8 < lat <= -11)

Inputs:
  OUTPUTS/calas_enriched.csv    (requiere columna oisss_sss_anom, de step 07b)

Outputs:
  PLOTS/13b_analysis_sss_catch_scatter_norte.png
  PLOTS/13b_analysis_sss_catch_scatter_centro_norte.png
  PLOTS/13b_analysis_sss_catch_scatter_centro_sur.png

Skip logic: skipped si todos los outputs ya existen.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr

from config import OUTPUTS, PLOTS

REGIONS = [
    ("norte",        -7.1,  None),
    ("centro_norte", -11.0, -7.1),
    ("centro_sur",   -15.8, -11.0),
]

AGG_LEVELS = [
    ("Diario",    "date"),
    ("Semanal",   "year_week"),
    ("Mensual",   "year_month"),
    ("Temporada", "season"),
]


def draw_scatter(ax, x, y, colors, s, alpha, label, ylabel=None):
    if hasattr(colors, "__iter__") and not isinstance(colors, str):
        ax.scatter(x, y, s=s, alpha=alpha, c=colors, rasterized=True, zorder=3,
                   edgecolors="white", linewidths=0.3)
    else:
        ax.scatter(x, y, s=s, alpha=alpha, color=colors, rasterized=True, zorder=3,
                   edgecolors="white", linewidths=0.3)
    slope, intercept = np.polyfit(x, y, 1)
    xline = np.linspace(x.min() - 0.05, x.max() + 0.05, 100)
    ax.plot(xline, slope * xline + intercept, color="black", lw=1.5, ls="--", alpha=0.7)
    ax.axvline(0, color="grey", lw=0.8, ls=":", alpha=0.5)
    rho, p_rho = spearmanr(x, y)
    pstr = "p<0.001" if p_rho < 0.001 else ("p<0.05" if p_rho < 0.05 else f"p={p_rho:.3f}")
    ax.text(0.03, 0.97, f"rho = {rho:.2f}  ({pstr})\nN = {len(x):,}",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="square,pad=0.3", fc="none", ec="none"))
    ax.set_title(label, loc="left", fontsize=10)
    ax.set_xlabel("Anomalia SSS OISSS (PSU)", fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    print(f"  {label:30s}  N={len(x):>6,}  rho={rho:+.3f}  {pstr}")


def make_figure(sub, comp_colors, reg_label):
    fig, axes = plt.subplots(1, 5, figsize=(22, 6))

    point_colors = [comp_colors[c] for c in sub["company"]]
    draw_scatter(axes[0],
                 sub["oisss_sss_anom"].values,
                 sub["log_catch"].values,
                 colors=point_colors, s=4, alpha=0.4,
                 label="Calas individuales",
                 ylabel="log(captura)")

    for ax, (label, tcol) in zip(axes[1:], AGG_LEVELS):
        agg = sub.groupby(["company", tcol]).agg(
            total_catch=("catch_tm", "sum"),
            mean_sss_anom=("oisss_sss_anom", "mean"),
        ).reset_index()
        agg = agg.dropna(subset=["mean_sss_anom"])
        point_colors_agg = [comp_colors[c] for c in agg["company"]]
        draw_scatter(ax,
                     agg["mean_sss_anom"].values,
                     np.log(agg["total_catch"].values),
                     colors=point_colors_agg, s=18, alpha=0.7,
                     label=f"Empresa x {label}")

    handles = [mpatches.Patch(color=comp_colors[c], label=c)
               for c in sorted(sub["company"].unique())]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"log(captura) ~ SSS OISSS  |  empresa x tiempo  |  Region {reg_label}  |  2015-2024",
        fontsize=11, x=0.01, ha="left")
    plt.tight_layout(rect=[0, 0.08, 1, 0.97])
    return fig


def main():
    outputs = {reg: PLOTS / f"13b_analysis_sss_catch_scatter_{reg}.png"
               for reg, _, _ in REGIONS}

    if all(p.exists() for p in outputs.values()):
        print("13b outputs exist -- skipping")
        return

    df = pd.read_csv(OUTPUTS / "calas_enriched.csv", low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["oisss_sss_anom", "catch_tm", "lat", "lon", "season", "company"])
    df = df[df["catch_tm"] > 0]
    df["log_catch"] = np.log(df["catch_tm"])

    iso = df["date"].dt.isocalendar()
    df["year_week"]  = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    companies   = sorted(df["company"].unique())
    comp_colors = {c: plt.cm.tab20.colors[i % 20] for i, c in enumerate(companies)}

    for reg, lat_min, lat_max in REGIONS:
        out = outputs[reg]
        if out.exists():
            print(f"{out.name} exists -- skipping")
            continue

        sub = df[df["lat"] > lat_min].copy()
        if lat_max is not None:
            sub = sub[sub["lat"] <= lat_max]

        reg_label = reg.replace("_", " ").title()
        print(f"\n--- {reg_label} ---")
        fig = make_figure(sub, comp_colors, reg_label)
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
