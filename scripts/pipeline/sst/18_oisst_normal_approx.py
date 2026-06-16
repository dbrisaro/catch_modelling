"""
18_oisst_normal_approx.py

Fits Normal(mu, sigma) distributions to OISST T1 and T2 seasonal anomaly
means per region, and produces two figures:

  18_oisst_normal_approx.png      — unconditional normal fit + histogram
  18b_oisst_enso_conditioned.png  — normals conditioned on ENSO phase

ENSO phase is derived from the ONI index (NOAA CPC):
  T1 (Apr-Jul) classified by AMJ ONI
  T2 (Nov-Dec) classified by OND ONI
  Threshold: El Niño >= +0.5, La Niña <= -0.5, else Neutral

Input  : FEATURES/oisst_seasonal_means_clim2005_2024.csv
         (produced by 17_oisst_seasonal_distribution.py)
Outputs: outputs/sst/18_oisst_normal_approx.png
         outputs/sst/18b_oisst_enso_conditioned.png
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FEATURES, PLOTS

OUT_DIR  = PLOTS / "sst"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IN_CSV   = FEATURES / "oisst_seasonal_means_clim2005_2024.csv"
OUT_FIG  = OUT_DIR / "18_oisst_normal_approx.png"
OUT_FIG2 = OUT_DIR / "18b_oisst_enso_conditioned.png"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REGIONS    = ["Norte", "Centro Norte", "Centro Sur"]
COL_T1     = "#2166ac"
COL_T2     = "#e07b39"
EVENT_YEARS = {2015: "2015-16", 2023: "2023-24"}

ENSO_COLORS = {
    "El Niño": "#d62728",
    "La Niña": "#1f77b4",
    "Neutral":  "#7f7f7f",
}
ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
ONI_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# ENSO classification
# ---------------------------------------------------------------------------

def load_oni():
    """
    Download ONI from NOAA CPC and return a DataFrame with columns:
    season, year, oni.
    Format: SEAS YR TOTAL CLIM ANOM
    """
    print("Downloading ONI from NOAA CPC...")
    with urllib.request.urlopen(ONI_URL, timeout=30) as r:
        raw = r.read().decode("utf-8")

    rows = []
    for line in raw.strip().split("\n"):
        parts = line.split()
        if len(parts) < 4 or not parts[1].isdigit():
            continue
        try:
            rows.append({"season": parts[0],
                         "year":   int(parts[1]),
                         "oni":    float(parts[3])})
        except (ValueError, IndexError):
            continue

    df = pd.DataFrame(rows)
    print(f"  ONI loaded: {df['year'].min()}-{df['year'].max()}, "
          f"{df['season'].nunique()} seasons")
    return df


def classify_enso(oni_val):
    if oni_val >= ONI_THRESHOLD:
        return "El Niño"
    if oni_val <= -ONI_THRESHOLD:
        return "La Niña"
    return "Neutral"


def build_enso_labels(oni_df, years):
    """
    For each year return the ENSO phase for T1 (AMJ) and T2 (OND).
    Returns dict {year: {"t1": phase, "t2": phase}}.
    """
    labels = {}
    for yr in years:
        amj = oni_df[(oni_df["year"] == yr) & (oni_df["season"] == "AMJ")]
        ond = oni_df[(oni_df["year"] == yr) & (oni_df["season"] == "OND")]
        labels[yr] = {
            "t1": classify_enso(float(amj["oni"].iloc[0])) if not amj.empty else "Neutral",
            "t2": classify_enso(float(ond["oni"].iloc[0])) if not ond.empty else "Neutral",
        }
    return labels


# ---------------------------------------------------------------------------
# Figure 1 — unconditional normal approximation
# ---------------------------------------------------------------------------

def plot_normal_approx(df, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    print(f"\n{'Region':<16} {'Season':<6}  {'mu':>6}  {'sigma':>6}  "
          f"{'p90':>6}  {'p95':>6}  {'p99':>6}  {'N':>4}")
    print("-" * 72)

    for ax, region in zip(axes, REGIONS):
        sub       = df[df["region"] == region].sort_values("year")
        years     = sub["year"].values
        trans     = ax.get_xaxis_transform()

        all_vals  = np.concatenate([sub["t1"].values, sub["t2"].values])
        xgrid     = np.linspace(all_vals.min() - 0.5, all_vals.max() + 0.5, 400)

        for col, color, label in [
            ("t1", COL_T1, "T1 (Apr-Jul)"),
            ("t2", COL_T2, "T2 (Nov-Dec)"),
        ]:
            vals  = sub[col].values
            mu, sigma = float(vals.mean()), float(vals.std(ddof=1))
            p90   = stats.norm.ppf(0.90, mu, sigma)
            p95   = stats.norm.ppf(0.95, mu, sigma)
            p99   = stats.norm.ppf(0.99, mu, sigma)

            print(f"{region:<16} {col.upper():<6}  {mu:>6.3f}  {sigma:>6.3f}  "
                  f"{p90:>6.3f}  {p95:>6.3f}  {p99:>6.3f}  {len(vals):>4}")

            ax.hist(vals, bins=10, density=True, color=color, alpha=0.20, edgecolor="none")
            pdf = stats.norm.pdf(xgrid, mu, sigma)
            ax.plot(xgrid, pdf, color=color, lw=2.2, label=label)
            ax.fill_between(xgrid, 0, pdf,
                            where=(xgrid >= mu - sigma) & (xgrid <= mu + sigma),
                            color=color, alpha=0.10)
            ax.axvline(mu, color=color, lw=0.8, alpha=0.45)

            for pct, val, ls, lw, ya in [(90, p90, "--", 1.2, 0.94),
                                          (95, p95, ":",  1.4, 0.78)]:
                ax.axvline(val, color=color, ls=ls, lw=lw, alpha=0.75)
                ax.text(val - 0.06, ya, f"p{pct}", fontsize=6.5, color=color,
                        rotation=90, ha="right", va="top", transform=trans)

            for v in vals:
                ax.plot(v, -0.03, "|", color=color, alpha=0.55, ms=5,
                        transform=trans, clip_on=False)

        year_list = list(years)
        for yr, lbl in EVENT_YEARS.items():
            if yr in year_list:
                xval = sub["t1"].values[year_list.index(yr)]
                ax.text(xval, 0.10, lbl, fontsize=6.5, color=COL_T1,
                        ha="center", va="bottom", transform=trans,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

        ax.set_title(region, fontsize=10)
        ax.set_xlabel("SST anomaly (°C)", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("density", fontsize=9)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axvline(0, color="#999999", lw=0.8, alpha=0.5)

    n = df["year"].nunique()
    fig.suptitle(
        f"OISST seasonal anomaly — normal approximation  |  clim 2005-2024  |  "
        f"N={n} years ({df['year'].min()}-{df['year'].max()})  |  "
        f"shaded = mu ± sigma  |  dashed=p90, dotted=p95",
        fontsize=8.5, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 — ENSO-conditioned normals
# ---------------------------------------------------------------------------

def plot_enso_conditioned(df, enso_labels, out_path):
    """
    2 rows (T1, T2) x 3 cols (regions).
    Each panel: normal PDFs for El Niño / La Niña / Neutral.
    The unconditional fit is shown in dashed gray for reference.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)

    print(f"\n{'Region':<16} {'Season':<6} {'Phase':<10}  "
          f"{'mu':>6}  {'sigma':>6}  {'p90':>6}  {'p95':>6}  {'N':>4}")
    print("-" * 70)

    for col_i, region in enumerate(REGIONS):
        sub       = df[df["region"] == region].sort_values("year")
        years     = sub["year"].values

        for row_i, (season_col, season_label) in enumerate([
            ("t1", "T1  Apr-Jul"),
            ("t2", "T2  Nov-Dec"),
        ]):
            ax    = axes[row_i, col_i]
            trans = ax.get_xaxis_transform()
            vals_all = sub[season_col].values

            xgrid = np.linspace(vals_all.min() - 0.5, vals_all.max() + 0.5, 400)

            # unconditional reference
            mu_all, sig_all = vals_all.mean(), vals_all.std(ddof=1)
            ax.plot(xgrid, stats.norm.pdf(xgrid, mu_all, sig_all),
                    color="#aaaaaa", lw=1.2, ls="--", label="All years", zorder=1)

            for phase, color in ENSO_COLORS.items():
                phase_years = [yr for yr, v in enso_labels.items()
                               if v[season_col] == phase]
                mask  = np.isin(years, phase_years)
                vals_p = vals_all[mask]
                n_p    = mask.sum()

                if n_p < 4:
                    continue

                mu_p, sig_p = vals_p.mean(), vals_p.std(ddof=1)
                p90_p = stats.norm.ppf(0.90, mu_p, sig_p)
                p95_p = stats.norm.ppf(0.95, mu_p, sig_p)

                print(f"{region:<16} {season_col.upper():<6} {phase:<10}  "
                      f"{mu_p:>6.3f}  {sig_p:>6.3f}  {p90_p:>6.3f}  {p95_p:>6.3f}  {n_p:>4}")

                pdf_p = stats.norm.pdf(xgrid, mu_p, sig_p)
                ax.plot(xgrid, pdf_p, color=color, lw=2.0,
                        label=f"{phase} (N={n_p})", zorder=3)
                ax.fill_between(xgrid, 0, pdf_p, color=color, alpha=0.08, zorder=2)

                # p90 line only (keep it clean)
                ax.axvline(p90_p, color=color, ls="--", lw=1.0, alpha=0.65, zorder=3)

                # rug
                for v in vals_p:
                    ax.plot(v, -0.03, "|", color=color, alpha=0.7, ms=5,
                            transform=trans, clip_on=False)

            ax.axvline(0, color="#cccccc", lw=0.8)
            ax.set_xlabel("SST anomaly (°C)", fontsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(fontsize=7.5, frameon=False, loc="upper right")

            if col_i == 0:
                ax.set_ylabel(f"{season_label}\n\ndensity", fontsize=9)
            if row_i == 0:
                ax.set_title(region, fontsize=10)

    fig.suptitle(
        "OISST seasonal anomaly — ENSO-conditioned normal distributions  |  "
        "clim 2005-2024  |  "
        "ONI ±0.5°C  (T1→AMJ, T2→OND)  |  dashed = p90  |  gray = unconditional",
        fontsize=8.5, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(
            f"{IN_CSV} not found — run 17_oisst_seasonal_distribution.py first"
        )
    df = pd.read_csv(IN_CSV)
    print(f"Loaded {len(df)} rows from {IN_CSV.name}")

    plot_normal_approx(df, OUT_FIG)

    oni_df      = load_oni()
    years_all   = sorted(df["year"].unique())
    enso_labels = build_enso_labels(oni_df, years_all)

    # print classification summary
    for season_col, oni_season in [("t1", "AMJ"), ("t2", "OND")]:
        counts = {}
        for v in enso_labels.values():
            counts[v[season_col]] = counts.get(v[season_col], 0) + 1
        parts = "  ".join(f"{ph}={n}" for ph, n in sorted(counts.items()))
        print(f"\nENSO classification ({oni_season}): {parts}")

    plot_enso_conditioned(df, enso_labels, OUT_FIG2)


if __name__ == "__main__":
    main()
