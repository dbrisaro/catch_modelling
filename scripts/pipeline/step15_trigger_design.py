"""
Step 15 - Trigger Design: spatial scale comparison and payout curves

Part A: Spatial scale comparison
  Runs OLS log(catch) ~ SST_anom at empresa x temporada level for:
    - All (Norte + Centro)
    - Norte only
    - Centro Norte  (-11.0 < lat <= -7.1)
    - Centro Sur    (-15.8 < lat <= -11.0)
    - Centro        (-15.8 < lat <= -7.1)
  Compares beta, R² and residuals to determine the optimal trigger region.

Part B: Payout curves
  Using beta from the recommended region (Centro Norte), plots:
    - OLS-implied (exponential) curve: payout = 1 - exp(beta * SST_anom)
    - Linear ramp (triangle): 0 below entry, linear to max at exit
    - Step design: 0 below trigger, max payout above trigger
  Entry, exit and trigger are calibrated to historical SST quantiles.

Inputs:
  OUTPUTS/calas_enriched.csv

Outputs:
  PLOTS/step15_spatial_comparison.png
  PLOTS/step15_payout_curves.png

Skip logic: skipped if both outputs already exist.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import statsmodels.formula.api as smf
from scipy import stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from config import OUTPUTS, PLOTS

# ── spatial scopes ────────────────────────────────────────────────────────────
SCOPES = [
    ("All\n(Norte+Centro)", -15.8, None),
    ("Norte",               -7.1,  None),
    ("Centro Norte",        -11.0, -7.1),
    ("Centro Sur",          -15.8, -11.0),
    ("Centro",              -15.8, -7.1),
]

# ── payout parameters ─────────────────────────────────────────────────────────
# Calibrated to Centro Norte (highest R²) using historical SST distribution
BETA_TRIGGER  = -0.816   # OLS M1 empresa x temporada, Centro Norte

ENTRY_SST     = 0.5      # deg C: payout begins (linear ramp entry / step trigger)
EXIT_SST      = 2.5      # deg C: maximum payout reached (linear ramp cap)
STEP_TRIGGER  = 0.96     # deg C: p90 of historical seasonal SST distribution

# Historical quantiles from step 13 (Centro seasonal SST distribution)
HIST_QUANTILES = {
    "p90": 0.96,
    "p95": 1.38,
    "p99": 2.75,
}


# ── data loader ───────────────────────────────────────────────────────────────
def load_calas():
    df = pd.read_csv(OUTPUTS / "calas_enriched.csv", low_memory=False)
    rename = {
        "temporada": "season", "declarado_tm": "catch_tm",
        "latitud": "lat", "longitud": "lon",
        "modis_sst_anomaly": "modis_sst_anom",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["catch_tm"] > 0].dropna(
        subset=["modis_sst_anom", "catch_tm", "lat", "season", "company"])
    return df


# ── OLS helper ────────────────────────────────────────────────────────────────
def run_ols_seasonal(sub):
    agg = sub.groupby(["company", "season"]).agg(
        total_catch=("catch_tm", "sum"),
        mean_sst=("modis_sst_anom", "mean"),
    ).reset_index()
    agg = agg[agg["total_catch"] > 0]
    if len(agg) < 10:
        return None, agg
    m = smf.ols("np.log(total_catch) ~ mean_sst", data=agg).fit()
    return m, agg


# ── zone color palette (shared between map and scatters) ──────────────────────
SCOPE_COLORS = {
    "All (Norte+Centro)": "#555555",
    "Norte":              "#d6604d",
    "Centro Norte":       "#2166ac",
    "Centro Sur":         "#4dac26",
    "Centro":             "#92c5de",
}

# lat boundaries for each zone band drawn on the map
ZONE_BANDS = [
    # (label, lat_south, lat_north, color)
    ("Norte",        -7.1,  -5.0,  "#d6604d"),
    ("Centro Norte", -11.0, -7.1,  "#2166ac"),
    ("Centro Sur",   -15.8, -11.0, "#4dac26"),
]


def plot_zone_map(ax):
    """Draw a cartopy map of Peru coast with colored latitude zone bands."""
    ax.set_extent([-82, -74, -17, -4], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,  facecolor="#f0ede8", zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor="#d6eaf8", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":", zorder=3)

    # Shade each zone band using Rectangle patches
    lon_w, lon_e = -82, -74
    for label, lat_s, lat_n, color in ZONE_BANDS:
        rect = mpatches.Rectangle(
            (lon_w, lat_s), lon_e - lon_w, lat_n - lat_s,
            transform=ccrs.PlateCarree(),
            color=color, alpha=0.35, zorder=2, linewidth=0,
        )
        ax.add_patch(rect)
        ax.text(-81.5, (lat_s + lat_n) / 2, label,
                transform=ccrs.PlateCarree(),
                fontsize=7, va="center", color=color,
                fontweight="bold", zorder=4)

    # Dashed latitude boundaries
    for lat in [-7.1, -11.0, -15.8]:
        ax.plot([lon_w, lon_e], [lat, lat],
                transform=ccrs.PlateCarree(),
                color="black", lw=0.8, ls="--", alpha=0.5, zorder=4)
        ax.text(-73.8, lat, f"{abs(lat):.1f}°S",
                transform=ccrs.PlateCarree(),
                fontsize=6.5, va="center", color="#333333", zorder=5)

    ax.gridlines(draw_labels=False, linewidth=0.3, color="grey", alpha=0.4)
    ax.set_title("Zonas espaciales\nevaluadas", fontsize=8, pad=4)


# ── Part A: spatial comparison ────────────────────────────────────────────────
def plot_spatial_comparison(df, outpath):
    n_scopes = len(SCOPES)
    fig = plt.figure(figsize=(22, 5.5))
    # Map gets ~1 unit, each scatter gets ~1 unit
    gs = gridspec.GridSpec(1, n_scopes + 1,
                           wspace=0.35,
                           width_ratios=[1.4] + [1] * n_scopes)

    # ── map panel ──
    ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    plot_zone_map(ax_map)

    r2_vals, beta_vals, pval_vals, scope_labels = [], [], [], []
    colors_scope = [SCOPE_COLORS[n.replace("\n", " ")] for n, _, _ in SCOPES]

    for col, (name, lat_min, lat_max) in enumerate(SCOPES):
        sub = df[df["lat"] > lat_min].copy()
        if lat_max is not None:
            sub = sub[sub["lat"] <= lat_max]

        m, agg = run_ols_seasonal(sub)
        short = name.replace("\n", " ")
        col_c = colors_scope[col]

        ax_sc = fig.add_subplot(gs[0, col + 1])
        ax_sc.scatter(agg["mean_sst"], np.log(agg["total_catch"]),
                      color=col_c, alpha=0.6, s=20, zorder=3)

        if m is not None:
            x_line = np.linspace(agg["mean_sst"].min() - 0.1,
                                 agg["mean_sst"].max() + 0.1, 100)
            ax_sc.plot(x_line,
                       m.params["Intercept"] + m.params["mean_sst"] * x_line,
                       color="black", lw=1.5, ls="--", alpha=0.8)
            beta = m.params["mean_sst"]
            p    = m.pvalues["mean_sst"]
            r2   = m.rsquared
            pstr = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            ax_sc.text(0.04, 0.97,
                       f"beta = {beta:+.3f}{pstr}\nR² = {r2:.3f}\nN = {len(agg):,}",
                       transform=ax_sc.transAxes, va="top", fontsize=8)
            r2_vals.append(r2); beta_vals.append(beta); pval_vals.append(p)
        else:
            r2_vals.append(0); beta_vals.append(0); pval_vals.append(1)

        scope_labels.append(short)
        ax_sc.axvline(0, color="grey", lw=0.7, ls=":", alpha=0.5)
        ax_sc.set_title(short, fontsize=9, color=col_c, fontweight="bold")
        ax_sc.set_xlabel("Anomalia SST promedio (°C)", fontsize=7.5)
        if col == 0:
            ax_sc.set_ylabel("log(captura empresa x temporada)", fontsize=7.5)
        ax_sc.spines["top"].set_visible(False)
        ax_sc.spines["right"].set_visible(False)

    # Print summary table
    print("\nSpatial comparison (empresa x temporada):")
    for label, b, r2, p in zip(scope_labels, beta_vals, r2_vals, pval_vals):
        pstr = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"  {label:25s}  beta={b:+.3f}  R²={r2:.3f}  p={p:.4f} {pstr}")

    plt.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {outpath}")


# ── Part B: payout curves ─────────────────────────────────────────────────────
def payout_ols(sst, beta=BETA_TRIGGER):
    """OLS-implied fractional loss: 1 - exp(beta * SST) for SST > 0, else 0."""
    return np.where(sst > 0, 1 - np.exp(beta * sst), 0.0)


def payout_linear(sst, entry=ENTRY_SST, exit_=EXIT_SST):
    """Linear ramp: 0 below entry, linear 0->1 between entry and exit, capped at 1."""
    return np.clip((sst - entry) / (exit_ - entry), 0.0, 1.0)


def payout_step(sst, trigger=STEP_TRIGGER):
    """Step: 0 below trigger, 1 at or above trigger."""
    return np.where(sst >= trigger, 1.0, 0.0)


def plot_payout_curves(outpath):
    sst_range = np.linspace(-0.5, 3.2, 500)

    # Normalize all curves to the same max (OLS value at EXIT_SST)
    max_ref = payout_ols(EXIT_SST, BETA_TRIGGER)
    y_ols    = payout_ols(sst_range)   / max_ref
    y_linear = payout_linear(sst_range)
    y_step   = payout_step(sst_range)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    q_colors = {"p90": "#fdae61", "p95": "#f46d43", "p99": "#d73027"}

    for ax, (title, y_curve, col, lbl_curve) in zip(axes, [
        ("Ramp lineal", y_linear, "#2166ac", "Ramp lineal"),
        ("Step (binario)", y_step, "#d6604d", "Step (disparador unico)"),
    ]):
        # OLS reference
        ax.plot(sst_range, y_ols, color="#888888", lw=1.8, ls="--",
                label=f"OLS (exponencial, beta={BETA_TRIGGER})", zorder=2)
        # Main curve
        ax.plot(sst_range, y_curve, color=col, lw=2.5, label=lbl_curve, zorder=3)
        # Fill under curve
        ax.fill_between(sst_range, 0, y_curve, color=col, alpha=0.12)

        # Historical quantile reference lines
        for qname, qval in HIST_QUANTILES.items():
            ax.axvline(qval, color=q_colors[qname], lw=1.2, ls=":",
                       label=f"{qname} = {qval}°C")

        ax.axhline(1.0, color="black", lw=0.7, ls="--", alpha=0.4)
        ax.axvline(0,   color="grey",  lw=0.7, ls=":",  alpha=0.5)

        ax.set_xlim(-0.5, 3.2)
        ax.set_ylim(-0.05, 1.25)
        ax.set_xlabel("Anomalia SST estacional (°C)", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Fraccion del pago maximo", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Annotate entry/exit for ramp, trigger for step
        if "Ramp" in title:
            ax.annotate("Entry\n(inicio pago)",
                        xy=(ENTRY_SST, 0), xytext=(ENTRY_SST + 0.15, 0.15),
                        fontsize=7.5, color="#2166ac",
                        arrowprops=dict(arrowstyle="->", color="#2166ac", lw=0.8))
            ax.annotate("Exit\n(pago maximo)",
                        xy=(EXIT_SST, 1.0), xytext=(EXIT_SST - 0.6, 1.08),
                        fontsize=7.5, color="#2166ac",
                        arrowprops=dict(arrowstyle="->", color="#2166ac", lw=0.8))
        else:
            ax.annotate(f"Trigger\n(p90 = {STEP_TRIGGER}°C)",
                        xy=(STEP_TRIGGER, 0.5), xytext=(STEP_TRIGGER + 0.2, 0.6),
                        fontsize=7.5, color="#d6604d",
                        arrowprops=dict(arrowstyle="->", color="#d6604d", lw=0.8))

    fig.suptitle(
        f"Diseno de pago  |  Centro Norte  |  beta = {BETA_TRIGGER}  |  "
        f"Entry = {ENTRY_SST}°C  |  Exit = {EXIT_SST}°C  |  Step trigger = p90 ({STEP_TRIGGER}°C)",
        fontsize=9.5, x=0.01, ha="left")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {outpath}")

    # Print key values
    print("\nPayout values at reference SST anomalies:")
    print(f"  {'SST':>6}  {'OLS':>8}  {'Ramp':>8}  {'Step':>8}")
    for sst in [0.0, 0.5, 0.96, 1.38, 2.0, 2.5, 2.75]:
        ols  = payout_ols(np.array([sst]))[0] / max_ref
        ramp = payout_linear(np.array([sst]))[0]
        step = payout_step(np.array([sst]))[0]
        print(f"  {sst:>6.2f}  {ols:>8.3f}  {ramp:>8.3f}  {step:>8.3f}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    out_spatial = PLOTS / "step15_spatial_comparison.png"
    out_payout  = PLOTS / "step15_payout_curves.png"

    if out_spatial.exists() and out_payout.exists():
        print("step15 outputs exist -- skipping")
        return

    df = load_calas()

    if not out_spatial.exists():
        print("Part A: Spatial comparison...")
        plot_spatial_comparison(df, out_spatial)

    if not out_payout.exists():
        print("\nPart B: Payout curves...")
        plot_payout_curves(out_payout)


if __name__ == "__main__":
    main()
