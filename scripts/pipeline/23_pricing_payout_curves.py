"""
21c_payout_curves.py

English version of 16_pricing_payout_curves.png.
Does not modify any existing figure.

Output: outputs/21c_payout_curves.png
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PLOTS

BETA_TRIGGER = -0.519
HIST_QUANTILES = {"p90": 1.116, "p95": 1.752, "p99": 2.715}
ENTRY_SST = HIST_QUANTILES["p90"]
EXIT_SST  = HIST_QUANTILES["p99"]
STEP_TRIGGER = HIST_QUANTILES["p90"]

Q_COLORS = {"p90": "#fdae61", "p95": "#f46d43", "p99": "#d73027"}


def payout_ols(sst):
    return np.where(sst > 0, 1 - np.exp(BETA_TRIGGER * sst), 0.0)

def payout_linear(sst):
    return np.clip((sst - ENTRY_SST) / (EXIT_SST - ENTRY_SST), 0.0, 1.0)

def payout_step(sst):
    return np.where(sst >= STEP_TRIGGER, 1.0, 0.0)


def main():
    sst_range = np.linspace(-0.5, 3.2, 500)
    max_ref  = payout_ols(np.array([EXIT_SST]))[0]
    y_ols    = payout_ols(sst_range) / max_ref
    y_linear = payout_linear(sst_range)
    y_step   = payout_step(sst_range)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    panels = [
        ("Linear ramp",  y_linear, "#2166ac", "Linear ramp"),
        ("Step (binary)", y_step,  "#d6604d", "Step (single trigger)"),
    ]

    for ax, (title, y_curve, col, lbl_curve) in zip(axes, panels):
        ax.plot(sst_range, y_ols, color="#888888", lw=1.8, ls="--",
                label=f"OLS (exponential, beta={BETA_TRIGGER})", zorder=2)
        ax.plot(sst_range, y_curve, color=col, lw=2.5, label=lbl_curve, zorder=3)
        ax.fill_between(sst_range, 0, y_curve, color=col, alpha=0.12)

        for qname, qval in HIST_QUANTILES.items():
            ax.axvline(qval, color=Q_COLORS[qname], lw=1.2, ls=":",
                       label=f"{qname} = {qval}°C")

        ax.axhline(1.0, color="black", lw=0.7, ls="--", alpha=0.4)
        ax.axvline(0,   color="grey",  lw=0.7, ls=":",  alpha=0.5)

        ax.set_xlim(-0.5, 3.2)
        ax.set_ylim(-0.05, 1.25)
        ax.set_xlabel("Seasonal SST anomaly (°C)", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Fraction of maximum payout", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if "ramp" in title.lower():
            ax.annotate(f"p90 = {ENTRY_SST}°C\n(payment starts)",
                        xy=(ENTRY_SST, 0), xytext=(ENTRY_SST + 0.15, 0.18),
                        fontsize=7.5, color="#2166ac",
                        arrowprops=dict(arrowstyle="->", color="#2166ac", lw=0.8))
            ax.annotate(f"p99 = {EXIT_SST}°C\n(maximum payout)",
                        xy=(EXIT_SST, 1.0), xytext=(EXIT_SST - 0.75, 1.10),
                        fontsize=7.5, color="#2166ac",
                        arrowprops=dict(arrowstyle="->", color="#2166ac", lw=0.8))
        else:
            ax.annotate(f"Trigger\n(p90 = {STEP_TRIGGER}°C)",
                        xy=(STEP_TRIGGER, 0.5), xytext=(STEP_TRIGGER + 0.2, 0.6),
                        fontsize=7.5, color="#d6604d",
                        arrowprops=dict(arrowstyle="->", color="#d6604d", lw=0.8))

    plt.tight_layout()
    out = PLOTS / "21c_payout_curves.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
