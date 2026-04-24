"""
Step 18 - Company-level parametric insurance analysis: COPEINCA and EXALMAR

For each company:
  1. OLS regression at empresa x temporada level: log(catch) ~ SST_anom.
     Company-specific beta compared to the area-wide product beta.
  2. Historical payout simulation using the area-wide Centro Norte SST trigger
     and the company's own baseline catch (mean historical seasonal catch).
  3. SST anomaly time series per season showing when the trigger fires.

Payout formula (linear ramp, same as step 15):
  payout_fraction = clip((SST_anom - ENTRY) / (EXIT - ENTRY), 0, 1)
  payout_tons     = baseline_catch * payout_fraction

Inputs:
  OUTPUTS/calas_enriched.csv

Outputs:
  PLOTS/step18_client_copeinca_exalmar.png

Skip logic: skipped if output already exists.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.formula.api as smf

from config import OUTPUTS, PLOTS

# ── company registry ─────────────────────────────────────────────────────────
# Values are case-insensitive regex patterns that match all known name variants:
#   COPEINCA: CFG-COPEINCA, CFG/COPEINCA, COPEINCA SA, copeinca, ...
#   EXALMAR:  PESQUERA EXALMAR S.A.A., EXALMAR-CENTINELA, EXALMAR, ...
COMPANIES = {
    "COPEINCA": r"copeinca",
    "EXALMAR":  r"exalmar",
    "DIAMANTE": r"diamante",
}

# ── product parameters (from step 15) ─────────────────────────────────────────
ENTRY_SST   =  0.5
EXIT_SST    =  2.5
BETA_MARKET = -0.816   # Centro Norte, empresa x temporada, M1

# ── trigger zone (Centro Norte) ───────────────────────────────────────────────
TRIGGER_LAT_S = -11.0
TRIGGER_LAT_N =  -7.1


# ── helpers ───────────────────────────────────────────────────────────────────
def season_sort_key(s):
    """Return (year, half) so seasons sort chronologically."""
    s = str(s).strip()
    if s.startswith("1ra"):
        return (int(s.split()[-1]), 0)
    if s.startswith("2da"):
        return (int(s.split()[-1]), 1)
    # e.g. "2024_CN-II"
    year = int(s.split("_")[0])
    half = 1 if "II" in s else 0
    return (year, half)


def season_label(s):
    s = str(s).strip()
    if s.startswith("1ra"):
        return f"T1-{s.split()[-1]}"
    if s.startswith("2da"):
        return f"T2-{s.split()[-1]}"
    year = s.split("_")[0]
    suffix = "T2" if "II" in s else "T1"
    return f"{suffix}-{year}"


def payout_fraction(sst):
    return float(np.clip((sst - ENTRY_SST) / (EXIT_SST - ENTRY_SST), 0.0, 1.0))


# ── data preparation ──────────────────────────────────────────────────────────
def prepare_data():
    df = pd.read_csv(OUTPUTS / "calas_enriched.csv", low_memory=False)
    rename = {
        "temporada": "season", "declarado_tm": "catch_tm",
        "latitud": "lat",      "longitud": "lon",
        "modis_sst_anomaly": "modis_sst_anom",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df = df[df["catch_tm"] > 0].dropna(
        subset=["modis_sst_anom", "catch_tm", "lat", "season", "company"])

    # Area-wide SST trigger: mean across ALL companies in Centro Norte
    cn = df[(df["lat"] > TRIGGER_LAT_S) & (df["lat"] <= TRIGGER_LAT_N)]
    sst_trigger = (cn.groupby("season")["modis_sst_anom"]
                     .mean()
                     .rename("sst_trigger"))

    results = {}
    for short, pattern in COMPANIES.items():
        mask = df["company"].str.contains(pattern, case=False, na=False)
        sub  = df[mask]
        variants = sub["company"].unique()
        print(f"  {short}: {len(sub):,} calas en {len(variants)} variante(s): {list(variants)}")

        # empresa x temporada aggregation (all variants merged)
        agg = (sub.groupby("season")
                  .agg(catch_kt=("catch_tm", lambda x: x.sum() / 1000),
                       sst_own=("modis_sst_anom", "mean"),
                       n_calas=("catch_tm", "count"))
                  .reset_index())

        # join area-wide SST trigger
        agg = agg.join(sst_trigger, on="season")

        # sort chronologically
        agg["_key"] = agg["season"].apply(season_sort_key)
        agg = agg.sort_values("_key").drop(columns="_key").reset_index(drop=True)
        agg["label"] = agg["season"].apply(season_label)

        # season type (T1 / T2)
        agg["stype"] = agg["season"].apply(
            lambda s: "T1" if season_sort_key(s)[1] == 0 else "T2")

        # baseline: mean catch per season type
        baseline = agg.groupby("stype")["catch_kt"].mean()
        agg["baseline_kt"] = agg["stype"].map(baseline)

        # payout simulation
        agg["payout_frac"]  = agg["sst_trigger"].apply(payout_fraction)
        agg["payout_kt"]    = agg["baseline_kt"] * agg["payout_frac"]
        agg["loss_kt"]      = (agg["baseline_kt"] - agg["catch_kt"]).clip(lower=0)
        agg["covered_kt"]   = agg[["payout_kt", "loss_kt"]].min(axis=1)
        agg["uncovered_kt"] = (agg["loss_kt"] - agg["payout_kt"]).clip(lower=0)

        # OLS: log(catch_kt) ~ sst_trigger (company-level)
        ols_df = agg.dropna(subset=["catch_kt", "sst_trigger"])
        ols_df = ols_df[ols_df["catch_kt"] > 0]
        m = smf.ols("np.log(catch_kt) ~ sst_trigger", data=ols_df).fit()
        beta_co = m.params["sst_trigger"]
        p_co    = m.pvalues["sst_trigger"]
        r2_co   = m.rsquared

        results[short] = {
            "agg": agg, "model": m,
            "beta": beta_co, "pval": p_co, "r2": r2_co,
        }

    return results


# ── plotting ──────────────────────────────────────────────────────────────────
def plot_company(axes_col, short, data, col_idx):
    """Fill one column of the figure for a given company."""
    ax_sc, ax_ts, ax_sst = axes_col
    agg   = data["agg"]
    model = data["model"]
    beta  = data["beta"]
    pval  = data["pval"]
    r2    = data["r2"]

    c_t1  = "#2166ac"
    c_t2  = "#d6604d"
    warm  = "#c0392b"
    cold  = "#2980b9"

    # ── scatter: log(catch) vs SST trigger ──────────────────────────────────
    for stype, c in [("T1", c_t1), ("T2", c_t2)]:
        sub = agg[agg["stype"] == stype]
        ax_sc.scatter(sub["sst_trigger"], np.log(sub["catch_kt"]),
                      color=c, s=40, zorder=3, label=stype, alpha=0.85)
        # annotate outliers
        for _, row in sub.iterrows():
            if abs(row["sst_trigger"]) > 1.5 or row["catch_kt"] < agg["catch_kt"].quantile(0.15):
                ax_sc.annotate(row["label"],
                               xy=(row["sst_trigger"], np.log(row["catch_kt"])),
                               fontsize=6, color="#333333",
                               xytext=(4, 0), textcoords="offset points")

    # company OLS line
    x_line = np.linspace(agg["sst_trigger"].min() - 0.1,
                         agg["sst_trigger"].max() + 0.1, 100)
    y_line = model.params["Intercept"] + beta * x_line
    ax_sc.plot(x_line, y_line, color="black", lw=1.5, ls="--",
               label=f"OLS empresa: beta={beta:+.3f}")

    # market beta reference line (same intercept, market slope)
    y_mkt = model.params["Intercept"] + BETA_MARKET * x_line
    ax_sc.plot(x_line, y_mkt, color="grey", lw=1.2, ls=":",
               label=f"Beta mercado: {BETA_MARKET:+.3f}")

    pstr = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
    ax_sc.text(0.03, 0.97,
               f"beta = {beta:+.3f}{pstr}   R2 = {r2:.2f}   N = {len(agg)}",
               transform=ax_sc.transAxes, va="top", fontsize=7.5,
               bbox=dict(boxstyle="square,pad=0.3", fc="none", ec="none"))
    ax_sc.axvline(0, color="grey", lw=0.7, ls=":", alpha=0.5)
    ax_sc.axvline(ENTRY_SST, color="#f39c12", lw=0.8, ls="--", alpha=0.6)
    ax_sc.set_xlabel("SST anomaly estacional - Centro Norte (deg C)", fontsize=8)
    ax_sc.set_ylabel("log(captura empresa kt)", fontsize=8)
    ax_sc.set_title(short, fontsize=11, pad=6)
    ax_sc.legend(fontsize=7, frameon=False, loc="lower left")
    ax_sc.spines["top"].set_visible(False)
    ax_sc.spines["right"].set_visible(False)

    # ── time series: catch + payout simulation ───────────────────────────────
    x     = np.arange(len(agg))
    w     = 0.6
    bar_c = [warm if row["sst_trigger"] > ENTRY_SST else cold
             for _, row in agg.iterrows()]

    bars = ax_ts.bar(x, agg["catch_kt"], width=w, color=bar_c, alpha=0.75,
                     zorder=2, label="Captura real (kt)")

    # payout: stacked on top of actual catch
    payout_mask = agg["payout_kt"] > 0
    ax_ts.bar(x[payout_mask], agg.loc[payout_mask, "payout_kt"],
              bottom=agg.loc[payout_mask, "catch_kt"],
              width=w, color="#f39c12", alpha=0.85, zorder=3,
              hatch="///", edgecolor="white", linewidth=0.4,
              label="Pago seguro (kt)")

    # baseline lines per season type
    for stype, ls_ in [("T1", "-"), ("T2", "--")]:
        sub = agg[agg["stype"] == stype]
        if sub.empty:
            continue
        base = sub["baseline_kt"].iloc[0]
        idx  = sub.index
        for i in range(len(idx)):
            xi = x[idx[i]]
            xj = x[idx[i + 1]] if i + 1 < len(idx) else xi + 1
            ax_ts.hlines(base, xi - w / 2, xi + w / 2 - 0.01,
                         colors="#333333", linewidth=1.5, zorder=4)

    ax_ts.set_xticks(x)
    ax_ts.set_xticklabels(agg["label"], rotation=45, ha="right", fontsize=7)
    ax_ts.set_ylabel("Captura (kt)", fontsize=8)
    ax_ts.set_title("Captura real + pago simulado vs linea base", fontsize=9)
    ax_ts.legend(fontsize=7, frameon=False, loc="upper left")
    ax_ts.spines["top"].set_visible(False)
    ax_ts.spines["right"].set_visible(False)

    # ── SST trigger time series ───────────────────────────────────────────────
    sst_c = [warm if v > 0 else cold for v in agg["sst_trigger"]]
    ax_sst.bar(x, agg["sst_trigger"], color=sst_c, alpha=0.80, width=w, zorder=2)
    ax_sst.axhline(0,         color="black", lw=0.6, ls="-",  alpha=0.5)
    ax_sst.axhline(ENTRY_SST, color="#f39c12", lw=1.2, ls="--",
                   label=f"Entry {ENTRY_SST} deg C")
    ax_sst.axhline(EXIT_SST,  color="#c0392b", lw=1.2, ls="--",
                   label=f"Exit {EXIT_SST} deg C")
    ax_sst.set_xticks(x)
    ax_sst.set_xticklabels(agg["label"], rotation=45, ha="right", fontsize=7)
    ax_sst.set_ylabel("SST anomaly (deg C)\nCentro Norte - area-wide", fontsize=8)
    ax_sst.set_title("Trigger SST estacional (zona Centro Norte)", fontsize=9)
    ax_sst.legend(fontsize=7, frameon=False, loc="upper right")
    ax_sst.spines["top"].set_visible(False)
    ax_sst.spines["right"].set_visible(False)


def print_summary(results):
    for short, data in results.items():
        agg = data["agg"]
        print(f"\n{'='*70}")
        print(f"  {short}  |  beta={data['beta']:+.3f}  R2={data['r2']:.2f}  p={data['pval']:.4f}")
        print(f"{'='*70}")
        print(f"{'Temporada':<14} {'Captura_kt':>10} {'Base_kt':>10} "
              f"{'SST_trigger':>12} {'Pago_frac':>10} {'Pago_kt':>10} "
              f"{'Perdida_kt':>11} {'Cobertura%':>11}")
        for _, row in agg.iterrows():
            cov = (row["covered_kt"] / row["loss_kt"] * 100
                   if row["loss_kt"] > 0 else float("nan"))
            print(f"{row['label']:<14} {row['catch_kt']:>10.0f} {row['baseline_kt']:>10.0f} "
                  f"{row['sst_trigger']:>12.2f} {row['payout_frac']:>10.2f} {row['payout_kt']:>10.0f} "
                  f"{row['loss_kt']:>11.0f} {cov:>10.0f}%"
                  if not np.isnan(cov)
                  else
                  f"{row['label']:<14} {row['catch_kt']:>10.0f} {row['baseline_kt']:>10.0f} "
                  f"{row['sst_trigger']:>12.2f} {row['payout_frac']:>10.2f} {row['payout_kt']:>10.0f} "
                  f"{row['loss_kt']:>11.0f} {'--':>10}")

        triggered = agg[agg["payout_frac"] > 0]
        print(f"\n  Temporadas con pago: {len(triggered)}/{len(agg)}")
        print(f"  Pago promedio cuando activa: {triggered['payout_kt'].mean():.0f} kt")
        print(f"  Max perdida historica:        {agg['loss_kt'].max():.0f} kt  "
              f"({agg.loc[agg['loss_kt'].idxmax(),'label']})")
        print(f"  Max pago historico simulado:  {agg['payout_kt'].max():.0f} kt  "
              f"({agg.loc[agg['payout_kt'].idxmax(),'label']})")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    outpath = PLOTS / "step18_client_copeinca_exalmar.png"

    if outpath.exists():
        print("step18 output exists -- skipping")
        return

    print("Preparing company data...")
    results = prepare_data()

    print_summary(results)

    print("\nGenerating figure...")
    n_co  = len(COMPANIES)
    fig, axes = plt.subplots(
        3, n_co,
        figsize=(n_co * 8, 14),
        gridspec_kw={"hspace": 0.55, "wspace": 0.35},
    )

    for col, short in enumerate(COMPANIES):
        plot_company(axes[:, col], short, results[short], col)

    fig.suptitle(
        "Analisis de producto de seguro parametrico por empresa  |  "
        "Trigger: anomalia SST estacional Centro Norte  |  "
        f"Entry={ENTRY_SST} deg C  Exit={EXIT_SST} deg C  Beta mercado={BETA_MARKET}",
        fontsize=11, y=1.01, x=0.01, ha="left",
    )

    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {outpath}")


def plot_company_nosc(axes_col, short, data):
    """Two-row column: catch + payout timeseries and SST trigger (no scatter)."""
    ax_ts, ax_sst = axes_col
    agg = data["agg"]

    c_t1 = "#2166ac"
    c_t2 = "#d6604d"
    warm = "#c0392b"
    cold = "#2980b9"

    x = np.arange(len(agg))
    w = 0.6
    bar_c = [warm if row["sst_trigger"] > ENTRY_SST else cold
             for _, row in agg.iterrows()]

    ax_ts.bar(x, agg["catch_kt"], width=w, color=bar_c, alpha=0.75,
              zorder=2, label="Captura real (kt)")

    payout_mask = agg["payout_kt"] > 0
    ax_ts.bar(x[payout_mask], agg.loc[payout_mask, "payout_kt"],
              bottom=agg.loc[payout_mask, "catch_kt"],
              width=w, color="#f39c12", alpha=0.85, zorder=3,
              hatch="///", edgecolor="white", linewidth=0.4,
              label="Pago seguro (kt)")

    for stype in ["T1", "T2"]:
        sub = agg[agg["stype"] == stype]
        if sub.empty:
            continue
        base = sub["baseline_kt"].iloc[0]
        for i, idx in enumerate(sub.index):
            xi = x[idx]
            ax_ts.hlines(base, xi - w / 2, xi + w / 2 - 0.01,
                         colors="#333333", linewidth=1.5, zorder=4)

    ax_ts.set_xticks(x)
    ax_ts.set_xticklabels(agg["label"], rotation=45, ha="right", fontsize=7)
    ax_ts.set_ylabel("Captura (kt)", fontsize=8)
    ax_ts.set_title(f"{short} - Captura real + pago simulado vs linea base", fontsize=9)
    ax_ts.legend(fontsize=7, frameon=False, loc="upper left")
    ax_ts.spines["top"].set_visible(False)
    ax_ts.spines["right"].set_visible(False)

    sst_c = [warm if v > 0 else cold for v in agg["sst_trigger"]]
    ax_sst.bar(x, agg["sst_trigger"], color=sst_c, alpha=0.80, width=w, zorder=2)
    ax_sst.axhline(0,         color="black", lw=0.6, ls="-",  alpha=0.5)
    ax_sst.axhline(ENTRY_SST, color="#f39c12", lw=1.2, ls="--",
                   label=f"Entry {ENTRY_SST} deg C")
    ax_sst.axhline(EXIT_SST,  color="#c0392b", lw=1.2, ls="--",
                   label=f"Exit {EXIT_SST} deg C")
    ax_sst.set_xticks(x)
    ax_sst.set_xticklabels(agg["label"], rotation=45, ha="right", fontsize=7)
    ax_sst.set_ylabel("SST anomaly (deg C)\nCentro Norte - area-wide", fontsize=8)
    ax_sst.set_title("Trigger SST estacional (zona Centro Norte)", fontsize=9)
    ax_sst.legend(fontsize=7, frameon=False, loc="upper right")
    ax_sst.spines["top"].set_visible(False)
    ax_sst.spines["right"].set_visible(False)


def main_nosc():
    outpath = PLOTS / "step18_client_copeinca_exalmar_nosc.png"
    print("Preparing company data...")
    results = prepare_data()

    n_co = len(COMPANIES)
    fig, axes = plt.subplots(
        2, n_co,
        figsize=(n_co * 8, 9),
        gridspec_kw={"hspace": 0.55, "wspace": 0.35},
    )

    for col, short in enumerate(COMPANIES):
        plot_company_nosc(axes[:, col], short, results[col if isinstance(col, str) else short])

    fig.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {outpath}")


if __name__ == "__main__":
    main()
    main_nosc()
