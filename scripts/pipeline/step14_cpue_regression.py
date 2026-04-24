"""
Step 14 - CPUE Regression: log(catch / effort) ~ SST anomaly

Effort is measured from SISESAT VMS data (available 2017-2024, not all
seasons). Each VMS ping represents ~9 minutes of vessel activity; total
effort per vessel per season = n_pings x (9/60) hours.

Vessel names in SISESAT (e.g. 'T17') are normalized to match calas
format (e.g. 'TASA 17') via a rule-based lookup.

Aggregation: empresa x temporada (M1 seasonal, same as step 11)
Region focus: Centro (-15.8 < lat <= -7.1)
SST anomaly: mean modis_sst_anom across calas for that empresa x season

Comparison output:
  - Step 11 beta: log(catch) ~ SST_anom  (empresa x temporada, Centro)
  - Step 14 beta: log(CPUE)  ~ SST_anom  (same aggregation, SISESAT years only)

Outputs:
  PLOTS/step14_cpue_betas.png

Skip logic: skipped if PLOTS/step14_cpue_betas.png already exists.
"""
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import statsmodels.formula.api as smf

from config import OUTPUTS, PLOTS, INPUTS

# ── constants ────────────────────────────────────────────────────────────────
LAT_MIN_CENTRO = -15.8
LAT_MAX_CENTRO = -7.1
PING_HOURS = 9 / 60        # each VMS ping ≈ 9 minutes
BETA_STEP11 = -0.849       # reference beta from step 11 M1 empresa x temporada Centro
IHMA_DIR = INPUTS / "ihma_data"

# Season DOY windows (same as rest of pipeline)
T1_DOY = (91, 212)    # Apr-Jul
T2_DOY = (305, 365)   # Nov-Dec

# Minimum days of VMS coverage to include a season file
MIN_DAYS_COVERAGE = 30


# ── SISESAT file catalogue ───────────────────────────────────────────────────
def build_sisesat_catalogue():
    """Return list of dicts {season, file} for all available anchoveta SISESAT files."""
    catalogue = []
    for year in range(2015, 2026):
        ydir = IHMA_DIR / str(year)
        if not ydir.exists():
            continue
        for fname in os.listdir(ydir):
            if "sisesat" not in fname.lower():
                continue
            if "anchoveta" not in fname.lower():
                continue
            if not fname.endswith(".csv"):
                continue
            flower = fname.lower()
            if "primera" in flower:
                season_key = f"1ra {year}"
            elif "segunda" in flower:
                season_key = f"2da {year}"
            else:
                continue
            catalogue.append({"season": season_key, "file": str(ydir / fname)})
    return catalogue


# ── vessel name normalisation ────────────────────────────────────────────────
def normalize_vessel(name: str) -> str:
    """Normalise vessel name for matching between calas and SISESAT.

    Rules applied (in order):
    1. Strip whitespace and upper-case.
    2. Replace non-breaking spaces with regular spaces.
    3. SISESAT codes 'T<num>' -> 'TASA <num>'  (e.g. 'T17' -> 'TASA 17').
    """
    if not isinstance(name, str):
        return ""
    name = name.replace("\xa0", " ").strip().upper()
    # TASA vessels: SISESAT uses 'T17', calas uses 'TASA 17'
    import re
    m = re.fullmatch(r"T(\d+)", name)
    if m:
        name = f"TASA {m.group(1)}"
    return name


# ── load effort from SISESAT ─────────────────────────────────────────────────
def load_sisesat_effort(catalogue):
    """Return DataFrame: vessel_norm, season, effort_hours.

    Filters each file to pings that fall within the correct season's DOY window
    AND the correct calendar year (from the season key), to handle files that
    span multiple seasons or contain corrupt/future timestamps.
    Seasons with fewer than MIN_DAYS_COVERAGE of VMS data are dropped.
    """
    records = []
    for entry in catalogue:
        season = entry["season"]
        fpath  = entry["file"]
        # Parse season key: "1ra 2020" or "2da 2020"
        tempo, yr_str = season.split()
        year = int(yr_str)
        doy_start, doy_end = T1_DOY if tempo == "1ra" else T2_DOY

        try:
            df = pd.read_csv(fpath, usecols=["Cod_Barco", "Date"], low_memory=False)
        except Exception as exc:
            print(f"  WARNING: could not read {fpath}: {exc}")
            continue
        if df.empty:
            continue

        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date"])

        # Filter: correct calendar year AND season DOY window
        df = df[(df["Date"].dt.year == year) &
                (df["Date"].dt.dayofyear >= doy_start) &
                (df["Date"].dt.dayofyear <= doy_end)]

        if df.empty:
            print(f"  {season}: SKIPPED (no valid pings after date filter)")
            continue

        n_days = (df["Date"].max() - df["Date"].min()).days
        if n_days < MIN_DAYS_COVERAGE:
            print(f"  {season}: SKIPPED (only {n_days} days coverage, need >= {MIN_DAYS_COVERAGE})")
            continue

        # Count pings per vessel
        effort = (
            df["Cod_Barco"]
            .dropna()
            .str.replace("\xa0", " ")
            .str.strip()
            .str.upper()
            .value_counts()
            .reset_index()
        )
        effort.columns = ["vessel_sisesat", "n_pings"]
        effort["vessel_norm"] = effort["vessel_sisesat"].apply(normalize_vessel)
        effort["effort_hours"] = effort["n_pings"] * PING_HOURS
        effort["season"] = season
        records.append(effort[["vessel_norm", "season", "effort_hours"]])
        print(f"  {season}: {len(effort)} vessels, "
              f"{effort['effort_hours'].sum():,.0f} effort-hours, "
              f"{n_days} days coverage")

    if not records:
        return pd.DataFrame(columns=["vessel_norm", "season", "effort_hours"])
    return pd.concat(records, ignore_index=True)


# ── load calas and build lookup tables ───────────────────────────────────────
def load_calas_centro():
    """Return calas_enriched filtered to Centro, with normalised vessel names."""
    df = pd.read_csv(OUTPUTS / "calas_enriched.csv", low_memory=False)
    rename = {
        "fecha_cala": "date", "fecha": "date",
        "temporada": "season", "declarado_tm": "catch_tm",
        "latitud": "lat", "longitud": "lon",
        "modis_sst_anomaly": "modis_sst_anom",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["date"] = pd.to_datetime(df["date"])

    # keep Centro only
    df = df[(df["lat"] > LAT_MIN_CENTRO) & (df["lat"] <= LAT_MAX_CENTRO)]
    df = df[df["catch_tm"] > 0]
    df = df.dropna(subset=["modis_sst_anom", "catch_tm", "season", "company"])

    df["vessel_norm"] = df["vessel"].apply(normalize_vessel)
    return df


# ── OLS helper ───────────────────────────────────────────────────────────────
def run_ols(x, y):
    tmp = pd.DataFrame({"y": y, "x": x}).dropna()
    if len(tmp) < 8:
        return None
    m = smf.ols("y ~ x", data=tmp).fit()
    return {
        "beta": m.params["x"],
        "se":   m.bse["x"],
        "p":    m.pvalues["x"],
        "r2":   m.rsquared,
        "N":    int(m.nobs),
    }


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    out = PLOTS / "step14_cpue_betas.png"
    if out.exists():
        print("step14_cpue_betas.png exists -- skipping")
        return

    # 1. Effort data
    catalogue = build_sisesat_catalogue()
    print(f"\nLoading SISESAT effort data ({len(catalogue)} season files)...")
    effort_df = load_sisesat_effort(catalogue)

    # Aggregate to vessel_norm x season (sum across duplicated vessel codes)
    effort_agg = effort_df.groupby(["vessel_norm", "season"])["effort_hours"].sum().reset_index()

    # 2. Calas data (Centro)
    print("\nLoading calas enriched (Centro)...")
    calas = load_calas_centro()

    # vessel -> company lookup (one vessel can map to one company; take mode)
    vc_lookup = (
        calas.groupby("vessel_norm")["company"]
        .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else None)
        .reset_index()
        .rename(columns={"company": "company_lookup"})
    )

    # calas aggregated to empresa x season
    calas_agg = (
        calas.groupby(["company", "season"])
        .agg(
            total_catch=("catch_tm", "sum"),
            mean_sst_anom=("modis_sst_anom", "mean"),
            n_calas=("catch_tm", "count"),
        )
        .reset_index()
    )

    # effort aggregated: add company via vessel lookup, then sum to empresa x season
    effort_with_co = effort_agg.merge(vc_lookup, on="vessel_norm", how="left")
    effort_with_co = effort_with_co.dropna(subset=["company_lookup"])
    effort_empresa = (
        effort_with_co.groupby(["company_lookup", "season"])["effort_hours"]
        .sum()
        .reset_index()
        .rename(columns={"company_lookup": "company"})
    )

    # 3. Merge catch + effort
    merged = calas_agg.merge(effort_empresa, on=["company", "season"], how="inner")
    merged = merged[merged["effort_hours"] > 0]
    merged["cpue"] = merged["total_catch"] / merged["effort_hours"]
    merged["log_cpue"] = np.log(merged["cpue"])
    merged["log_catch"] = np.log(merged["total_catch"])

    print(f"\nMerged dataset: {len(merged)} empresa x season observations")
    print(f"Seasons covered: {sorted(merged['season'].unique())}")
    print(f"Companies: {merged['company'].nunique()}")
    print(f"\nCatch range:  {merged['total_catch'].min():.0f} - {merged['total_catch'].max():.0f} tm")
    print(f"Effort range: {merged['effort_hours'].min():.0f} - {merged['effort_hours'].max():.0f} hr")
    print(f"CPUE range:   {merged['cpue'].min():.2f} - {merged['cpue'].max():.2f} tm/hr")

    # 4. Regressions on the matched (SISESAT) dataset
    res_catch = run_ols(merged["mean_sst_anom"], merged["log_catch"])
    res_cpue  = run_ols(merged["mean_sst_anom"], merged["log_cpue"])

    # Also run step11-style regression restricted to same seasons (for fair comparison)
    sisesat_seasons = merged["season"].unique()
    calas_sisesat = calas_agg[calas_agg["season"].isin(sisesat_seasons)].copy()
    calas_sisesat["log_catch"] = np.log(calas_sisesat["total_catch"])
    res_step11_restricted = run_ols(calas_sisesat["mean_sst_anom"], calas_sisesat["log_catch"])

    print("\n--- Regression results ---")
    for label, res in [
        ("Step 11 log(catch) 2015-2024 [reference]", {"beta": BETA_STEP11, "se": None, "p": None, "r2": None, "N": "all"}),
        ("Step 11 log(catch) SISESAT seasons only", res_step11_restricted),
        ("Step 14 log(catch) matched vessels", res_catch),
        ("Step 14 log(CPUE) matched vessels", res_cpue),
    ]:
        if res and res.get("beta") is not None:
            se_str = f"se={res['se']:.3f}" if res.get("se") else "se=n/a"
            p_str  = f"p={res['p']:.4f}"   if res.get("p")  else "p=n/a"
            r2_str = f"R2={res['r2']:.3f}" if res.get("r2") else "R2=n/a"
            print(f"  {label}: beta={res['beta']:+.3f}  {se_str}  {p_str}  {r2_str}  N={res.get('N','?')}")

    # 5. Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel A: Beta comparison (4 bars) ---
    ax = axes[0]
    labels_bar = [
        "Step 11\nlog(catch)\n2015-2024\n(reference)",
        "Step 11\nlog(catch)\nSISESAT seasons\n(2017-2022)",
        "Step 14\nlog(catch)\nmatched vessels\n(2017-2022)",
        "Step 14\nlog(CPUE)\nmatched vessels\n(2017-2022)",
    ]
    betas  = [BETA_STEP11,
              res_step11_restricted["beta"] if res_step11_restricted else np.nan,
              res_catch["beta"] if res_catch else np.nan,
              res_cpue["beta"]  if res_cpue  else np.nan]
    ses    = [0.0,
              res_step11_restricted["se"] if res_step11_restricted else 0.0,
              res_catch["se"] if res_catch else 0.0,
              res_cpue["se"]  if res_cpue  else 0.0]
    pvals  = [None,
              res_step11_restricted["p"] if res_step11_restricted else None,
              res_catch["p"] if res_catch else None,
              res_cpue["p"]  if res_cpue  else None]
    ns     = ["full dataset",
              res_step11_restricted["N"] if res_step11_restricted else "?",
              res_catch["N"] if res_catch else "?",
              res_cpue["N"]  if res_cpue  else "?"]
    colors = ["#2166ac", "#6baed6", "#d6604d", "#1a9641"]
    x_pos  = np.arange(len(labels_bar))

    yvals_for_lim = [b for b in betas if not np.isnan(b)]
    all_se = [s for s in ses if s > 0]
    ymax = max(abs(b) + 1.96 * s for b, s in zip(yvals_for_lim, ([0] + all_se + [0, 0])[:len(yvals_for_lim)])) * 1.3

    for i, (b, se, p, col) in enumerate(zip(betas, ses, pvals, colors)):
        if np.isnan(b):
            continue
        ax.bar(x_pos[i], b, color=col, alpha=0.85, width=0.5, zorder=3)
        if se > 0:
            ax.errorbar(x_pos[i], b, yerr=1.96 * se, fmt="none",
                        color="black", capsize=4, lw=1.2, zorder=4)
        # significance star
        if p is not None:
            pstr = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            ax.text(x_pos[i], b - np.sign(b) * ymax * 0.04,
                    pstr, ha="center", va="top" if b < 0 else "bottom", fontsize=8)

    # beta + N annotations above/below bars
    for i, (b, n) in enumerate(zip(betas, ns)):
        if np.isnan(b):
            continue
        ax.text(x_pos[i], b - np.sign(b) * ymax * 0.12,
                f"{b:+.3f}\n(N={n})", ha="center", va="top" if b < 0 else "bottom", fontsize=7)

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_ylim(-ymax * 1.5, ymax * 0.4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_bar, fontsize=8)
    ax.set_ylabel("Beta (OLS coefficient)", fontsize=10)
    ax.set_title("Beta comparison: catch vs CPUE | Centro", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Panel B: Scatter log(CPUE) vs SST_anom ---
    ax2 = axes[1]
    sc = ax2.scatter(merged["mean_sst_anom"], merged["log_cpue"],
                     c=merged["mean_sst_anom"], cmap="RdBu_r",
                     alpha=0.7, s=45, zorder=3)
    plt.colorbar(sc, ax=ax2, label="SST anomaly (deg C)", shrink=0.8)

    if res_cpue:
        x_line = np.linspace(merged["mean_sst_anom"].min(), merged["mean_sst_anom"].max(), 100)
        intercept = merged["log_cpue"].mean() - res_cpue["beta"] * merged["mean_sst_anom"].mean()
        y_line = res_cpue["beta"] * x_line + intercept
        ax2.plot(x_line, y_line, color="black", lw=1.5, ls="--",
                 label=f"OLS: beta={res_cpue['beta']:.3f}, p={res_cpue['p']:.3f}")

    ax2.axvline(0, color="gray", lw=0.7, ls=":")
    ax2.set_xlabel("Mean SST anomaly by empresa x temporada (deg C)", fontsize=10)
    ax2.set_ylabel("log(CPUE)  =  log(catch / VMS effort-hours)", fontsize=10)
    ax2.set_title("CPUE vs SST anomaly | Centro | empresa x temporada", fontsize=10)
    ax2.legend(frameon=False, fontsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(
        "Step 14: CPUE regression | Centro | VMS effort (9-min pings, 2017-2022 clean seasons)",
        fontsize=10, x=0.01, ha="left"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
