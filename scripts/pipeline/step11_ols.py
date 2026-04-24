"""
Step 11 - OLS: log(catch) ~ SST  [M1]  vs  log(CPUE) ~ SST  [M2]

Methodology 1 (M1) - sin normalizacion de esfuerzo:
  OLS: log(catch) ~ SST_anom
  Aggregation levels: calas individuales, empresa x diario, semanal, mensual, temporada

Methodology 2 (M2) - normalizado por esfuerzo VMS (SISESAT):
  OLS: log(CPUE) ~ SST_anom   where CPUE = catch / effort_hours
  Effort from SISESAT VMS pings (9-min resolution, 2017-2022 clean seasons)
  Aggregation levels: empresa x diario, semanal, mensual, temporada
  (individual calas level excluded: VMS pings cannot be matched to single hauls)

Regions: Norte (lat > -7.1), Centro (-15.8 < lat <= -7.1)

Inputs:
  OUTPUTS/calas_enriched.csv
  INPUTS/ihma_data/{year}/SISESAT files (anchoveta)

Outputs:
  PLOTS/step11_ols_betas.png

Skip logic: skipped if PLOTS/step11_ols_betas.png already exists.
"""
import warnings
warnings.filterwarnings("ignore")

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from matplotlib.patches import Patch

from config import OUTPUTS, PLOTS, INPUTS

# ── constants ─────────────────────────────────────────────────────────────────
REGIONS = [
    ("Norte",        -7.1,  None),
    ("Centro Norte", -11.0, -7.1),
    ("Centro Sur",   -15.8, -11.0),
]

AGG_LEVELS_M1 = [
    ("Calas\nindividuales", None),
    ("Empresa\nx Diario",   "date"),
    ("Empresa\nx Semanal",  "year_week"),
    ("Empresa\nx Mensual",  "year_month"),
    ("Empresa\nx Temporada","season"),
]

# M2 skips individual calas (VMS can't map to single hauls)
AGG_LEVELS_M2 = [
    ("Empresa\nx Diario",   "date"),
    ("Empresa\nx Semanal",  "year_week"),
    ("Empresa\nx Mensual",  "year_month"),
    ("Empresa\nx Temporada","season"),
]

COLOR_M1 = "#2166ac"
COLOR_M2 = "#1a9641"
BAR_W    = 0.35

PING_HOURS        = 9 / 60   # each VMS ping ~ 9 minutes
T1_DOY            = (91, 212)
T2_DOY            = (305, 365)
MIN_DAYS_COVERAGE = 30
IHMA_DIR          = INPUTS / "ihma_data"


# ── SISESAT helpers ───────────────────────────────────────────────────────────
def normalize_vessel(name):
    if not isinstance(name, str):
        return ""
    name = name.replace("\xa0", " ").strip().upper()
    m = re.fullmatch(r"T(\d+)", name)
    if m:
        name = f"TASA {m.group(1)}"
    return name


def build_sisesat_catalogue():
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


def load_sisesat_effort_daily(catalogue):
    """Return DataFrame: vessel_norm, date, season, effort_hours (daily resolution).

    Filters each file to the correct season DOY window and calendar year.
    Drops seasons with fewer than MIN_DAYS_COVERAGE days of clean VMS data.
    """
    records = []
    for entry in catalogue:
        season = entry["season"]
        fpath  = entry["file"]
        tempo, yr_str = season.split()
        year = int(yr_str)
        doy_start, doy_end = T1_DOY if tempo == "1ra" else T2_DOY

        try:
            df = pd.read_csv(fpath, usecols=["Cod_Barco", "Date"], low_memory=False)
        except Exception:
            continue
        if df.empty:
            continue

        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df[(df["Date"].dt.year == year) &
                (df["Date"].dt.dayofyear >= doy_start) &
                (df["Date"].dt.dayofyear <= doy_end)]

        if df.empty:
            continue
        n_days = (df["Date"].max() - df["Date"].min()).days
        if n_days < MIN_DAYS_COVERAGE:
            continue

        # Count pings per vessel per calendar day
        df["date"] = df["Date"].dt.normalize()
        daily = (
            df.groupby(["Cod_Barco", "date"])
            .size()
            .reset_index(name="n_pings")
        )
        daily["vessel_norm"]  = daily["Cod_Barco"].apply(normalize_vessel)
        daily["effort_hours"] = daily["n_pings"] * PING_HOURS
        daily["season"]       = season
        records.append(daily[["vessel_norm", "date", "season", "effort_hours"]])

    if not records:
        return pd.DataFrame(columns=["vessel_norm", "date", "season", "effort_hours"])
    return pd.concat(records, ignore_index=True)


# ── calas loader ─────────────────────────────────────────────────────────────
def load_calas():
    df = pd.read_csv(OUTPUTS / "calas_enriched.csv", low_memory=False)
    rename_map = {
        "fecha_cala": "date", "fecha": "date",
        "temporada": "season", "declarado_tm": "catch_tm",
        "latitud": "lat", "longitud": "lon",
        "modis_sst_anomaly": "modis_sst_anom",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df["date"] = pd.to_datetime(df["date"])
    df["vessel_norm"] = df["vessel"].apply(normalize_vessel)
    return df


# ── OLS helper ────────────────────────────────────────────────────────────────
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


# ── M1: log(catch) at all aggregation levels ──────────────────────────────────
def collect_m1(sub, reg_name):
    betas, ses, pvals, ns, labels = [], [], [], [], []
    for label, tcol in AGG_LEVELS_M1:
        if tcol is None:
            res = run_ols(sub["modis_sst_anom"], sub["log_catch"])
        else:
            agg = sub.groupby(["company", tcol]).agg(
                total_catch=("catch_tm", "sum"),
                mean_sst_anom=("modis_sst_anom", "mean"),
            ).reset_index()
            res = run_ols(agg["mean_sst_anom"], np.log(agg["total_catch"]))
        if res is None:
            continue
        betas.append(res["beta"])
        ses.append(res["se"])
        pvals.append(res["p"])
        ns.append(res["N"])
        labels.append(label)
        print(f"M1 | {reg_name:6s} | {label.replace(chr(10),' '):25s} | "
              f"beta={res['beta']:+.3f}  se={res['se']:.3f}  "
              f"p={res['p']:.4f}  R2={res['r2']:.3f}  N={res['N']:,}")
    return betas, ses, pvals, ns, labels


# ── M2: log(CPUE) at all supported aggregation levels ────────────────────────
def collect_m2(sub, effort_empresa_daily, reg_name):
    """M2: log(CPUE) ~ SST at empresa x time aggregation levels.

    effort_empresa_daily: DataFrame with company, date, season, effort_hours
    """
    # Add time keys to effort
    eff = effort_empresa_daily.copy()
    iso = eff["date"].dt.isocalendar()
    eff["year_week"]  = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    eff["year_month"] = eff["date"].dt.to_period("M").astype(str)

    betas, ses, pvals, ns, labels = [], [], [], [], []
    for label, tcol in AGG_LEVELS_M2:
        # Aggregate calas to empresa x tcol
        calas_agg = (
            sub.groupby(["company", tcol]).agg(
                total_catch=("catch_tm", "sum"),
                mean_sst_anom=("modis_sst_anom", "mean"),
            ).reset_index()
        )
        # Aggregate effort to empresa x tcol
        eff_agg = (
            eff.groupby(["company", tcol])["effort_hours"]
            .sum().reset_index()
        )

        merged = calas_agg.merge(eff_agg, on=["company", tcol], how="inner")
        merged = merged[merged["effort_hours"] > 0]
        if len(merged) < 10:
            continue

        merged["log_cpue"] = np.log(merged["total_catch"] / merged["effort_hours"])
        res = run_ols(merged["mean_sst_anom"], merged["log_cpue"])
        if res is None:
            continue

        betas.append(res["beta"])
        ses.append(res["se"])
        pvals.append(res["p"])
        ns.append(res["N"])
        labels.append(label)
        print(f"M2 | {reg_name:6s} | {label.replace(chr(10),' '):25s} | "
              f"beta={res['beta']:+.3f}  se={res['se']:.3f}  "
              f"p={res['p']:.4f}  R2={res['r2']:.3f}  N={res['N']:,}")

    return betas, ses, pvals, ns, labels


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    out = PLOTS / "step11_ols_betas.png"
    if out.exists():
        print("step11_ols_betas.png exists -- skipping")
        return

    # --- Load calas ---
    df = load_calas()
    df = df.dropna(subset=["modis_sst_anom", "catch_tm", "lat", "lon", "season", "company"])
    df = df[df["catch_tm"] > 0]
    df["log_catch"] = np.log(df["catch_tm"])

    iso = df["date"].dt.isocalendar()
    df["year_week"]  = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    # --- Load SISESAT effort (daily resolution) ---
    print("Loading SISESAT effort data (daily resolution)...")
    catalogue   = build_sisesat_catalogue()
    effort_raw  = load_sisesat_effort_daily(catalogue)

    # Map vessel -> company from calas
    vc_lookup = (
        df.groupby("vessel_norm")["company"]
        .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else None)
        .reset_index()
        .rename(columns={"company": "company_lookup"})
    )
    effort_with_co = effort_raw.merge(vc_lookup, on="vessel_norm", how="left")
    effort_with_co = effort_with_co.dropna(subset=["company_lookup"])
    effort_empresa_daily = (
        effort_with_co.groupby(["company_lookup", "date", "season"])["effort_hours"]
        .sum().reset_index()
        .rename(columns={"company_lookup": "company"})
    )
    print(f"  Effort: {len(effort_empresa_daily):,} empresa x day records from SISESAT")

    # --- Collect results per region ---
    panel_data = []
    for reg_name, lat_min, lat_max in REGIONS:
        sub = df[df["lat"] > lat_min].copy()
        if lat_max is not None:
            sub = sub[sub["lat"] <= lat_max]

        print(f"\n--- {reg_name} | M1 ---")
        m1 = collect_m1(sub, reg_name)

        print(f"\n--- {reg_name} | M2 ---")
        m2 = collect_m2(sub, effort_empresa_daily, reg_name)

        panel_data.append((reg_name, m1, m2))

    # --- Shared y-axis across all panels and both methodologies ---
    all_b, all_s = [], []
    for _, (b1, s1, *_), (b2, s2, *_) in panel_data:
        all_b.extend(b1 + b2)
        all_s.extend(s1 + s2)
    ymax = (np.abs(all_b) + 1.96 * np.array(all_s)).max() * 1.3
    ylim = (-ymax, ymax)

    # x-positions based on M1 labels (5 levels)
    m1_all_labels = [lbl for lbl, _ in AGG_LEVELS_M1]
    x_pos = np.arange(len(m1_all_labels))

    # --- Plot ---
    fig, axes = plt.subplots(1, len(REGIONS), figsize=(8 * len(REGIONS), 6))

    for ax, (reg_name, m1, m2) in zip(axes, panel_data):
        b1, se1, p1, n1, lab1 = m1
        b2, se2, p2, n2, lab2 = m2

        # Build lookup for M2 results by label
        m2_by_label = {lbl: (b, se, p, n) for lbl, b, se, p, n in zip(lab2, b2, se2, p2, n2)}

        # M1 bars
        for i, (lbl, b, se, p, n) in enumerate(zip(lab1, b1, se1, p1, n1)):
            col = COLOR_M1 if p < 0.05 else "#aaaaaa"
            ax.bar(x_pos[i] - BAR_W / 2, b, width=BAR_W,
                   color=col, alpha=0.85, zorder=3)
            ax.errorbar(x_pos[i] - BAR_W / 2, b, yerr=1.96 * se,
                        fmt="none", color="black", capsize=3, lw=1.0, zorder=4)
            pstr = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            ax.text(x_pos[i] - BAR_W / 2,
                    b + np.sign(b) * (1.96 * se + ymax * 0.03),
                    pstr, ha="center", va="bottom" if b >= 0 else "top", fontsize=7)
            ax.text(x_pos[i] - BAR_W / 2, ylim[0] + ymax * 0.02,
                    f"{n:,}", ha="center", va="bottom", fontsize=5.5, color="#333333")

        # M2 bars (only at levels where M2 was computed)
        for i, lbl in enumerate(m1_all_labels):
            if lbl not in m2_by_label:
                continue
            b, se, p, n = m2_by_label[lbl]
            col = COLOR_M2 if p < 0.05 else "#bbbbbb"
            ax.bar(x_pos[i] + BAR_W / 2, b, width=BAR_W,
                   color=col, alpha=0.85, zorder=3)
            ax.errorbar(x_pos[i] + BAR_W / 2, b, yerr=1.96 * se,
                        fmt="none", color="black", capsize=3, lw=1.0, zorder=4)
            pstr = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            ax.text(x_pos[i] + BAR_W / 2,
                    b + np.sign(b) * (1.96 * se + ymax * 0.03),
                    pstr, ha="center", va="bottom" if b >= 0 else "top", fontsize=7)
            ax.text(x_pos[i] + BAR_W / 2, ylim[0] + ymax * 0.02,
                    f"{n:,}", ha="center", va="bottom", fontsize=5.5, color="#555555")

        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.set_ylim(ylim)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(m1_all_labels, fontsize=8)
        ax.set_ylabel("Beta (OLS coefficient)", fontsize=9)
        ax.set_title(f"Region {reg_name}", loc="left", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(handles=[
            Patch(color=COLOR_M1, alpha=0.85,
                  label="M1: log(captura) ~ SST  (2015-2024)"),
            Patch(color=COLOR_M2, alpha=0.85,
                  label="M2: log(CPUE) ~ SST  (SISESAT 2017-2022)"),
            Patch(color="#aaaaaa", alpha=0.85, label="p >= 0.05"),
        ], frameon=False, fontsize=7, loc="upper right")

    fig.suptitle(
        "OLS betas  |  M1 = log(captura)  vs  M2 = log(CPUE = captura / esfuerzo VMS)  |  "
        "por nivel de agregacion  |  Norte  |  Centro Norte (-11 a -7.1 S)  |  Centro Sur (-15.8 a -11 S)",
        fontsize=10, x=0.01, ha="left")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
