"""
Step 09 - Spatial Tobit Model

Panel regression (0.25-degree grid cell x ISO week x season) modeling
log-catch as left-censored. Cells with no calas in a given week are
treated as left-censored at log(1 tm). Runs pooled, by season type,
by domain threshold, and by IMARPE region.

Inputs:
  OUTPUTS/calas_all_data.csv                   (from step 02)
  FEATURES/SST_weekly_2015-2024_4N_74W_16S_83W_0.25deg.nc  (reference grid)
  FEATURES/sst_anomaly_daily_{year}.nc         (from step 04b)

Outputs:
  PLOTS/step8b_tobit_decay.png
  PLOTS/step8c_tobit_sensitivity_by_season.png
  PLOTS/step8c_tobit_sensitivity_pooled.png
  PLOTS/step13_spatial_tobit.png
  PLOTS/step13_censure_map.png
  PLOTS/step13_sensitivity_domain.png
  PLOTS/step13_by_region.png

Prints Tobit coefficient estimates to stdout.

Skip logic: skipped if step13_spatial_tobit.png already exists.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from scipy.optimize import minimize, approx_fprime
from scipy.stats import norm as sp_norm
from tqdm import tqdm

from config import FEATURES, OUTPUTS, PLOTS

PROJ  = ccrs.PlateCarree()
TAB   = plt.cm.tab10.colors
L_CENS = 0.0   # log(1 tm) - left-censoring point


def get_season_type(s):
    s = str(s)
    if s.startswith("1ra") or s.endswith("CN-I"):  return "1ra"
    if s.startswith("2da") or s.endswith("CN-II"): return "2da"
    return np.nan


def snap_to_grid(vals, centers):
    idx = np.argmin(np.abs(centers[:, None] - vals[None, :]), axis=0)
    return centers[idx]


# ---------------------------------------------------------------------------
# Tobit MLE
# ---------------------------------------------------------------------------

def tobit_nll(params, X, y, censored, L=L_CENS):
    beta  = params[:-1]
    sigma = np.exp(params[-1])
    xb    = X @ beta
    ll    = np.zeros(len(y))
    ll[~censored] = sp_norm.logpdf(y[~censored], loc=xb[~censored], scale=sigma)
    ll[ censored] = sp_norm.logcdf(L,            loc=xb[censored],  scale=sigma)
    return -ll.sum()


def fit_tobit(x_vals, y_vals, cens_mask, L=L_CENS):
    X = np.column_stack([np.ones(len(x_vals)), x_vals])
    y = y_vals.copy()
    y[cens_mask] = L
    obs = ~cens_mask
    if obs.sum() < 5:
        return None
    b0    = np.linalg.lstsq(X[obs], y[obs], rcond=None)[0]
    resid = y[obs] - X[obs] @ b0
    p0    = np.append(b0, np.log(resid.std() + 1e-6))
    res   = minimize(tobit_nll, p0, args=(X, y, cens_mask, L),
                     method="L-BFGS-B", options={"maxiter": 5000, "ftol": 1e-10})
    if not res.success:
        return None
    eps  = 1e-5; n = len(res.x)
    grad0 = approx_fprime(res.x, lambda p: tobit_nll(p, X, y, cens_mask, L), eps)
    hess  = np.zeros((n, n))
    for i in range(n):
        xf = res.x.copy(); xf[i] += eps
        hess[i] = (approx_fprime(xf, lambda p: tobit_nll(p, X, y, cens_mask, L), eps) - grad0) / eps
    hess = (hess + hess.T) / 2
    vcov = np.linalg.pinv(hess)
    se   = np.sqrt(np.maximum(np.diag(vcov)[:-1], 0))
    return {"beta": res.x[1], "se": se[1], "sigma": np.exp(res.x[-1]),
            "intercept": res.x[0]}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    if (PLOTS / "step13_spatial_tobit.png").exists():
        print("step09 outputs exist -- skipping")
        return

    # --- reference grid ---
    ref_path = FEATURES / "SST_weekly_2015-2024_4N_74W_16S_83W_0.25deg.nc"
    if not ref_path.exists():
        print(f"Reference grid not found: {ref_path}  -- cannot run step09")
        return
    REF_DS = xr.open_dataset(ref_path)
    LATS   = REF_DS.latitude.values
    LONS   = REF_DS.longitude.values
    REF_DS.close()
    print(f"Grid: {len(LATS)} lat x {len(LONS)} lon = {len(LATS)*len(LONS)} cells")

    # --- 1. historical domain ---
    df_raw = pd.read_csv(OUTPUTS / "calas_all_data.csv", low_memory=False)
    df_raw["fecha_cala"] = pd.to_datetime(df_raw["fecha_cala"], errors="coerce")
    df_raw["season_type"] = df_raw["temporada"].apply(get_season_type)
    df_raw = df_raw.dropna(subset=["latitud", "longitud", "fecha_cala", "season_type"])

    df_raw["lat_g"] = snap_to_grid(df_raw["latitud"].values,  LATS)
    df_raw["lon_g"] = snap_to_grid(df_raw["longitud"].values, LONS)

    cell_counts = df_raw.groupby(["lat_g", "lon_g"]).size().rename("n_calas_hist")
    MIN_CALAS   = 1
    domain_cells = set(cell_counts[cell_counts >= MIN_CALAS].index.tolist())
    print(f"Domain (threshold={MIN_CALAS}): {len(domain_cells)} cells")

    # --- 2. full spatial panel ---
    season_windows = (
        df_raw.groupby("temporada")["fecha_cala"]
        .agg(["min", "max"]).reset_index()
    )
    season_windows.columns = ["temporada", "date_min", "date_max"]

    df_raw["iso_week"] = df_raw["fecha_cala"].dt.to_period("W").dt.start_time
    catch_by_cell_week = (
        df_raw.groupby(["temporada", "iso_week", "lat_g", "lon_g"])["declarado_tm"]
        .sum().reset_index().rename(columns={"declarado_tm": "catch_tm"})
    )

    rows = []
    domain_list = list(domain_cells)
    for _, sw in tqdm(season_windows.iterrows(), total=len(season_windows), desc="Building panel"):
        weeks = pd.date_range(sw["date_min"], sw["date_max"], freq="W-MON")
        for week in weeks:
            for (lat_g, lon_g) in domain_list:
                rows.append({"temporada": sw["temporada"], "iso_week": week,
                             "lat_g": lat_g, "lon_g": lon_g})

    panel = pd.DataFrame(rows)
    panel = panel.merge(catch_by_cell_week,
                        on=["temporada", "iso_week", "lat_g", "lon_g"], how="left")
    panel["censored"]  = panel["catch_tm"].isna() | (panel["catch_tm"] <= 0)
    panel["log_catch"] = np.where(panel["censored"], np.nan,
                                  np.log(panel["catch_tm"].clip(lower=1e-6)))

    n_obs  = (~panel["censored"]).sum()
    n_cens = panel["censored"].sum()
    print(f"Panel: {len(panel):,} rows | observed: {n_obs:,} | censored: {n_cens:,} "
          f"({n_cens/len(panel)*100:.1f}%)")

    # --- 3. weekly SST anomaly ---
    sst_weekly_list = []
    for year in tqdm(range(2015, 2026), desc="SST anomaly years"):
        fpath = FEATURES / f"sst_anomaly_daily_{year}.nc"
        if not fpath.exists():
            continue
        ds = xr.open_dataset(fpath)
        ds_interp = ds.interp(lat=LATS, lon=LONS, method="linear")
        ds_w = ds_interp.resample(time="W-MON").mean()
        sst_weekly_list.append(ds_w)
        ds.close()

    if not sst_weekly_list:
        print("No SST anomaly files found -- cannot run spatial Tobit")
        return

    sst_weekly = xr.concat(sst_weekly_list, dim="time")
    print("SST weekly ready:", sst_weekly.sizes)

    # --- 4. merge panel with SST ---
    df_sst = (
        sst_weekly["sst_anomaly"]
        .to_dataframe().reset_index()
        .rename(columns={"time": "iso_week", "lat": "lat_g", "lon": "lon_g",
                         "sst_anomaly": "sst_anom"})
        .dropna(subset=["sst_anom"])
    )
    df_sst["iso_week"] = pd.to_datetime(df_sst["iso_week"])
    panel["iso_week"]  = pd.to_datetime(panel["iso_week"])
    panel = panel.merge(df_sst, on=["iso_week", "lat_g", "lon_g"], how="left")
    panel = panel.dropna(subset=["sst_anom"])

    # --- 5. fit Tobit ---
    # pooled
    x_all = panel["sst_anom"].values
    y_all = panel["log_catch"].fillna(L_CENS).values
    c_all = panel["censored"].values

    print("Fitting pooled Tobit...")
    r_pool = fit_tobit(x_all, y_all, c_all)
    if r_pool:
        print(f"Pooled: beta_SST={r_pool['beta']:+.4f}  SE={r_pool['se']:.4f}  "
              f"t={r_pool['beta']/r_pool['se']:.2f}")
        print(f"Semi-elasticity: 1 deg anomaly -> {(np.exp(r_pool['beta'])-1)*100:+.1f}%")

    # by season type
    panel["season_type"] = panel["temporada"].apply(get_season_type)
    results = {}
    for stype in ["1ra", "2da"]:
        sub = panel[panel["season_type"] == stype]
        r   = fit_tobit(sub["sst_anom"].values,
                        sub["log_catch"].fillna(L_CENS).values,
                        sub["censored"].values)
        results[stype] = r
        if r:
            label = "Primera" if stype == "1ra" else "Segunda"
            print(f"{label}: beta={r['beta']:+.4f}  SE={r['se']:.4f}  "
                  f"t={r['beta']/r['se']:.2f}  -> {(np.exp(r['beta'])-1)*100:+.1f}% per deg")

    # decay curve plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    labels_map = {"1ra": "Primera temporada", "2da": "Segunda temporada"}
    for ax, stype in zip(axes, ["1ra", "2da"]):
        r   = results[stype]
        if r is None:
            continue
        sub = panel[panel["season_type"] == stype]
        x_range = np.linspace(sub["sst_anom"].quantile(0.02),
                              sub["sst_anom"].quantile(0.98), 300)
        y_tb = (np.exp(r["beta"] * x_range) - 1) * 100
        ax.plot(x_range, y_tb, color=TAB[0], lw=2)
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.axvline(0, color="grey", lw=0.8, ls=":", alpha=0.7)
        obs_x  = sub.loc[~sub["censored"], "sst_anom"]
        cens_x = sub.loc[sub["censored"],  "sst_anom"]
        ymin, ymax = ax.get_ylim()
        rug_y = ymin + 0.01 * (ymax - ymin)
        ax.plot(obs_x,  np.full(len(obs_x),  rug_y), "|", color=TAB[0], alpha=0.2, ms=5)
        ax.plot(cens_x, np.full(len(cens_x), rug_y), "|", color=TAB[3], alpha=0.15, ms=5)
        n_obs_s  = (~sub["censored"]).sum()
        n_cens_s = sub["censored"].sum()
        ax.set_title(f"{labels_map[stype]}\nbeta={r['beta']:+.3f}  SE={r['se']:.3f}  "
                     f"(obs={n_obs_s:,} | cens={n_cens_s:,})", fontsize=9, loc="left")
        ax.set_xlabel("SST anomaly (deg C)"); ax.set_ylabel("% change in catch vs reference")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle("Spatial Tobit: catch decay vs SST anomaly | cell-week panel 0.25 deg",
                 x=0.01, ha="left", fontsize=10, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(PLOTS / "step13_spatial_tobit.png", dpi=120, bbox_inches="tight")
    plt.close()

    # censorship frequency map
    cens_freq = (panel.groupby(["lat_g", "lon_g"])["censored"]
                 .mean().rename("cens_rate").reset_index())
    cc_df = cell_counts.reset_index()
    lon_grid, lat_grid = np.meshgrid(LONS, LATS)
    cens_map = np.full(lon_grid.shape, np.nan)
    hist_map = np.full(lon_grid.shape, np.nan)
    for _, row in cens_freq.iterrows():
        i = np.argmin(np.abs(LATS - row["lat_g"]))
        j = np.argmin(np.abs(LONS - row["lon_g"]))
        cens_map[i, j] = row["cens_rate"]
    for _, row in cc_df.iterrows():
        i = np.argmin(np.abs(LATS - row["lat_g"]))
        j = np.argmin(np.abs(LONS - row["lon_g"]))
        hist_map[i, j] = row["n_calas_hist"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={"projection": PROJ})
    for ax, data, title, cmap in [
        (axes[0], cens_map, "Censorship rate (fraction of weeks)", "RdYlGn_r"),
        (axes[1], np.log1p(hist_map), "log(1 + historical calas)", "YlOrRd"),
    ]:
        ax.set_extent([LONS.min()-0.5, LONS.max()+0.5, LATS.min()-0.5, LATS.max()+0.5])
        ax.add_feature(cfeature.LAND, facecolor="wheat", zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":", zorder=3)
        mesh = ax.pcolormesh(LONS, LATS, data, transform=PROJ, cmap=cmap, zorder=1)
        plt.colorbar(mesh, ax=ax, shrink=0.7)
        ax.set_title(title, loc="left")
    plt.tight_layout()
    plt.savefig(PLOTS / "step13_censure_map.png", dpi=120, bbox_inches="tight")
    plt.close()

    # domain sensitivity
    print("\nDomain sensitivity analysis...")
    sensitivity_rows = []
    for min_c in [1, 5, 10, 20, 50]:
        dom = set(cell_counts[cell_counts >= min_c].index.tolist())
        sub_panel = panel[panel.set_index(["lat_g", "lon_g"]).index.isin(dom)
                          if False else panel["lat_g"].map(lambda x: True)].copy()
        # rebuild panel for this threshold
        rows2 = []
        for _, sw in season_windows.iterrows():
            weeks = pd.date_range(sw["date_min"], sw["date_max"], freq="W-MON")
            for week in weeks:
                for (lg, lo) in dom:
                    rows2.append({"temporada": sw["temporada"], "iso_week": week,
                                  "lat_g": lg, "lon_g": lo})
        p2 = pd.DataFrame(rows2)
        p2 = p2.merge(catch_by_cell_week, on=["temporada", "iso_week", "lat_g", "lon_g"],
                      how="left")
        p2["censored"]  = p2["catch_tm"].isna() | (p2["catch_tm"] <= 0)
        p2["log_catch"] = np.where(p2["censored"], np.nan,
                                   np.log(p2["catch_tm"].clip(lower=1e-6)))
        p2 = p2.merge(df_sst, on=["iso_week", "lat_g", "lon_g"], how="left")
        p2 = p2.dropna(subset=["sst_anom"])
        r = fit_tobit(p2["sst_anom"].values, p2["log_catch"].fillna(L_CENS).values,
                      p2["censored"].values)
        if r:
            cens_rate = p2["censored"].mean() * 100
            sensitivity_rows.append({"min_calas": min_c, "n_cells": len(dom),
                                     "cens_rate": round(cens_rate, 1),
                                     "beta": round(r["beta"], 4), "se": round(r["se"], 4)})
            print(f"  min_calas={min_c:3d}: {len(dom)} cells | cens={cens_rate:.1f}% | "
                  f"beta={r['beta']:+.4f} SE={r['se']:.4f}")

    if sensitivity_rows:
        df_sens = pd.DataFrame(sensitivity_rows)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].errorbar(df_sens["min_calas"], df_sens["beta"],
                         yerr=1.96 * df_sens["se"], fmt="o-", color=TAB[0], capsize=4)
        axes[0].axhline(0, color="grey", lw=0.8, ls="--")
        axes[0].set_xlabel("Minimum historical calas per cell")
        axes[0].set_ylabel("beta (Tobit)")
        axes[0].set_title("Sensitivity of beta to domain threshold", loc="left")
        axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)
        axes[1].plot(df_sens["min_calas"], df_sens["cens_rate"], "s-", color=TAB[1])
        axes[1].set_xlabel("Minimum historical calas per cell")
        axes[1].set_ylabel("Censorship rate (%)")
        axes[1].set_title("Censorship rate vs domain threshold", loc="left")
        axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(PLOTS / "step13_sensitivity_domain.png", dpi=120, bbox_inches="tight")
        plt.close()

    # by IMARPE region
    def assign_region(lat):
        if lat > -8:   return "Norte"
        elif lat > -12: return "Centro"
        else:           return "Sur"

    panel["region"] = panel["lat_g"].apply(assign_region)

    THRESHOLDS = [1500, 2000]
    all_results = {}
    for MIN_C in THRESHOLDS:
        domain_c = set(cell_counts[cell_counts >= MIN_C].index.tolist())
        if len(domain_c) == 0:
            print(f"Threshold {MIN_C}: no cells in domain, skipping")
            continue
        print(f"\n{'='*60}\nThreshold {MIN_C} | domain: {len(domain_c)} cells")

        rows3 = []
        for _, sw in season_windows.iterrows():
            weeks = pd.date_range(sw["date_min"], sw["date_max"], freq="W-MON")
            for week in weeks:
                for (lg, lo) in domain_c:
                    rows3.append({"temporada": sw["temporada"], "iso_week": week,
                                  "lat_g": lg, "lon_g": lo})
        p3 = pd.DataFrame(rows3)
        p3 = p3.merge(catch_by_cell_week, on=["temporada", "iso_week", "lat_g", "lon_g"],
                      how="left")
        p3["censored"]   = p3["catch_tm"].isna() | (p3["catch_tm"] <= 0)
        p3["log_catch"]  = np.where(p3["censored"], np.nan,
                                    np.log(p3["catch_tm"].clip(lower=1e-6)))
        p3["season_type"] = p3["temporada"].apply(get_season_type)
        p3["region"]      = p3["lat_g"].apply(assign_region)
        p3 = p3.merge(df_sst, on=["iso_week", "lat_g", "lon_g"], how="left")
        p3 = p3.dropna(subset=["sst_anom"])

        result_rows = []
        for stype in ["1ra", "2da"]:
            for region in ["Norte", "Centro", "Sur"]:
                sub = p3[(p3["season_type"] == stype) & (p3["region"] == region)]
                r = fit_tobit(sub["sst_anom"].values,
                              sub["log_catch"].fillna(L_CENS).values,
                              sub["censored"].values)
                if r:
                    seas = "Primera" if stype == "1ra" else "Segunda"
                    print(f"  {region} | {seas}: beta={r['beta']:+.4f}  SE={r['se']:.4f}  "
                          f"-> {(np.exp(r['beta'])-1)*100:+.1f}%")
                    result_rows.append({"grupo": "Region x Temporada", "region": region,
                                        "seas": seas, "beta": r["beta"], "se": r["se"]})

        rp = fit_tobit(p3["sst_anom"].values, p3["log_catch"].fillna(L_CENS).values,
                       p3["censored"].values)
        if rp:
            result_rows.append({"grupo": "Pooled", "region": "All", "seas": "All",
                                 "beta": rp["beta"], "se": rp["se"]})
        all_results[MIN_C] = {"results": pd.DataFrame(result_rows)}

    if all_results:
        REGION_COLORS = {"Norte": "#2196F3", "Centro": "#FF9800", "Sur": "#4CAF50"}
        fig, axes = plt.subplots(1, len(THRESHOLDS), figsize=(14, 5), sharey=True)
        for ax, thr in zip(axes, THRESHOLDS):
            if thr not in all_results:
                continue
            df_r   = all_results[thr]["results"]
            df_reg = df_r[df_r["grupo"] == "Region x Temporada"].copy()
            df_pool_rows = df_r[df_r["grupo"] == "Pooled"]
            if len(df_pool_rows):
                df_pool = df_pool_rows.iloc[0]
                ax.axhline(df_pool["beta"], color="k", lw=1.2, ls="--",
                           label=f"Pooled b={df_pool['beta']:+.3f}")
            ax.axhline(0, color="grey", lw=0.6, ls=":")
            x_pos  = {"Norte": 0, "Centro": 1, "Sur": 2}
            offset = {"Primera": -0.15, "Segunda": 0.15}
            marker = {"Primera": "o", "Segunda": "s"}
            for _, row in df_reg.iterrows():
                x = x_pos[row["region"]] + offset[row["seas"]]
                ax.errorbar(x, row["beta"], yerr=1.96 * row["se"],
                            fmt=marker[row["seas"]], color=REGION_COLORS[row["region"]],
                            capsize=4, ms=7)
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(["Norte", "Centro", "Sur"])
            ax.set_ylabel("beta (Tobit)")
            ax.set_title(f"Domain threshold: {thr} historical calas", loc="left")
            ax.legend(frameon=False, fontsize=8)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(PLOTS / "step13_by_region.png", dpi=120, bbox_inches="tight")
        plt.close()

    print(f"\nAll step09 outputs saved to {PLOTS}")


if __name__ == "__main__":
    main()
