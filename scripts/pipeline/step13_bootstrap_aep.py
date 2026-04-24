"""
Step 13 - Block bootstrap AEP of catch losses (Centro region)

Workflow:
  1. Load MODIS SST anomaly daily series 2002-2025 for Centro fishing pixels.
  2. Compute baseline seasonal catch (T1/T2) from observed calas data.
  3. Block bootstrap (block=7d, window=+-10d, N=4000 synthetic years):
       - Build each synthetic year as a full 365-day daily series by
         stitching consecutive 7-day blocks, each sampled from historical
         data within +-10 DOY of the current position (cross-year).
       - Compute T1 mean (Apr-Jul, DOY 91-212) and T2 mean (Nov-Dec,
         DOY 305-365) from the synthetic daily series.
       - Convert to loss in tons: baseline_catch * (1 - exp(beta * SST))
         when SST > 0.
       - Annual loss = Loss_T1 + Loss_T2.
  4. AEP curve: annual exceedance probability vs loss in tons (90% CI).
  5. Part 1 figure: monthly SST time series + seasonal distribution with
     p90/p95/p99 triggers.

Beta used: -0.849 (OLS M1 empresa x temporada, Centro, from step 11).
Bootstrap procedure follows the block bootstrap spec (PDF 2026-04-20).

Inputs:
  sources/modis/sst/sst_merged_daily_complete.nc  (raw MODIS SST 2002-2025)
  OUTPUTS/calas_enriched.csv                       (baseline catch)

Outputs:
  PLOTS/step13_sst_timeseries.png
  PLOTS/step13_bootstrap_aep.png

Skip logic: skipped if both outputs exist.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import xarray as xr
from scipy.stats import gaussian_kde

from config import FEATURES, OUTPUTS, PLOTS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LAT_MIN, LAT_MAX = -11.0, -7.1   # Centro Norte (consistent with trigger region)
LON_BROAD_W, LON_BROAD_E = -82.0, -74.0  # broad bbox for initial SST slice

# Season DOY ranges (1-indexed)
T1_DOY_START, T1_DOY_END = 91, 212   # Apr 1 - Jul 31
T2_DOY_START, T2_DOY_END = 305, 365  # Nov 1 - Dec 31

T1_MONTHS = [4, 5, 6, 7]
T2_MONTHS = [11, 12]

BETA        = -0.816   # OLS M1 empresa x temporada, Centro Norte (step15, highest R²=0.261)
N_BOOTSTRAP = 4000     # synthetic years

TRIGGERS     = [0.90, 0.95, 0.99]
TRIGGER_LBLS = ["p90", "p95", "p99"]
TRIGGER_COLS = ["#f4a261", "#e76f51", "#9b2226"]

RAW_SST_MERGED = ("/home/jupyter-daniela/suyana/sources/modis/sst/"
                  "sst_merged_daily_complete.nc")
CLIM_YEAR_START = 2005   # fixed climatology baseline: 2005-2024 (20 full years)
CLIM_YEAR_END   = 2024


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def build_fishing_polygon():
    """
    Fishing corridor for Centro Norte: 5th-95th percentile of observed
    cala longitudes per 1-degree lat band, bounded by LAT_MIN/LAT_MAX.
    Returns a matplotlib.path.Path for 2D containment testing.
    """
    from matplotlib.path import Path

    df = pd.read_csv(
        str(OUTPUTS / "calas_all_data.csv"),
        usecols=["latitud", "longitud"], low_memory=False,
    ).rename(columns={"latitud": "lat", "longitud": "lon"}).dropna()
    df = df[(df["lat"] >= LAT_MIN) & (df["lat"] <= LAT_MAX)]

    band_lo_edges = np.arange(int(np.floor(LAT_MIN)), int(np.ceil(LAT_MAX)), 1.0)

    west_lons, east_lons, valid_lats = [], [], []
    for lo in band_lo_edges:
        band = df[(df["lat"] >= lo) & (df["lat"] < lo + 1.0)]
        if len(band) < 20:
            continue
        west_lons.append(np.percentile(band["lon"], 5))
        east_lons.append(np.percentile(band["lon"], 95))
        valid_lats.append(lo + 0.5)

    valid_lats = np.array(valid_lats)
    west_lons  = np.array(west_lons)
    east_lons  = np.array(east_lons)

    lat_full  = np.concatenate([[LAT_MIN], valid_lats, [LAT_MAX]])
    west_full = np.concatenate([[west_lons[0]], west_lons, [west_lons[-1]]])
    east_full = np.concatenate([[east_lons[0]], east_lons, [east_lons[-1]]])

    poly_lons = np.concatenate([west_full, east_full[::-1], [west_full[0]]])
    poly_lats = np.concatenate([lat_full,  lat_full[::-1],  [lat_full[0]]])

    print(f"  Fishing polygon: {len(valid_lats)} lat bands, "
          f"lon {west_lons.min():.2f} to {east_lons.max():.2f}")

    verts = list(zip(poly_lons, poly_lats))
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
    return Path(verts, codes)


def _polygon_spatial_mean(ds, var, fishing_path):
    """Apply fishing polygon mask and return spatial-mean time series."""
    lat_vals = ds["lat"].values
    lon_vals = ds["lon"].values
    lat_m = (lat_vals >= LAT_MIN) & (lat_vals <= LAT_MAX)
    lon_m = (lon_vals >= LON_BROAD_W) & (lon_vals <= LON_BROAD_E)
    lat_sub = lat_vals[lat_m]
    lon_sub = lon_vals[lon_m]

    LON_2D, LAT_2D = np.meshgrid(lon_sub, lat_sub)
    points  = np.column_stack([LON_2D.ravel(), LAT_2D.ravel()])
    mask_2d = fishing_path.contains_points(points).reshape(lat_sub.size, lon_sub.size)

    mask_da = xr.DataArray(mask_2d, dims=["lat", "lon"],
                           coords={"lat": lat_sub, "lon": lon_sub})
    series = (ds[var].isel(lat=lat_m, lon=lon_m)
              .where(mask_da).mean(dim=["lat", "lon"]).compute().to_series())
    series.index = pd.to_datetime(series.index)
    return series.dropna()


def load_daily_sst():
    """
    Area-average raw SST over the Centro Norte fishing corridor, then compute
    anomalies relative to a fixed DOY climatology 2005-2024 (20 full years).

    Sources (raw SST, not pre-computed anomalies):
      - sst_merged_daily_complete.nc  (2002 - Feb 2025)
      - sst_daily_{year}.nc in FEATURES for any year beyond the merged file

    Spatial domain: actual fishing polygon (5th-95th pct of cala longitudes
    per 1-deg lat band).
    Returns pd.Series indexed by date.
    """
    fishing_path = build_fishing_polygon()

    # ── Source 1: merged file (2002 to Feb 2025) ────────────────────────────
    ds_merged = xr.open_dataset(RAW_SST_MERGED)
    raw = _polygon_spatial_mean(ds_merged, "sst", fishing_path)
    last_merged = raw.index.max()
    ds_merged.close()
    print(f"  Merged file: {len(raw)} days up to {last_merged.date()}")

    # ── Source 2: per-year daily files for any year not in merged ───────────
    years_needed = range(last_merged.year, 2027)
    for yr in years_needed:
        f = FEATURES / f"sst_daily_{yr}.nc"
        if not f.exists():
            continue
        ds_yr = xr.open_dataset(f)
        s = _polygon_spatial_mean(ds_yr, "sst", fishing_path)
        ds_yr.close()
        # keep only dates strictly after the merged file's last date
        s = s[s.index > last_merged]
        if len(s):
            raw = pd.concat([raw, s]).sort_index()
            print(f"  Added {yr}: {len(s)} days up to {s.index.max().date()}")

    n_pixels = int(fishing_path.contains_points(
        np.column_stack([np.array([-79.0]), np.array([-9.0])])
    ).sum())  # just a placeholder print; real count logged above
    print(f"  Total: {len(raw)} days  {raw.index.min().date()} - {raw.index.max().date()}")

    # ── Climatology: DOY mean over 2005-2024 only ───────────────────────────
    clim_mask = (raw.index.year >= CLIM_YEAR_START) & (raw.index.year <= CLIM_YEAR_END)
    clim = raw[clim_mask].groupby(raw[clim_mask].index.dayofyear).mean()
    print(f"  Climatology: {CLIM_YEAR_START}-{CLIM_YEAR_END} "
          f"({clim_mask.sum()} days, {int(clim_mask.sum()/365.25):.0f} yrs)")

    anom = raw - raw.index.map(lambda d: clim.get(d.dayofyear, np.nan))
    return anom.dropna().sort_index()


def load_baseline_catch():
    """
    Mean total seasonal catch (all companies, Centro) by season type.
    Returns: (mean_t1_tons, mean_t2_tons, seasonal_df)
    """
    df = pd.read_csv(OUTPUTS / "calas_enriched.csv")
    df = df.rename(columns={"temporada": "season", "declarado_tm": "catch_tm"})
    df = df[(df["lat"] > LAT_MIN) & (df["lat"] <= LAT_MAX) & (df["catch_tm"] > 0)]
    seas = df.groupby("season")["catch_tm"].sum().reset_index()
    seas["tipo"] = seas["season"].apply(
        lambda s: "T1" if "1ra" in s else ("T2" if "2da" in s else "other")
    )
    t1 = seas[seas["tipo"] == "T1"]["catch_tm"]
    t2 = seas[seas["tipo"] == "T2"]["catch_tm"]
    return float(t1.mean()), float(t2.mean()), seas


# ---------------------------------------------------------------------------
# 2. Parametric bootstrap - Normal fit to historical seasonal means
# ---------------------------------------------------------------------------

def get_seasonal_sst_means(daily_series):
    """
    Compute observed T1 and T2 seasonal mean SST anomalies per year.
    T1: DOY 91-212 (Apr-Jul), T2: DOY 305-365 (Nov-Dec).
    Returns (t1_means, t2_means) as arrays.
    """
    df = daily_series.rename("sst").reset_index()
    df.columns = ["date", "sst"]
    df["year"] = df["date"].dt.year
    df["doy"]  = df["date"].dt.dayofyear

    t1 = (df[df["doy"].between(T1_DOY_START, T1_DOY_END)]
          .groupby("year")["sst"].mean())
    t2 = (df[df["doy"].between(T2_DOY_START, T2_DOY_END)]
          .groupby("year")["sst"].mean())

    common = t1.index.intersection(t2.index)
    return t1.loc[common].values, t2.loc[common].values


def parametric_bootstrap(daily_series, n_sims=N_BOOTSTRAP):
    """
    Fit independent Normal distributions to observed T1 and T2 seasonal
    SST anomaly means, then draw n_sims synthetic pairs.

    This allows sampling beyond the historical range, generating more
    extreme warm events than the ~23 observed years contain.

    Returns (t1_synth, t2_synth) arrays of shape (n_sims,).
    """
    from scipy.stats import norm

    t1_obs, t2_obs = get_seasonal_sst_means(daily_series)

    mu1, sd1 = t1_obs.mean(), t1_obs.std(ddof=1)
    mu2, sd2 = t2_obs.mean(), t2_obs.std(ddof=1)

    print(f"  T1 fit: mean={mu1:.3f}C  sd={sd1:.3f}C  N={len(t1_obs)}")
    print(f"  T2 fit: mean={mu2:.3f}C  sd={sd2:.3f}C  N={len(t2_obs)}")

    rng = np.random.default_rng(42)
    t1_synth = rng.normal(mu1, sd1, n_sims)
    t2_synth = rng.normal(mu2, sd2, n_sims)
    return t1_synth, t2_synth, mu1, mu2


# ---------------------------------------------------------------------------
# 3. Loss in tons
# ---------------------------------------------------------------------------

def sst_to_loss_tons(sst_anom, baseline_tons, beta=BETA):
    """
    Catch loss in tons for a seasonal SST anomaly.
    Only warm anomalies (SST > 0) produce losses.
    loss = baseline * (1 - exp(beta * SST))  if SST > 0 else 0
    """
    loss_frac = np.where(sst_anom > 0, 1 - np.exp(beta * sst_anom), 0.0)
    return baseline_tons * loss_frac


# ---------------------------------------------------------------------------
# 4. AEP helpers
# ---------------------------------------------------------------------------

def empirical_aep(values):
    """Weibull plotting position. Returns (sorted_vals, exceedance_prob)."""
    sv = np.sort(values)
    n  = len(sv)
    ep = (n + 1 - np.arange(1, n + 1)) / (n + 1)
    return sv, ep


def aep_ci(all_values, n_chunks=20):
    """
    90% CI on AEP by splitting synthetic pool into chunks.
    Returns (ep_grid, ci_lo, ci_median, ci_hi).
    ep_grid is ascending (rare -> common).
    """
    sv_full, ep_full = empirical_aep(all_values)
    ep_grid = np.linspace(ep_full.min(), ep_full.max(), 300)
    chunk   = len(all_values) // n_chunks
    curves  = []
    for i in range(n_chunks):
        sub       = all_values[i * chunk:(i + 1) * chunk]
        sv_i, ep_i = empirical_aep(sub)
        val_i     = np.interp(ep_grid, ep_i[::-1], sv_i[::-1])
        curves.append(val_i)
    curves = np.array(curves)
    median = np.interp(ep_grid, ep_full[::-1], sv_full[::-1])
    ci_lo  = np.percentile(curves,  5, axis=0)
    ci_hi  = np.percentile(curves, 95, axis=0)
    return ep_grid, ci_lo, median, ci_hi


# ---------------------------------------------------------------------------
# 5. Figures
# ---------------------------------------------------------------------------

def season_label_from_monthly(monthly_sst):
    df = monthly_sst.rename("sst").reset_index()
    df.columns = ["date", "sst"]
    df["month"] = df["date"].dt.month
    df["year"]  = df["date"].dt.year
    df["tipo"]  = df["month"].map(
        lambda m: "T1" if m in T1_MONTHS else ("T2" if m in T2_MONTHS else None))
    df = df.dropna(subset=["tipo"])
    seas = df.groupby(["year", "tipo"])["sst"].mean().reset_index()
    seas["season_id"] = seas["year"].astype(str) + "-" + seas["tipo"]
    return seas.sort_values("season_id").reset_index(drop=True)


def plot_timeseries(daily, seasonal_df, triggers_vals):
    monthly = daily.resample("ME").mean()
    fig, axes = plt.subplots(1, 2, figsize=(18, 6),
                             gridspec_kw={"width_ratios": [3, 1.5]})

    ax = axes[0]
    ax.fill_between(monthly.index, monthly.values, 0,
                    where=monthly.values >= 0, alpha=0.3, color="#e76f51",
                    interpolate=True)
    ax.fill_between(monthly.index, monthly.values, 0,
                    where=monthly.values < 0, alpha=0.3, color="#2166ac",
                    interpolate=True)
    ax.plot(monthly.index, monthly.values, color="#333333", lw=0.8, alpha=0.7)

    t1 = seasonal_df[seasonal_df["tipo"] == "T1"]
    t2 = seasonal_df[seasonal_df["tipo"] == "T2"]
    ax.scatter(pd.to_datetime(t1["year"].astype(str) + "-06-01"),
               t1["sst"], s=55, color="#e76f51", zorder=5,
               marker="o", label="T1 (abr-jul)", edgecolors="white", lw=0.5)
    ax.scatter(pd.to_datetime(t2["year"].astype(str) + "-11-15"),
               t2["sst"], s=55, color="#2166ac", zorder=5,
               marker="s", label="T2 (nov-dic)", edgecolors="white", lw=0.5)

    top3 = seasonal_df.nlargest(3, "sst")
    for _, row in top3.iterrows():
        date = (pd.to_datetime(f"{row['year']}-06-01") if row["tipo"] == "T1"
                else pd.to_datetime(f"{row['year']}-11-15"))
        ax.annotate(row["season_id"], xy=(date, row["sst"]),
                    xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=8, fontweight="normal",
                    arrowprops=dict(arrowstyle="-", color="#555", lw=0.8))

    for lbl, col, val in zip(TRIGGER_LBLS, TRIGGER_COLS, triggers_vals.values()):
        ax.axhline(val, color=col, lw=1.2, ls="--", alpha=0.8,
                   label=f"{lbl} = {val:.2f}C")
    ax.axhline(0, color="black", lw=0.8, ls=":", alpha=0.4)
    ax.set_ylabel("SST anomalia MODIS (C)", fontsize=9)
    ax.set_title("SST anomalia - Region Centro fishing pixels  (MODIS 2002-2025)",
                 loc="left", fontsize=11, fontweight="normal")
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax2 = axes[1]
    svals = seasonal_df["sst"].values
    ax2.hist(svals, bins=12, density=True, color="#888888", alpha=0.5,
             orientation="horizontal")
    kde = gaussian_kde(svals)
    yg  = np.linspace(svals.min() - 0.2, svals.max() + 0.2, 200)
    ax2.plot(kde(yg), yg, color="#333333", lw=1.5)
    for lbl, col, val in zip(TRIGGER_LBLS, TRIGGER_COLS, triggers_vals.values()):
        ax2.axhline(val, color=col, lw=1.5, ls="--", label=f"{lbl}={val:.2f}C")
    ax2.set_xlabel("Densidad", fontsize=9)
    ax2.set_ylabel("SST anomalia temporada (C)", fontsize=9)
    ax2.set_title("Distribucion\nanomalias de temporada", loc="left",
                  fontsize=10, fontweight="normal")
    ax2.legend(fontsize=7, frameon=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


def plot_aep(annual_loss_tons, triggers_vals, baseline_t1, baseline_t2,
             seasonal_df, out_path, mean_sst_t1=None, mean_sst_t2=None,
             aal_simulated=None, aal_historical=None):
    """
    AEP curve: annual exceedance probability (x) vs catch loss in tons (y).
    X decreasing left-to-right (common on left, rare on right).
    """
    ep_grid, ci_lo, median, ci_hi = aep_ci(annual_loss_tons)
    baseline_annual = baseline_t1 + baseline_t2

    # reverse so x goes from common (left) to rare (right) after invert_xaxis
    ep_plot  = ep_grid[::-1]
    med_plot = median[::-1]
    lo_plot  = ci_lo[::-1]
    hi_plot  = ci_hi[::-1]

    # historical annual losses
    years_t1 = dict(zip(seasonal_df[seasonal_df["tipo"] == "T1"]["year"],
                        seasonal_df[seasonal_df["tipo"] == "T1"]["sst"]))
    years_t2 = dict(zip(seasonal_df[seasonal_df["tipo"] == "T2"]["year"],
                        seasonal_df[seasonal_df["tipo"] == "T2"]["sst"]))
    hist_annual = np.array(sorted(
        sst_to_loss_tons(years_t1[y], baseline_t1) +
        sst_to_loss_tons(years_t2[y], baseline_t2)
        for y in set(years_t1) & set(years_t2)
    ))
    n_h  = len(hist_annual)
    ep_h = (n_h + 1 - np.arange(1, n_h + 1)) / (n_h + 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax_idx, (ax, y_med, y_lo, y_hi, ylabel, title) in enumerate(zip(
        axes,
        [med_plot,                        med_plot / baseline_annual * 100],
        [lo_plot,                         lo_plot  / baseline_annual * 100],
        [hi_plot,                         hi_plot  / baseline_annual * 100],
        ["Perdida anual estimada (toneladas)",
         "Perdida anual estimada (% captura baseline)"],
        ["AEP - Perdidas de captura en toneladas",
         "AEP - Perdidas relativas al baseline anual"],
    )):
        # x = loss, y = AEP
        ax.fill_betweenx(ep_plot, y_lo, y_hi, alpha=0.25, color="#888888",
                         label="CI 90% (bootstrap)")
        ax.plot(y_med, ep_plot, color="#333333", lw=2.5,
                label="Mediana bootstrap")

        y_h = hist_annual if ax_idx == 0 else hist_annual / baseline_annual * 100
        ax.scatter(y_h, ep_h, color="#e76f51", zorder=5, s=50,
                   edgecolors="white", lw=0.5, label=f"Historico ({n_h} anos)")

        for lbl, col, tval in zip(TRIGGER_LBLS, TRIGGER_COLS,
                                   triggers_vals.values()):
            loss_t = sst_to_loss_tons(tval, (baseline_t1 + baseline_t2) / 2)
            y_t = loss_t if ax_idx == 0 else loss_t / baseline_annual * 100
            if y_t <= y_med.max():
                ax.axvline(y_t, color=col, lw=1.0, ls="--", alpha=0.7)
                ax.text(y_t, 0.97, f"{lbl}\n{tval:.2f}°C",
                        transform=ax.get_xaxis_transform(),
                        fontsize=7, color=col, ha="left", va="top",
                        rotation=90, linespacing=1.3)

        # AAL lines: simulated (bootstrap) and historical observed
        for aal_val, col, short in [
            (aal_simulated,  "#43A047", "sim"),
            (aal_historical, "#1565C0", "hist"),
        ]:
            if aal_val is None:
                continue
            y_v = aal_val if ax_idx == 0 else aal_val / baseline_annual * 100
            ax.axvline(y_v, color=col, lw=1.8, ls="--", alpha=0.9, zorder=4)
            lbl = (f"{short} {y_v/1e6:.2f}M tn"
                   if ax_idx == 0 else f"{short} {y_v:.1f}%")
            ax.text(y_v, 0.97, lbl,
                    fontsize=7, color=col, ha="left", va="top",
                    transform=ax.get_xaxis_transform(), rotation=90)

        ax.set_xlabel(ylabel, fontsize=9)
        ax.set_ylabel("Probabilidad de excedencia anual (AEP)", fontsize=9)
        ax.set_title(title, loc="left", fontsize=11)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
        _idx = ax_idx
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _, idx=_idx:
                                  f"{x/1e6:.1f}M" if idx == 0 else f"{x:.0f}%"))
        ax.legend(fontsize=8, frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"AEP parametrico (Normal fit)  |  Centro  |  beta={BETA}  "
        f"|  N={N_BOOTSTRAP} anos sinteticos",
        fontsize=11, x=0.01, ha="left")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_ts  = PLOTS / "step13_sst_timeseries.png"
    out_aep = PLOTS / "step13_bootstrap_aep.png"
    if out_ts.exists() and out_aep.exists():
        print("step13 outputs exist -- skipping")
        return

    print("Loading SST anomaly (MODIS 2002-2025)...")
    daily = load_daily_sst()
    print(f"  {len(daily):,} days  "
          f"{daily.index[0].date()} - {daily.index[-1].date()}")

    print("Loading baseline catch (Centro)...")
    baseline_t1, baseline_t2, _ = load_baseline_catch()
    baseline_annual = baseline_t1 + baseline_t2
    print(f"  T1 baseline: {baseline_t1/1e6:.2f}M tn  "
          f"T2 baseline: {baseline_t2/1e6:.2f}M tn  "
          f"Total: {baseline_annual/1e6:.2f}M tn/year")

    monthly     = daily.resample("ME").mean()
    seasonal_df = season_label_from_monthly(monthly)

    sst_seas = seasonal_df["sst"].values
    triggers_vals = {lbl: float(np.percentile(sst_seas, p * 100))
                     for lbl, p in zip(TRIGGER_LBLS, TRIGGERS)}
    print("Historical SST triggers:")
    for lbl, val in triggers_vals.items():
        loss = sst_to_loss_tons(val, (baseline_t1 + baseline_t2) / 2)
        print(f"  {lbl}: {val:.3f}C -> ~{loss/1e6:.2f}M tn loss per season")

    if not out_ts.exists():
        print("\nGenerating time series figure...")
        fig = plot_timeseries(daily, seasonal_df, triggers_vals)
        fig.savefig(out_ts, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved -> {out_ts}")

    if not out_aep.exists():
        print(f"\nRunning parametric bootstrap  (Normal fit, N={N_BOOTSTRAP})...")
        t1_synth, t2_synth, mu_t1, mu_t2 = parametric_bootstrap(daily)

        loss_t1     = sst_to_loss_tons(t1_synth, baseline_t1)
        loss_t2     = sst_to_loss_tons(t2_synth, baseline_t2)
        annual_loss = loss_t1 + loss_t2

        aal_sim = float(annual_loss.mean())

        # Historical AAL from observed seasons
        t1_obs, t2_obs = get_seasonal_sst_means(daily)
        hist_t1_loss = sst_to_loss_tons(t1_obs, baseline_t1)
        hist_t2_loss = sst_to_loss_tons(t2_obs, baseline_t2)
        aal_hist = float((hist_t1_loss + hist_t2_loss).mean())

        print(f"  Mean SST anomaly: T1 = {mu_t1:.3f}°C  T2 = {mu_t2:.3f}°C")
        print(f"  AAL simulada  (bootstrap, N={N_BOOTSTRAP}): "
              f"{aal_sim/1e6:.3f}M tn ({aal_sim/baseline_annual*100:.1f}% baseline)")
        print(f"  AAL historica (obs, N={len(t1_obs)} anos): "
              f"{aal_hist/1e6:.3f}M tn ({aal_hist/baseline_annual*100:.1f}% baseline)")

        print("  Annual loss distribution (tons):")
        for p in [50, 75, 90, 95, 99]:
            v = np.percentile(annual_loss, p)
            print(f"    p{p:02d}: {v/1e6:.3f}M tn  "
                  f"({v/baseline_annual*100:.1f}% baseline)")

        plot_aep(annual_loss, triggers_vals, baseline_t1, baseline_t2,
                 seasonal_df, out_aep, mean_sst_t1=mu_t1, mean_sst_t2=mu_t2,
                 aal_simulated=aal_sim, aal_historical=aal_hist)


if __name__ == "__main__":
    main()
