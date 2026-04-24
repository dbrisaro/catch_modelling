"""
Step 10 - Scatter: SST Anomaly vs log(catch) [M1]  and  SST vs log(CPUE) [M2]

M1 - sin normalizacion de esfuerzo:
  5 panels: calas individuales, empresa x diario, semanal, mensual, temporada
  y-axis: log(captura)

M2 - normalizado por esfuerzo VMS (SISESAT):
  4 panels: empresa x diario, semanal, mensual, temporada
  y-axis: log(CPUE = captura / horas-esfuerzo VMS)
  (individual calas excluded: VMS pings cannot be matched to single hauls)

Regions: Norte (lat > -7.1), Centro (-15.8 < lat <= -7.1)

Inputs:
  OUTPUTS/calas_enriched.csv
  INPUTS/ihma_data/{year}/SISESAT files (anchoveta)

Outputs:
  PLOTS/step10_empresa_agregacion_m1_norte.png
  PLOTS/step10_empresa_agregacion_m1_centro.png
  PLOTS/step10_empresa_agregacion_m2_norte.png
  PLOTS/step10_empresa_agregacion_m2_centro.png
  PLOTS/step10_mapa_regiones_celdas.png

Skip logic: skipped if all output files already exist.
"""
import warnings
warnings.filterwarnings("ignore")

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as mplpatch
from scipy.stats import spearmanr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from config import OUTPUTS, PLOTS, INPUTS

PROJ     = ccrs.PlateCarree()
CELL_DEG = 1.0

REGIONS = [
    ("norte",  -7.1,  None),
    ("centro", -15.8, -7.1),
]

AGG_LEVELS = [
    ("Diario",    "date"),
    ("Semanal",   "year_week"),
    ("Mensual",   "year_month"),
    ("Temporada", "season"),
]

PING_HOURS        = 9 / 60
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
    """Return DataFrame: vessel_norm, date, season, effort_hours (daily resolution)."""
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
        if (df["Date"].max() - df["Date"].min()).days < MIN_DAYS_COVERAGE:
            continue

        df["date"] = df["Date"].dt.normalize()
        daily = df.groupby(["Cod_Barco", "date"]).size().reset_index(name="n_pings")
        daily["vessel_norm"]  = daily["Cod_Barco"].apply(normalize_vessel)
        daily["effort_hours"] = daily["n_pings"] * PING_HOURS
        daily["season"]       = season
        records.append(daily[["vessel_norm", "date", "season", "effort_hours"]])

    if not records:
        return pd.DataFrame(columns=["vessel_norm", "date", "season", "effort_hours"])
    return pd.concat(records, ignore_index=True)


# ── calas loader ──────────────────────────────────────────────────────────────
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


# ── scatter helpers ───────────────────────────────────────────────────────────
def draw_scatter(ax, x, y, colors, s, alpha, label, ylabel=None):
    if hasattr(colors, "__iter__") and not isinstance(colors, str):
        ax.scatter(x, y, s=s, alpha=alpha, c=colors, rasterized=True, zorder=3,
                   edgecolors="white", linewidths=0.3)
    else:
        ax.scatter(x, y, s=s, alpha=alpha, color=colors, rasterized=True, zorder=3,
                   edgecolors="white", linewidths=0.3)
    slope, intercept = np.polyfit(x, y, 1)
    xline = np.linspace(x.min() - 0.1, x.max() + 0.1, 100)
    ax.plot(xline, slope * xline + intercept, color="black", lw=1.5, ls="--", alpha=0.7)
    ax.axvline(0, color="grey", lw=0.8, ls=":", alpha=0.5)
    rho, p_rho = spearmanr(x, y)
    pstr = "p<0.001" if p_rho < 0.001 else ("p<0.05" if p_rho < 0.05 else f"p={p_rho:.3f}")
    ax.text(0.03, 0.97, f"rho = {rho:.2f}  ({pstr})\nN = {len(x):,}",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="square,pad=0.3", fc="none", ec="none"))
    ax.set_title(label, loc="left", fontsize=10)
    ax.set_xlabel("Anomalia SST MODIS (C)", fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    print(f"  {label:30s}  N={len(x):>6,}  rho={rho:+.3f}  {pstr}")


def make_region_map(df, cell_deg):
    fig = plt.figure(figsize=(7, 10))
    ax  = fig.add_subplot(1, 1, 1, projection=PROJ)
    EXT = [-83, -74, -17, -5]
    ax.set_extent(EXT, crs=PROJ)
    ax.add_feature(cfeature.LAND,      facecolor="#f0ead6", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8,       zorder=3)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.5, linestyle=":", zorder=3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="grey", alpha=0.5, linestyle="--")
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {"size": 7}; gl.ylabel_style = {"size": 7}

    styles = {
        "norte":  {"color": "#d6604d", "label": "Norte (lat > -7.1°)"},
        "centro": {"color": "#2166ac", "label": "Centro (-15.8° < lat <= -7.1°)"},
    }
    for reg, lat_min, lat_max in REGIONS:
        sub = df[df["lat"] > lat_min]
        if lat_max is not None:
            sub = sub[sub["lat"] <= lat_max]
        y_bot = lat_min if lat_min > -100 else EXT[2]
        y_top = lat_max if lat_max is not None else EXT[3]
        ax.add_patch(mplpatch.Rectangle(
            (EXT[0], y_bot), EXT[1] - EXT[0], y_top - y_bot,
            transform=PROJ, color=styles[reg]["color"], alpha=0.08, zorder=1))
        boundary = y_bot if lat_min > -100 else y_top
        ax.plot([EXT[0], EXT[1]], [boundary, boundary],
                color=styles[reg]["color"], lw=1.2, ls="--", transform=PROJ, zorder=4)
        sub2 = sub.copy()
        sub2["lat_g"] = (sub2["lat"] / cell_deg).round() * cell_deg
        sub2["lon_g"] = (sub2["lon"] / cell_deg).round() * cell_deg
        for _, cell in sub2[["lat_g", "lon_g"]].drop_duplicates().iterrows():
            ax.add_patch(mplpatch.Rectangle(
                (cell["lon_g"] - cell_deg / 2, cell["lat_g"] - cell_deg / 2),
                cell_deg, cell_deg, transform=PROJ,
                facecolor=styles[reg]["color"], alpha=0.35,
                edgecolor=styles[reg]["color"], linewidth=0.4, zorder=3))
    handles = [mplpatch.Patch(facecolor=styles[r]["color"], alpha=0.5, label=styles[r]["label"])
               for r, _, _ in REGIONS]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8)
    ax.set_title(f"Regiones  |  2015-2024", fontsize=10, loc="left")
    plt.tight_layout()
    return fig


# ── M1 figure: log(catch) vs SST, 5 panels ───────────────────────────────────
def make_figure_m1(sub, comp_colors, reg_companies, reg_label):
    fig, axes = plt.subplots(1, 5, figsize=(22, 6))

    point_colors = [comp_colors[c] for c in sub["company"]]
    draw_scatter(axes[0],
                 sub["modis_sst_anom"].values,
                 sub["log_catch"].values,
                 colors=point_colors, s=4, alpha=0.4,
                 label="Calas individuales",
                 ylabel="log(captura)")

    for ax, (label, tcol) in zip(axes[1:], AGG_LEVELS):
        agg = sub.groupby(["company", tcol]).agg(
            total_catch=("catch_tm", "sum"),
            mean_sst_anom=("modis_sst_anom", "mean"),
        ).reset_index()
        point_colors_agg = [comp_colors[c] for c in agg["company"]]
        draw_scatter(ax,
                     agg["mean_sst_anom"].values,
                     np.log(agg["total_catch"].values),
                     colors=point_colors_agg, s=18, alpha=0.7,
                     label=f"Empresa x {label}")

    handles = [mpatches.Patch(color=comp_colors[c], label=c) for c in reg_companies]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"M1: log(captura) ~ SST  |  empresa x tiempo  |  Region {reg_label}  |  2015-2024",
        fontsize=11, x=0.01, ha="left")
    plt.tight_layout(rect=[0, 0.08, 1, 0.97])
    return fig


# ── M2 figure: log(CPUE) vs SST, 4 panels ────────────────────────────────────
def make_figure_m2(sub, effort_empresa_daily, comp_colors, reg_companies, reg_label):
    """4-panel scatter: empresa x diario/semanal/mensual/temporada CPUE vs SST."""

    eff = effort_empresa_daily.copy()
    iso = eff["date"].dt.isocalendar()
    eff["year_week"]  = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    eff["year_month"] = eff["date"].dt.to_period("M").astype(str)

    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    any_data = False

    for ax, (label, tcol) in zip(axes, AGG_LEVELS):
        calas_agg = sub.groupby(["company", tcol]).agg(
            total_catch=("catch_tm", "sum"),
            mean_sst_anom=("modis_sst_anom", "mean"),
        ).reset_index()
        eff_agg = eff.groupby(["company", tcol])["effort_hours"].sum().reset_index()

        merged = calas_agg.merge(eff_agg, on=["company", tcol], how="inner")
        merged = merged[merged["effort_hours"] > 0].dropna(subset=["mean_sst_anom"])
        if len(merged) < 5:
            ax.set_visible(False)
            continue

        merged["log_cpue"] = np.log(merged["total_catch"] / merged["effort_hours"])
        point_colors_agg = [comp_colors[c] for c in merged["company"]]
        draw_scatter(ax,
                     merged["mean_sst_anom"].values,
                     merged["log_cpue"].values,
                     colors=point_colors_agg, s=18, alpha=0.7,
                     label=f"Empresa x {label}",
                     ylabel="log(CPUE)" if ax is axes[0] else None)
        any_data = True

    if not any_data:
        plt.close(fig)
        return None

    handles = [mpatches.Patch(color=comp_colors[c], label=c) for c in reg_companies]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"M2: log(CPUE = captura/esfuerzo VMS) ~ SST  |  empresa x tiempo  |  "
        f"Region {reg_label}  |  SISESAT 2017-2022",
        fontsize=11, x=0.01, ha="left")
    plt.tight_layout(rect=[0, 0.08, 1, 0.97])
    return fig


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    out_map = PLOTS / "step10_mapa_regiones_celdas.png"
    outputs_m1 = {reg: PLOTS / f"step10_empresa_agregacion_m1_{reg}.png" for reg, _, _ in REGIONS}
    outputs_m2 = {reg: PLOTS / f"step10_empresa_agregacion_m2_{reg}.png" for reg, _, _ in REGIONS}

    all_outputs = list(outputs_m1.values()) + list(outputs_m2.values()) + [out_map]
    if all(p.exists() for p in all_outputs):
        print("step10 outputs exist -- skipping")
        return

    # --- Load calas ---
    df = load_calas()
    df = df.dropna(subset=["modis_sst_anom", "catch_tm", "lat", "lon", "season", "company"])
    df = df[df["catch_tm"] > 0]
    df["log_catch"] = np.log(df["catch_tm"])

    iso = df["date"].dt.isocalendar()
    df["year_week"]  = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    # --- Load SISESAT effort (daily) ---
    print("Loading SISESAT effort data...")
    catalogue  = build_sisesat_catalogue()
    effort_raw = load_sisesat_effort_daily(catalogue)

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
    print(f"  {len(effort_empresa_daily):,} empresa x day effort records")

    # --- Map ---
    if not out_map.exists():
        fig = make_region_map(df, CELL_DEG)
        fig.savefig(out_map, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved -> {out_map}")

    # --- Color map ---
    companies  = sorted(df["company"].unique())
    comp_colors = {c: plt.cm.tab20.colors[i % 20] for i, c in enumerate(companies)}

    # --- Figures per region ---
    for reg, lat_min, lat_max in REGIONS:
        sub = df[df["lat"] > lat_min].copy()
        if lat_max is not None:
            sub = sub[sub["lat"] <= lat_max]
        reg_companies = sorted(sub["company"].unique())
        reg_label     = reg.capitalize()

        # M1
        out = outputs_m1[reg]
        if not out.exists():
            print(f"\n--- {reg_label} | M1 ---")
            fig = make_figure_m1(sub, comp_colors, reg_companies, reg_label)
            fig.savefig(out, dpi=130, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved -> {out}")

        # M2
        out = outputs_m2[reg]
        if not out.exists():
            print(f"\n--- {reg_label} | M2 ---")
            fig = make_figure_m2(sub, effort_empresa_daily, comp_colors, reg_companies, reg_label)
            if fig is not None:
                fig.savefig(out, dpi=130, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved -> {out}")
            else:
                print(f"  {reg_label} M2: no data to plot")


if __name__ == "__main__":
    main()
