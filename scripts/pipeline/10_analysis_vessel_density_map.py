"""
Step 08 - Vessel Density Heatmap

2D density map of cala locations pooled across all years, split by
primera and segunda temporada. Uses Gaussian smoothing.

Also produces a per-company density map for the top 4 companies.

Inputs:
  OUTPUTS/calas_all_data.csv   (from step 02)

Outputs:
  PLOTS/10_analysis_vessel_density_by_season.png
  PLOTS/10_analysis_vessel_density_by_company.png

Skip logic: skipped if both output files already exist.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter

from config import OUTPUTS, PLOTS

PROJ = ccrs.PlateCarree()

# Canonical name for each known variant (strip+upper applied first)
EMPRESA_ALIASES = {
    "PESQUERA DIAMANTE S.A.":   "DIAMANTE",
    "PESQUERA DIAMANTE S.A":    "DIAMANTE",
    "PESQUERA DIAMANTE SA":     "DIAMANTE",
    "PESQUERA EXALMAR S.A.A.":  "PESQUERA EXALMAR",
    "PESQUERA EXALMAR S.A.A":   "PESQUERA EXALMAR",
    "EXALMAR-CENTINELA":        "PESQUERA EXALMAR",
    "PESQUERA CENTINELA S.A.C": "CENTINELA",
    "PESQUERA CENTINELA S.A.C.":"CENTINELA",
    "LOS HALCONES SA":          "LOS HALCONES",
    "INVERSIONES ECCOLA":       "INVERSIONES ECCOLA",   # already canonical
    "INVERSIONES QUIAZA SAC":   "QUIAZA",
}


def normalize_empresa(name):
    if pd.isna(name):
        return np.nan
    clean = str(name).strip().upper()
    return EMPRESA_ALIASES.get(clean, clean)


def add_gridlines(ax):
    gl = ax.gridlines(crs=PROJ, draw_labels=True,
                      linewidth=0.4, color="grey", alpha=0.5, linestyle="--")
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}


def get_season_type(s):
    if pd.isna(s):
        return np.nan
    s = str(s)
    if s.startswith("1ra") or s.endswith("CN-I"):
        return "1ra"
    if s.startswith("2da") or s.endswith("CN-II"):
        return "2da"
    return np.nan


def density_map_panel(ax, lon_c, lat_c, grid, cmap, title, n_total, states_10m, ext, vmax):
    """Render one density panel onto ax."""
    H_masked = np.ma.masked_where(grid == 0, grid)
    ax.set_extent(ext, crs=PROJ)
    ax.add_feature(cfeature.LAND,      facecolor="#f0ead6", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7,       zorder=3)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.6, linestyle="-", zorder=4)
    ax.add_feature(states_10m, linewidth=0.35, edgecolor="#666666", zorder=4)
    add_gridlines(ax)
    mesh = ax.pcolormesh(lon_c, lat_c, H_masked,
                         transform=PROJ, cmap=cmap,
                         vmin=0, vmax=vmax, zorder=1)
    cb = plt.colorbar(mesh, ax=ax, shrink=0.55, pad=0.05)
    cb.set_label("calas (suavizado)", size=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_title(f"{title}\n(n = {n_total:,} calas)", fontsize=8, pad=4)


def main():
    out_season  = PLOTS / "10_analysis_vessel_density_by_season.png"
    out_company = PLOTS / "10_analysis_vessel_density_by_company.png"
    if out_season.exists() and out_company.exists():
        print("step08 outputs exist -- skipping")
        return

    df_raw = pd.read_csv(OUTPUTS / "calas_all_data.csv", low_memory=False)
    rename_map = {
        "fecha_cala": "date", "temporada": "season",
        "latitud": "lat", "longitud": "lon", "declarado_tm": "catch_tm",
    }
    df_raw = df_raw.rename(columns={k: v for k, v in rename_map.items() if k in df_raw.columns})
    df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")
    df_raw["season_type"] = df_raw["season"].apply(get_season_type)
    df_raw["empresa"] = df_raw["empresa"].apply(normalize_empresa)
    print(f"Total calas: {len(df_raw):,}")
    print(df_raw["season_type"].value_counts())
    print("\nEmpresas (normalizadas):")
    print(df_raw["empresa"].value_counts().to_string())

    LON_MIN, LON_MAX = -82, -74
    LAT_MIN, LAT_MAX = -16, -5
    LON_EDGES = np.linspace(LON_MIN, LON_MAX, 81)
    LAT_EDGES = np.linspace(LAT_MIN, LAT_MAX, 111)
    LON_C = (LON_EDGES[:-1] + LON_EDGES[1:]) / 2
    LAT_C = (LAT_EDGES[:-1] + LAT_EDGES[1:]) / 2
    EXT_ZOOM = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]

    states_10m = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_1_states_provinces_lines",
        scale="10m",
        facecolor="none",
    )

    # ------------------------------------------------------------------ #
    # Plot 1 - density by season                                          #
    # ------------------------------------------------------------------ #
    if not out_season.exists():
        grids = {}
        for stype in ["1ra", "2da"]:
            sub = df_raw[df_raw["season_type"] == stype].dropna(subset=["lat", "lon"])
            H, _, _ = np.histogram2d(sub["lon"], sub["lat"], bins=[LON_EDGES, LAT_EDGES])
            grids[stype] = gaussian_filter(H.T.astype(float), sigma=1.5)

        vmax = max(g.max() for g in grids.values())
        labels = {"1ra": "Primera temporada", "2da": "Segunda temporada"}
        cmaps  = {"1ra": "YlOrRd", "2da": "YlOrRd"}

        fig, axes = plt.subplots(1, 2, figsize=(13, 7),
                                 subplot_kw={"projection": PROJ})
        for ax, stype in zip(axes, ["1ra", "2da"]):
            H_masked = np.ma.masked_where(grids[stype] == 0, grids[stype])
            ax.set_extent(EXT_ZOOM, crs=PROJ)
            ax.add_feature(cfeature.LAND,      facecolor="#f0ead6", zorder=2)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.7,       zorder=3)
            ax.add_feature(cfeature.BORDERS,   linewidth=0.6, linestyle="-", zorder=4)
            ax.add_feature(states_10m, linewidth=0.35, edgecolor="#666666", zorder=4)
            add_gridlines(ax)
            mesh = ax.pcolormesh(LON_C, LAT_C, H_masked,
                                 transform=PROJ, cmap="YlOrRd",
                                 vmin=0, vmax=vmax, zorder=1)
            cb = plt.colorbar(mesh, ax=ax, shrink=0.55, pad=0.05)
            cb.set_label("numero de calas (suavizado)", size=8)
            cb.ax.tick_params(labelsize=7)
            n_total = int(df_raw[df_raw["season_type"] == stype]
                          .dropna(subset=["lat", "lon"]).shape[0])
            ax.set_title(f"{labels[stype]}  (n = {n_total:,} calas)", fontsize=9, pad=4)

        fig.suptitle("Densidad de ubicacion de embarcaciones por temporada  |  2015-2025",
                     fontsize=10, x=0.01, ha="left", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(out_season, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved -> {out_season}")
    else:
        print(f"Output already exists: {out_season} -- skipping")

    # ------------------------------------------------------------------ #
    # Plot 2 - density by top-4 company, one color per empresa           #
    # ------------------------------------------------------------------ #
    if not out_company.exists():
        top4 = df_raw["empresa"].value_counts().head(4).index.tolist()
        print(f"Top 4 companies: {top4}")

        # Distinct single-hue colormaps for each company
        CMAPS = ["Blues", "Oranges", "Greens", "Purples"]

        fig, axes = plt.subplots(2, 2, figsize=(13, 14),
                                 subplot_kw={"projection": PROJ})
        axes = axes.flatten()

        # Build all grids first so we can set a shared vmax
        company_data = []
        for empresa in top4:
            sub = df_raw[df_raw["empresa"] == empresa].dropna(subset=["lat", "lon"])
            H, _, _ = np.histogram2d(sub["lon"], sub["lat"], bins=[LON_EDGES, LAT_EDGES])
            grid = gaussian_filter(H.T.astype(float), sigma=1.5)
            company_data.append((empresa, sub, grid))

        shared_vmax = max(g.max() for _, _, g in company_data)

        for ax, (empresa, sub, grid), cmap in zip(axes, company_data, CMAPS):
            density_map_panel(ax, LON_C, LAT_C, grid, cmap,
                              empresa, len(sub), states_10m, EXT_ZOOM,
                              vmax=shared_vmax)

        fig.suptitle("Densidad de calas por empresa (top 4)  |  2015-2025",
                     fontsize=11, x=0.01, ha="left", y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(out_company, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved -> {out_company}")
    else:
        print(f"Output already exists: {out_company} -- skipping")


if __name__ == "__main__":
    main()
