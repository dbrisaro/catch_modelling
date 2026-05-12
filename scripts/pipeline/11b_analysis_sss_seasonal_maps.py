"""
Step 11b - Seasonal mean SSS anomaly maps

Analogo exacto a 11_analysis_sst_seasonal_maps.py pero usando
sss_anomaly_weekly_{year}.nc (OISSS, de step 03b).

Para cada temporada (T1 = Abr-Jul, T2 = Nov-Dic) y cada año disponible,
calcula la media espacial de la anomalia de SSS y la grafica como un mapa
de la zona costera peruana. El corredor de pesca se superpone como poligono.

Inputs:
  FEATURES/sss_anomaly_weekly_{year}.nc   (de step 03b)
  OUTPUTS/calas_all_data.csv

Outputs:
  PLOTS/11b_analysis_sss_t1_maps.png
  PLOTS/11b_analysis_sss_t2_maps.png

Skip logic: skipped si ambos outputs ya existen.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from config import FEATURES, OUTPUTS, PLOTS, SSS_YEARS

T1_DOY = (91,  212)   # Abr - Jul
T2_DOY = (305, 365)   # Nov - Dic

SEASON_META = {
    "T1": {"doy": T1_DOY, "label": "1ra temporada  (Abr - Jul)"},
    "T2": {"doy": T2_DOY, "label": "2da temporada  (Nov - Dic)"},
}

LON_W, LON_E   = -82.0, -74.0
LAT_S, LAT_N   = -16.0,  -5.0
ZONE_LAT_S     = -15.8
ZONE_LAT_N     =  -7.1
NROWS          = 3
VMAX           = 0.40   # PSU, calibrado al p98 de anomalias observadas


def compute_fishing_polygon():
    df = (pd.read_csv(OUTPUTS / "calas_all_data.csv",
                      usecols=["latitud", "longitud"], low_memory=False)
          .rename(columns={"latitud": "lat", "longitud": "lon"})
          .dropna())
    df = df[(df["lat"] >= ZONE_LAT_S) & (df["lat"] <= ZONE_LAT_N)]

    band_lo_edges = np.arange(-15, -7, 1.0)
    band_centers  = band_lo_edges + 0.5

    west_lons, east_lons, valid_lats = [], [], []
    for lo, ctr in zip(band_lo_edges, band_centers):
        band = df[(df["lat"] >= lo) & (df["lat"] < lo + 1.0)]
        if len(band) < 20:
            continue
        west_lons.append(np.percentile(band["lon"], 5))
        east_lons.append(np.percentile(band["lon"], 95))
        valid_lats.append(ctr)

    west_lons  = np.array(west_lons)
    east_lons  = np.array(east_lons)
    valid_lats = np.array(valid_lats)

    lat_full  = np.concatenate([[ZONE_LAT_S], valid_lats, [ZONE_LAT_N]])
    west_full = np.concatenate([[west_lons[0]], west_lons, [west_lons[-1]]])
    east_full = np.concatenate([[east_lons[0]], east_lons, [east_lons[-1]]])
    poly_lons = np.concatenate([west_full, east_full[::-1], [west_full[0]]])
    poly_lats = np.concatenate([lat_full,  lat_full[::-1],  [lat_full[0]]])
    return poly_lons, poly_lats


def load_seasonal_means():
    out = {"T1": {}, "T2": {}}
    for year in SSS_YEARS:
        f = FEATURES / f"sss_anomaly_weekly_{year}.nc"
        if not f.exists():
            continue
        ds    = xr.open_dataset(f)
        da    = ds["sss_anomaly"]
        for sname, meta in SEASON_META.items():
            doy_lo, doy_hi = meta["doy"]
            doy  = da.time.dt.dayofyear
            da_s = da.isel(time=(doy >= doy_lo) & (doy <= doy_hi))
            if len(da_s.time) < 3:
                continue
            mean_map = da_s.mean(dim="time").compute()
            out[sname][year] = mean_map
            print(f"  {sname} {year}: {len(da_s.time)} pasos")
        ds.close()
    return out


def plot_season_grid(season_maps, sname, outpath, fishing_poly):
    years     = sorted(season_maps.keys())
    n         = len(years)
    nrows     = NROWS
    ncols     = int(np.ceil(n / nrows))

    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-VMAX, vcenter=0, vmax=VMAX)
    proj = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 3.4, nrows * 4.2),
        subplot_kw={"projection": proj},
        gridspec_kw={"hspace": 0.12, "wspace": 0.04},
    )
    axes_flat = np.array(axes).flatten()
    poly_lons, poly_lats = fishing_poly

    for i, year in enumerate(years):
        ax = axes_flat[i]
        da = season_maps[year]

        ax.pcolormesh(da.lon.values, da.lat.values, da.values,
                      cmap=cmap, norm=norm,
                      transform=proj, shading="auto", rasterized=True)

        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color="black", zorder=3)
        ax.add_feature(cfeature.LAND,      facecolor="#e8e8e8", zorder=2)
        ax.set_extent([LON_W, LON_E, LAT_S, LAT_N], crs=proj)

        ax.plot(poly_lons, poly_lats, transform=proj,
                color="black", linewidth=0.9, linestyle="--", zorder=4, alpha=0.9)

        is_left   = (i % ncols == 0)
        is_bottom = (i >= (nrows - 1) * ncols)
        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.25, color="grey", alpha=0.5,
            xlocs=[-80, -76],
            ylocs=[-14, -10, -6],
        )
        gl.top_labels    = False
        gl.right_labels  = False
        gl.left_labels   = is_left
        gl.bottom_labels = is_bottom
        gl.xlabel_style  = {"size": 5}
        gl.ylabel_style  = {"size": 5}

        ax.set_title(str(year), fontsize=9, pad=3)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes_flat.tolist(),
                        orientation="vertical",
                        fraction=0.012, pad=0.02, shrink=0.6, aspect=25)
    cbar.set_label("Anomalia SSS media estacional (PSU)", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    plt.savefig(outpath, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Saved -> {outpath}")


def main():
    out_t1 = PLOTS / "11b_analysis_sss_t1_maps.png"
    out_t2 = PLOTS / "11b_analysis_sss_t2_maps.png"

    if out_t1.exists() and out_t2.exists():
        print("11b outputs exist -- skipping")
        return

    print("Calculando poligono de pesca...")
    fishing_poly = compute_fishing_polygon()

    print("\nCargando medias estacionales SSS...")
    season_maps = load_seasonal_means()

    if not out_t1.exists():
        print("\nPlotting T1...")
        plot_season_grid(season_maps["T1"], "T1", out_t1, fishing_poly)

    if not out_t2.exists():
        print("\nPlotting T2...")
        plot_season_grid(season_maps["T2"], "T2", out_t2, fishing_poly)


if __name__ == "__main__":
    main()
