import numpy as np 
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.colors as mcolors
from scipy.stats import pearsonr
from pyproj import datadir

try:
    datadir.set_data_dir("/home/jupyter-daniela/.conda/envs/peru_environment/share/proj")
except:
    pass

class CatchesCorrelator:
    """
    Unified class to correlate catches with environmental variables (SST, SSS, SSD).
    Uses configuration-driven approach to handle differences between variables.
    """
    
    def __init__(self, data_dir, variable_type, config_path=None):

        self.data_dir = Path(data_dir)
        self.variable_type = variable_type.lower()
        
        if config_path is None:
            config_path = Path(__file__).parent / "correlation_config.json"
        
        with open(config_path, 'r') as f:
            self.full_config = json.load(f)
        
        if self.variable_type not in self.full_config:
            raise ValueError(f"Variable type '{variable_type}' not found in configuration")
        
        self.config = self.full_config[self.variable_type]
        self.default_settings = self.full_config["default_settings"]
        
        self.catches_data = pd.DataFrame()
        self.env_data = xr.Dataset()
        self.env_weekly = xr.Dataset()
        self.env_df = pd.DataFrame()
        self.merge_datasets = pd.DataFrame()
        self.results = pd.DataFrame()
        self.grouped = pd.DataFrame()
        self.results_beta = pd.DataFrame()
        
    def load_catches_data(self, filename):
        self.catches_data = pd.read_csv(self.data_dir / filename)
        
    def load_environmental_data(self, filepath):
        self.env_data = xr.open_dataset(filepath)
        
    def preprocess_catches_data(self):
        def parse_temporada(row):
            parte, anio = row.split()
            return int(anio)
        
        self.catches_data["anio"] = self.catches_data["temporada"].apply(parse_temporada)
        self.catches_data["fecha"] = pd.to_datetime(
            self.catches_data["anio"].astype(str), format="%Y"
        ) + pd.to_timedelta((self.catches_data["semana"] - 1) * 7, unit="D")
        
    def preprocess_environmental_data(self):
        years = np.arange(2015, 2025)
        ds_list = []
        
        for y in years:
            ds_year = self.env_data.sel(time=slice(f"{y}-01-01", f"{y}-12-31"))
            
            if self.config["aggregation_method"] == "max":
                ds_year_weekly = ds_year.resample(time="7D", origin=f"{y}-01-01").max()
            else: 
                ds_year_weekly = ds_year.resample(time="7D", origin=f"{y}-01-01").mean()
                
            ds_list.append(ds_year_weekly)
        
        env_weekly = xr.concat(ds_list, dim="time")
        
        semanas = env_weekly["time"].dt.isocalendar().week
        season1 = self.default_settings["season_weeks"]["season1"]
        season2_start = self.default_settings["season_weeks"]["season2_start"]
        season2_end = self.default_settings["season_weeks"]["season2_end"]
        
        mask_temporada1 = (semanas >= season1["start"]) & (semanas <= season1["end"])
        mask_temporada2 = (semanas >= season2_start) | (semanas <= season2_end)
        mask_temporadas = mask_temporada1 | mask_temporada2
        
        self.env_weekly = env_weekly.where(mask_temporadas, drop=True)
        
    def convert_xarray_to_dataframe(self):
        """Convert xarray dataset to pandas DataFrame with appropriate column names."""
        var_name = self.config["variable_name"]
        
        if var_name not in self.env_weekly.data_vars:
            raise ValueError(f"El dataset no contiene la variable '{var_name}'")
        
        df = self.env_weekly[var_name].to_dataframe().reset_index()
        df["semana"] = df["time"].dt.isocalendar().week
        
        df.rename(columns=self.config["column_renames"], inplace=True)
        
        if self.config["coordinate_rounding"]:
            precision = self.config["coordinate_precision"]
            df["lat_bin"] = df["lat_bin"].round(precision)
            df["lon_bin"] = df["lon_bin"].round(precision)
        
        self.env_df = df
        
    def merge_datasets_catches_env(self):
        """Merge catch and environmental datasets."""
        if self.catches_data.empty or self.env_df.empty:
            raise ValueError("Ambos datasets deben estar cargados y preprocesados")
        
        env_column = self.config["variable_name"]
        
        merged = pd.merge(
            self.catches_data[["fecha", "lat_bin", "lon_bin", "suma_pescado", "temporada"]],
            self.env_df[["fecha", "lat_bin", "lon_bin", env_column]],
            on=["fecha", "lat_bin", "lon_bin"],
            how="inner"
        )
        
        self.merge_datasets = merged[merged["suma_pescado"] != 0]
        
    def grouped_merged(self, step=3):
        """Group merged data spatially."""
        merged = self.merge_datasets
        
        if step == 1:
            self.grouped = merged.copy()
            return
        
        df = merged.copy()
        df["lat_idx"] = df["lat_bin"].rank(method="dense").astype(int)
        df["lon_idx"] = df["lon_bin"].rank(method="dense").astype(int)
        
        df["lat_block"] = df["lat_idx"] // step
        df["lon_block"] = df["lon_idx"] // step
        
        df["lat_center"] = df.groupby("lat_block")["lat_bin"].transform("mean")
        df["lon_center"] = df.groupby("lon_block")["lon_bin"].transform("mean")
        
        env_column = self.config["variable_name"]
        
        agg_dict = {
            "suma_pescado": "sum",
            "lat_center": "first",
            "lon_center": "first"
        }
        
        if self.config["aggregation_method"] == "max":
            agg_dict[env_column] = "max"
        else:
            agg_dict[env_column] = "mean"
        
        grouped = (
            df.groupby(["fecha", "temporada", "lat_block", "lon_block"], as_index=False)
            .agg(agg_dict)
            .rename(columns={"lat_center": "lat_bin", "lon_center": "lon_bin"})
        )
        
        self.grouped = grouped
        
    def correlate(self):
        """Calculate spatial correlations between catches and environmental variable."""
        results = []
        grouped = self.grouped
        
        var_name = self.config["variable_name"] 
        env_columns = [col for col in grouped.columns if col in [var_name] or 
                      col in self.config["column_renames"].values()]
        env_column = env_columns[0] if env_columns else var_name
        
        correlation_threshold = self.default_settings["correlation_threshold"]
        
        for (lat, lon), df_group in grouped.groupby(["lat_bin", "lon_bin"]):
            df_group = df_group[["suma_pescado", env_column]].dropna()
            n_registros = df_group.shape[0]
            
            if n_registros > correlation_threshold:
                if self.variable_type == "ssd":
                    corr, pval = pearsonr(df_group["suma_pescado"], df_group[env_column])
                    signif = pval < self.default_settings["significance_level"]
                    results.append({
                        "lat": lat,
                        "lon": lon, 
                        "correlation": corr,
                        "p_value": pval,
                        "significant": signif,
                        "n_registros": n_registros
                    })
                else:
                    corr = df_group[["suma_pescado", env_column]].corr().iloc[0, 1]
                    results.append({
                        "lat": lat,
                        "lon": lon,
                        "correlation": corr,
                        "n_registros": n_registros
                    })
        
        self.results = pd.DataFrame(results)
        
    def save_results(self, output_filename):
        """Save correlation results to CSV."""
        self.results.to_csv(self.data_dir / output_filename, index=False)
        
    def visualizar_correlacion(self, shapefile_path=None, bathymetry_path=None, mode="grid", if_significant=True):
        """Visualize spatial correlation patterns."""
        if self.results.empty:
            raise ValueError("Primero debes correr el método correlate()")
        
        data = self.results
        if if_significant and "significant" in data.columns:
            data = data[data["significant"]]
        
        fig = plt.figure(figsize=(8, 8))
        ax = plt.axes([0.05, 0.05, 0.85, 0.9])
        
        if shapefile_path:
            boundary = gpd.read_file(shapefile_path)
            boundary.plot(ax=ax, facecolor="lightgrey", edgecolor="black", linewidth=0.5)
        
        if bathymetry_path:
            bathy = xr.open_dataset(bathymetry_path)
            domain = self.default_settings["spatial_domain"]

            lats = bathy["lat"].values
            if lats[0] < lats[-1]:
                bathy = bathy.sel(
                    lon=slice(domain["lon_min"], domain["lon_max"]),
                    lat=slice(domain["lat_min"], domain["lat_max"])
                )
            else:
                bathy = bathy.sel(
                    lon=slice(domain["lon_min"], domain["lon_max"]),
                    lat=slice(domain["lat_max"], domain["lat_min"])
                )
            
            bathy_settings = self.default_settings["bathymetry_settings"]
            bathy_mesh = ax.pcolormesh(
                bathy["lon"], bathy["lat"], bathy["elevation"],
                cmap=bathy_settings["cmap"], 
                alpha=bathy_settings["alpha"],
                shading="auto",
                vmin=bathy_settings["vmin"], 
                vmax=bathy_settings["vmax"]
            )

        if mode == "scatter":
            sc = ax.scatter(
                data["lon"], data["lat"],
                c=data["correlation"],
                cmap="RdBu_r",
                vmin=-1, vmax=1,
                s=50
            )
            plt.colorbar(sc, ax=ax, label="Correlación")
            
        elif mode == "grid":
            grid = data.pivot_table(index="lat", columns="lon", values="correlation")
            lons = grid.columns.values
            lats = grid.index.values
            Z = grid.values
            
            grid_offset = self.config["grid_offset"]
            
            if self.variable_type in ["ssd", "sss"]:
                Z[Z < -0.3] = np.nan
            
            levels = np.linspace(-0.8, 0.8, 17)
            norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=256)
            
            mesh = ax.pcolormesh(
                lons, lats, Z + grid_offset,
                cmap="RdBu_r", norm=norm,
                shading="auto"
            )
            
            cax = plt.axes([0.92, 0.1, 0.02, 0.8])
            plt.colorbar(mesh, cax=cax, ticks=levels, label="Correlación")
        else:
            raise ValueError("mode debe ser 'scatter' o 'grid'")
        
        domain = self.default_settings["spatial_domain"]
        ax.set_xlim([domain["lon_min"], domain["lon_max"]])
        ax.set_ylim([domain["lat_min"], domain["lat_max"]])
        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")
        ax.set_title(self.config["correlation_title"], fontsize=10, loc="left")
        
        plt.show()
        
        
    def calcular_beta_espacial(self, if_significant=True, if_difference_catches=True):
        """Calculate spatial beta coefficients (only for SSD)."""
        if not self.config["has_beta_analysis"]:
            raise ValueError(f"Análisis beta no está disponible para {self.variable_type.upper()}")
        
        if self.grouped.empty:
            raise ValueError("Primero debes ejecutar grouped_merged()")
        if self.results.empty:
            raise ValueError("Primero debes ejecutar correlate()")
        
        # Get environmental variable column name
        var_name = self.config["variable_name"]
        env_columns = [col for col in self.grouped.columns if col in [var_name] or 
                      col in self.config["column_renames"].values()]
        env_column = env_columns[0] if env_columns else var_name
        
        if if_significant and "significant" in self.results.columns:
            results_filtered = self.results[self.results["significant"]]
        else:
            results_filtered = self.results
        
        coords_validos = set(zip(results_filtered["lat"], results_filtered["lon"]))
        resultados = []
        
        correlation_threshold = self.default_settings["correlation_threshold"]
        
        for (lat, lon), df_group in self.grouped.groupby(["lat_bin", "lon_bin"]):
            if (lat, lon) not in coords_validos:
                continue
            
            df_group = df_group[(df_group["suma_pescado"] > 0) & df_group[env_column].notna()]
            
            if len(df_group) > correlation_threshold:
                x = df_group[env_column].values
                y = df_group["suma_pescado"].values
                
                if if_difference_catches:
                    y = y - np.mean(y)
                
                A = np.vstack([x, np.ones(len(x))]).T
                beta, alpha = np.linalg.lstsq(A, y, rcond=None)[0]
                
                resultados.append({
                    "lat": lat,
                    "lon": lon,
                    "beta": beta,
                    "alpha": alpha,
                    "n": len(df_group)
                })
        
        self.results_beta = pd.DataFrame(resultados)
        
    def visualizar_beta(self, shapefile_path=None, bathymetry_path=None):
        """Visualize spatial beta coefficients (only for SSD)."""
        if not self.config["has_beta_analysis"]:
            raise ValueError(f"Visualización beta no está disponible para {self.variable_type.upper()}")
        
        if not hasattr(self, "results_beta") or self.results_beta.empty:
            raise ValueError("Primero debes ejecutar calcular_beta_espacial()")
        
        fig = plt.figure(figsize=(8, 8))
        ax = plt.axes([0.05, 0.05, 0.85, 0.9])
        
        if shapefile_path:
            boundary = gpd.read_file(shapefile_path)
            boundary.plot(ax=ax, facecolor="lightgrey", edgecolor="black", linewidth=0.5)
        
        if bathymetry_path:
            bathy = xr.open_dataset(bathymetry_path)
            domain = self.default_settings["spatial_domain"]

            lats = bathy["lat"].values
            if lats[0] < lats[-1]:
                bathy = bathy.sel(
                    lon=slice(domain["lon_min"], domain["lon_max"]),
                    lat=slice(domain["lat_min"], domain["lat_max"])
                )
            else:
                bathy = bathy.sel(
                    lon=slice(domain["lon_min"], domain["lon_max"]),
                    lat=slice(domain["lat_max"], domain["lat_min"])
                )
            
            
            bathy_settings = self.default_settings["bathymetry_settings"]
            bathy_mesh = ax.pcolormesh(
                bathy["lon"], bathy["lat"], bathy["elevation"],
                cmap=bathy_settings["cmap"],
                alpha=bathy_settings["alpha"],
                shading="auto",
                vmin=bathy_settings["vmin"],
                vmax=bathy_settings["vmax"]
            )
        
        grid = self.results_beta.pivot_table(index="lat", columns="lon", values="beta")
        lons = grid.columns.values
        lats = grid.index.values
        Z = grid.values
        
        vmin, vmax = np.nanpercentile(Z, [5, 95])
        Z[Z < -3000] = np.nan
        
        print(f"Media beta: {np.nanmean(Z)}")
        print(f"Celdas válidas: {np.count_nonzero(~np.isnan(Z))}")
        
        vmin, vmax = 0, 8000
        
        mesh = ax.pcolormesh(lons, lats, Z, cmap="Reds", shading="auto", vmin=vmin, vmax=vmax)
        
        cax = plt.axes([0.92, 0.1, 0.02, 0.8])
        plt.colorbar(mesh, cax=cax, label="Beta producción vs densidad")
        
        domain = self.default_settings["spatial_domain"]
        ax.set_xlim([domain["lon_min"], domain["lon_max"]])
        ax.set_ylim([domain["lat_min"], domain["lat_max"]])
        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")
        ax.set_title(self.config["beta_title"], fontsize=10, loc="left")
        
        plt.show()
        
    def visualizar_distribucion_anomalias(self, if_significant=True, bins=30, show_kde=True):
        """Visualize distribution of environmental anomalies (only for SSD)."""
        if not self.config["has_distribution_analysis"]:
            raise ValueError(f"Análisis de distribución no está disponible para {self.variable_type.upper()}")
        
        if self.results.empty:
            raise ValueError("Primero debes correr el método correlate()")
        if not hasattr(self, "env_data"):
            raise AttributeError("El objeto no tiene datos ambientales disponibles.")
        
        data = self.results
        if if_significant and "significant" in data.columns:
            data = data[data["significant"]]
        
        if data.empty:
            raise ValueError("No hay puntos para mostrar la distribución.")
        
        selected_lats = data["lat"].values
        selected_lons = data["lon"].values
        
        var_name = self.config["variable_name"]
        env_sel = self.env_data.sel(latitude=selected_lats, longitude=selected_lons, method="nearest")
        env_anoms = env_sel[var_name].values.flatten()
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(env_anoms, bins=bins, density=True, alpha=0.6, color="steelblue", edgecolor="white")
        
        if show_kde:
            from scipy.stats import gaussian_kde
            valid_anoms = env_anoms[~np.isnan(env_anoms)]
            if len(valid_anoms) > 10:
                kde = gaussian_kde(valid_anoms)
                x_grid = np.linspace(np.nanmin(env_anoms), np.nanmax(env_anoms), 200)
                ax.plot(x_grid, kde(x_grid), color="darkblue", lw=0.5)
        
        # Add percentiles
        p95 = np.nanpercentile(env_anoms, 95)
        p99 = np.nanpercentile(env_anoms, 99)
        
        ax.axvline(p95, color="orange", linestyle="--", label=f"P95: {p95:.2f}")
        ax.axvline(p99, color="red", linestyle="--", label=f"P99: {p99:.2f}")
        ax.axvline(np.nanmean(env_anoms), color="k", linestyle="--", 
                  label=f"Media: {np.nanmean(env_anoms):.2f}")
        
        ax.set_xlabel(f"Anomalía de {self.config['full_name'].lower()} ({self.config['units']})")
        ax.set_ylabel("Densidad de probabilidad")
        ax.set_title(self.config["distribution_title"], loc='left', fontsize=11)
        ax.set_xlim([-2, 2])
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.show()
