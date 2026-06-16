"""
19_analysis_betas_by_region.py

OLS betas al nivel empresa x temporada para cada region.
Mismo modelo y nivel de agregacion que en step 14 (columna Empresa x Temporada),
pero guarda los resultados numericos a CSV en vez de solo el grafico.

Beta conservador = limite superior del IC95% (beta + 1.96*SE), es decir el valor
mas cercano a cero -- el efecto mas debil plausible. Ej: de CI [-0.91, -0.52]
el conservador es -0.52.

Regiones: Norte, Centro Norte, Centro Sur, All (Norte+Centro)

Inputs:
  OUTPUTS/calas_enriched.csv

Outputs:
  OUTPUTS/betas_by_region.csv
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUTS

REGIONS = [
    ("Norte",        -7.1,   None),
    ("Centro Norte", -11.0,  -7.1),
    ("Centro Sur",   -15.8, -11.0),
    ("All",          -15.8,   None),
]


def load_calas():
    df = pd.read_csv(OUTPUTS / "calas_enriched.csv", low_memory=False)
    df = df.dropna(subset=["modis_sst_anom", "catch_tm", "lat", "company", "season"])
    return df[df["catch_tm"] > 0].copy()


def run_ols_empresa_temporada(sub):
    agg = (sub.groupby(["company", "season"])
              .agg(total_catch=("catch_tm", "sum"),
                   mean_sst_anom=("modis_sst_anom", "mean"))
              .reset_index())
    agg = agg[agg["total_catch"] > 0].copy()
    if len(agg) < 10:
        return None
    agg["log_catch"] = np.log(agg["total_catch"])
    m = smf.ols("log_catch ~ mean_sst_anom", data=agg).fit()
    beta = float(m.params["mean_sst_anom"])
    se   = float(m.bse["mean_sst_anom"])
    return {
        "beta":         beta,
        "se":           se,
        "ci_lower":     beta - 1.96 * se,
        "ci_upper":     beta + 1.96 * se,
        "beta_conserv": beta + 1.96 * se,   # menos negativo = mas conservador
        "p":            float(m.pvalues["mean_sst_anom"]),
        "r2":           float(m.rsquared),
        "N":            int(m.nobs),
    }


def main():
    print("Cargando calas enriquecidas...")
    df = load_calas()
    print(f"  {len(df):,} calas validas\n")

    rows = []
    print(f"{'Region':<15}  {'beta':>7}  {'SE':>6}  {'CI':>20}  {'conserv':>8}  {'p':>7}  {'R2':>6}  {'N':>6}")
    print("-" * 80)

    for reg_name, lat_min, lat_max in REGIONS:
        sub = df.copy()
        if lat_min is not None:
            sub = sub[sub["lat"] > lat_min]
        if lat_max is not None:
            sub = sub[sub["lat"] <= lat_max]

        res = run_ols_empresa_temporada(sub)
        if res is None:
            print(f"  {reg_name}: datos insuficientes (N < 10)")
            continue

        ci_str = f"[{res['ci_lower']:+.3f}, {res['ci_upper']:+.3f}]"
        print(f"{reg_name:<15}  {res['beta']:>+7.3f}  {res['se']:>6.3f}  "
              f"{ci_str:>20}  {res['beta_conserv']:>+8.3f}  "
              f"{res['p']:>7.4f}  {res['r2']:>6.3f}  {res['N']:>6,}")

        rows.append({"region": reg_name, **res})

    out_df = pd.DataFrame(rows)
    out_path = OUTPUTS / "betas_by_region.csv"
    out_df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\nSaved -> {out_path}")
    print("\nNota: beta_conserv = beta + 1.96*SE (limite superior IC95%, mas cercano a 0)")


if __name__ == "__main__":
    main()
