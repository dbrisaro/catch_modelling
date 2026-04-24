"""
Step 06 - Regression: Environmental Anomalies vs Catch Tonnage

Runs semi-log OLS regressions to estimate catch elasticity to temperature and
salinity anomalies at three aggregation levels (individual cala, daily, weekly).
Also runs multivariate specifications (combined, interaction, quadratic).

Inputs:
  OUTPUTS/calas_enriched.csv   (from step 05)

Outputs:
  Prints regression tables to stdout (no files saved).
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from config import OUTPUTS


def reg_table(data, regressors, label):
    """Semi-log OLS. Returns a DataFrame of semi-elasticities."""
    rows = []
    for var in regressors:
        if var not in data.columns:
            continue
        m = smf.ols(f"log_catch ~ {var}", data=data.dropna(subset=[var])).fit()
        beta = m.params[var]
        pval = m.pvalues[var]
        ci_lo, ci_hi = m.conf_int().loc[var]
        rows.append({
            "variable":         var,
            "beta":             round(beta, 4),
            "% change per unit": round(beta * 100, 2),
            "95% CI (%)":       f"[{ci_lo*100:.2f}, {ci_hi*100:.2f}]",
            "p-value":          round(pval, 4),
            "R2":               round(m.rsquared, 4),
            "N":                int(m.nobs),
        })
    result = pd.DataFrame(rows).set_index("variable")
    print(f"\n{'='*65}\n{label}\n{'='*65}")
    print(result.to_string())
    return result


def multi_reg(data, formula, label, level):
    """Run OLS and return a summary row + print coefficients."""
    subset = data.dropna(subset=["log_catch"]).copy()
    vars_in = [v.strip() for v in formula.split("~")[1]
               .replace("*", "+").replace("I(", "").replace("**2)", "").split("+")]
    vars_in = [v.strip() for v in vars_in if v.strip() in subset.columns]
    subset = subset.dropna(subset=vars_in)
    if len(subset) < 10:
        print(f"  {label} [{level}]: insufficient data")
        return None
    m = smf.ols(formula, data=subset).fit()
    coefs = pd.DataFrame({"coef": m.params, "p-value": m.pvalues}).round(4)
    coefs = coefs[coefs.index != "Intercept"]
    print(f"\n--- {label} [{level}]  R2={m.rsquared:.4f}  N={int(m.nobs)} ---")
    print(coefs.to_string())
    return {"label": label, "level": level, "R2": round(m.rsquared, 4), "N": int(m.nobs)}


def main():
    df = pd.read_csv(OUTPUTS / "calas_enriched.csv", parse_dates=["date"])

    df_clean = df.dropna(subset=["modis_sst_anom", "catch_tm"])
    df_clean = df_clean[df_clean["catch_tm"] > 0].copy()

    Q1, Q3 = df_clean["catch_tm"].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df_clean = df_clean[
        (df_clean["catch_tm"] >= Q1 - 1.5 * IQR) &
        (df_clean["catch_tm"] <= Q3 + 1.5 * IQR)
    ].copy()

    df_clean["log_catch"] = np.log(df_clean["catch_tm"])
    print(f"N calas: {len(df_clean):,}")

    # --- daily and weekly aggregations ---
    agg_cols = {
        "catch_tm":       ("catch_tm",       "sum"),
        "modis_sst_anom": ("modis_sst_anom",  "mean"),
    }

    daily = df_clean.groupby("date").agg(**agg_cols).reset_index()
    daily["log_catch"] = np.log(daily["catch_tm"])

    iso = df_clean["date"].dt.isocalendar()
    df_clean["year_week"] = (iso.year.astype(str) + "-W"
                             + iso.week.astype(str).str.zfill(2))
    weekly = df_clean.groupby("year_week").agg(**agg_cols).reset_index()
    weekly["log_catch"] = np.log(weekly["catch_tm"])

    # --- univariate semi-log regressions ---
    regressors = ["modis_sst_anom"]

    df_ind = df_clean.copy()
    ind        = reg_table(df_ind,  regressors, "Individual cala")
    daily_res  = reg_table(daily,   regressors, "Daily aggregation")
    weekly_res = reg_table(weekly,  regressors, "Weekly aggregation")

    comparison = pd.concat([
        ind[["% change per unit", "p-value", "R2"]].add_suffix(" (cala)"),
        daily_res[["% change per unit", "p-value", "R2"]].add_suffix(" (daily)"),
        weekly_res[["% change per unit", "p-value", "R2"]].add_suffix(" (weekly)"),
    ], axis=1)
    print("\n\n=== SUMMARY: % change in catches per 1-unit anomaly ===")
    print(comparison.to_string())

    # --- multivariate specifications ---
    datasets = {
        "cala":   df_ind,
        "daily":  daily,
        "weekly": weekly,
    }
    specs = {
        "linear":    "log_catch ~ modis_sst_anom",
        "quadratic": "log_catch ~ modis_sst_anom + I(modis_sst_anom**2)",
    }

    summary_rows = []
    for spec_name, formula in specs.items():
        print(f"\n{'#'*65}\nSPEC: {spec_name}\n{'#'*65}")
        for level, data in datasets.items():
            row = multi_reg(data, formula, spec_name, level)
            if row:
                summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows).pivot(
        index="label", columns="level", values="R2"
    )[["cala", "daily", "weekly"]]
    print("\n\n=== R2 comparison across specs and levels ===")
    print(df_summary.to_string())


if __name__ == "__main__":
    main()
