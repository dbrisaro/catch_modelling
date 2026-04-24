"""
Peru Catch Modeling Pipeline
============================

Runs the full pipeline in order. Intermediate files are not rebuilt if they
already exist (each step checks before running).

Usage
-----
  Run all steps:
    python run_pipeline.py

  Run specific steps only (space-separated numbers):
    python run_pipeline.py --steps 2 3 5

  Force re-run a step even if outputs exist:
    python run_pipeline.py --force --steps 5

Pipeline stages
---------------
  Step  Input(s)                              Output(s)
  ----  ------                                --------
  00    sources/hycom/hycom_*.nc              features/hycom_water_temp_daily_{year}.nc
                                              features/hycom_salinity_daily_{year}.nc

  01    features/hycom_water_temp_daily_*.nc  features/hycom_water_temp_climatology.nc
                                              features/hycom_water_temp_anomaly_daily_{year}.nc

  02    inputs/ihma_data/{year}/*.csv         outputs/calas_all_data.csv

  03    outputs/calas_all_data.csv            outputs/calas_daily_with_coordinates.csv

  04    sources/chl_peru/AQUA_MODIS.*.nc      features/chl_daily_{year}.nc
                                              features/chl_climatology_doy.nc
                                              features/chl_anomaly_daily_{year}.nc

  04b   sources/sst_peru/AQUA_MODIS.*.nc      features/sst_daily_{year}.nc
                                              features/sst_climatology_doy.nc
                                              features/sst_anomaly_daily_{year}.nc

  05    outputs/calas_daily_with_coordinates  outputs/calas_enriched.csv
        + all features/*.nc

  06    outputs/calas_enriched.csv     (stdout: regression tables)

  07    outputs/calas_enriched.csv     outputs/step07_crossval.png
        + features/*.nc

  08    outputs/calas_all_data.csv            outputs/step08_vessel_density_by_season.png
                                              outputs/step08_vessel_density_by_company.png

  09    outputs/calas_enriched.csv            outputs/step09_nino12_catch.png
        + NOAA CPC URL (fallback embedded)

  10    outputs/calas_enriched.csv            outputs/step10_sst_catch_scatter.png

  11    outputs/calas_enriched.csv            outputs/step11_ols_diagnostics.png
        + NOAA CPC URL (fallback embedded)    outputs/step11_sst_partial_response.png

  12    outputs/calas_enriched.csv            outputs/step12_tobit_decay.png
        + features/sst_anomaly_daily_*.nc
        + NOAA CPC URL (fallback embedded)

  13    outputs/calas_all_data.csv            outputs/step13_spatial_tobit.png
        + features/SST_weekly_*.nc            outputs/step13_censure_map.png
        + features/sst_anomaly_daily_*.nc     outputs/step13_sensitivity_domain.png
                                              outputs/step13_by_region.png

  15    outputs/calas_enriched.csv            outputs/step15_spatial_comparison.png
                                              outputs/step15_payout_curves.png

  16    features/sst_anomaly_daily_*.nc       outputs/step16_sst_ridge.png
                                              outputs/step16_sst_correlations.png
"""
import sys
import argparse
import importlib
import os
from pathlib import Path

# make pipeline package importable when run from the scripts/ directory
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR / "pipeline"))


STEPS = {
    "00":  "step00_hycom_resample",
    "01":  "step01_hycom_temp_anomalies",
    "02":  "step02_calas_all_years",
    "03":  "step03_calas_daily",
    "04":  "step04_chl_data",
    "04b": "step04b_sst_modis",
    "05":  "step05_data_enrichment",
    "06":  "step06_regression",
    "07":  "step07_catch_habitat",
    "08":  "step08_vessel_density",
    "09":  "step09_enso_catch",
    "10":  "step10_sst_scatter",
    "11":  "step11_ols",
    "13":  "step13_bootstrap_aep",
    "15":  "step15_trigger_design",
    "16":  "step16_sst_homogeneity",
    "17":  "step17_seasonal_sst_maps",
    "18":  "step18_client_company_analysis",
}

STEP_ORDER = ["00", "01", "02", "03", "04", "04b", "05", "06", "07", "08", "09", "10", "11", "13", "15", "16", "17", "18"]


def run_step(step_id, force=False):
    module_name = STEPS[step_id]
    print(f"\n{'='*65}")
    print(f"  STEP {step_id:>3s}  {module_name}")
    print(f"{'='*65}")

    if force:
        # remove skip-sentinel files so each step re-runs
        # (steps implement their own skip logic by checking output files)
        print("  --force: step will overwrite existing outputs if applicable")

    mod = importlib.import_module(module_name)
    mod.main()


def main():
    parser = argparse.ArgumentParser(description="Peru catch modeling pipeline")
    parser.add_argument(
        "--steps", nargs="+", metavar="STEP",
        help="Steps to run (e.g. --steps 2 3 5). Defaults to all steps."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Hint to steps that they should re-run (individual steps may ignore this)."
    )
    parser.add_argument(
        "--list", action="store_true", help="List all steps and exit."
    )
    args = parser.parse_args()

    if args.list:
        print("Available steps:")
        for sid in STEP_ORDER:
            print(f"  {sid:>3s}  {STEPS[sid]}")
        return

    if args.steps:
        # normalise: accept '4' as '04', '4b' as '04b', etc.
        selected = []
        for s in args.steps:
            if s in STEPS:
                selected.append(s)
            elif s.zfill(2) in STEPS:
                selected.append(s.zfill(2))
            elif s.zfill(2).replace("0b", "b") in STEPS:
                selected.append(s.zfill(2))
            else:
                print(f"Unknown step: {s} -- skipping")
        steps_to_run = [s for s in STEP_ORDER if s in selected]
    else:
        steps_to_run = STEP_ORDER

    print(f"Running steps: {steps_to_run}")

    for step_id in steps_to_run:
        try:
            run_step(step_id, force=args.force)
        except Exception as exc:
            print(f"\nERROR in step {step_id}: {exc}")
            import traceback
            traceback.print_exc()
            print(f"\nPipeline aborted at step {step_id}.")
            sys.exit(1)

    print(f"\n{'='*65}")
    print("  Pipeline complete.")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
