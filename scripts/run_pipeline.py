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
    python run_pipeline.py --steps 5 6 7

  Force re-run a step even if outputs exist:
    python run_pipeline.py --force --steps 7

  List all steps:
    python run_pipeline.py --list

Pipeline stages
---------------
  Step  File                                  Output(s)
  ----  ----                                  ---------
  01    01_wrangling_hycom_resample_daily      features/hycom_water_temp_daily_{year}.nc
                                              features/hycom_salinity_daily_{year}.nc

  02    02_wrangling_hycom_temp_anomalies      features/hycom_water_temp_climatology.nc
                                              features/hycom_water_temp_anomaly_daily_{year}.nc

  03    03_wrangling_sst_modis_anomalies       features/sst_daily_{year}.nc
                                              features/sst_climatology_doy.nc
                                              features/sst_anomaly_daily_{year}.nc

  04    04_wrangling_chl_modis_anomalies       features/chl_daily_{year}.nc
                                              features/chl_climatology_doy.nc
                                              features/chl_anomaly_daily_{year}.nc

  05    05_wrangling_calas_consolidation       outputs/calas_all_data.csv

  06    06_wrangling_calas_daily_aggregation   outputs/calas_daily_with_coordinates.csv

  07    07_wrangling_calas_enrichment          outputs/calas_enriched.csv

  08    08_wrangling_catch_summary_by_region   outputs/catch_summary_by_region.csv
                                              outputs/catch_summary_by_region.md
                                              plots/08_wrangling_catch_timeseries.png

  09    09_analysis_hycom_modis_crossval       plots/09_analysis_hycom_modis_crossval.png

  10    10_analysis_vessel_density_map         plots/10_analysis_vessel_density_by_season.png
                                              plots/10_analysis_vessel_density_by_company.png

  11    11_analysis_sst_seasonal_maps          plots/11_analysis_sst_t1_maps.png
                                              plots/11_analysis_sst_t2_maps.png

  12    12_analysis_sst_spatial_homogeneity    plots/12_analysis_sst_ridge.png
                                              plots/12_analysis_sst_correlations.png

  13    13_analysis_sst_catch_scatter          plots/13_analysis_sst_catch_scatter_norte.png
                                              plots/13_analysis_sst_catch_scatter_centro.png

  14    14_analysis_sst_catch_ols              plots/14_analysis_sst_catch_ols_betas.png

  15    15_pricing_bootstrap_aep               plots/15_pricing_bootstrap_aep_ramp.png

  16    16_pricing_trigger_design              plots/16_pricing_spatial_comparison.png
                                              plots/16_pricing_payout_curves.png

  17    17_pricing_baseline_report             outputs/report_baseline.md

  18    18_pricing_document                    outputs/report_pricing_document.md

  19    19_analysis_betas_by_region            outputs/betas_by_region.csv

  20    20_pricing_empresa_report              outputs/report_pricing_{empresa}.md

  21    21_pricing_reinsurance                 outputs/21_reinsurance_analysis.md
                                              plots/21_reinsurance_aep.png

  21b   21b_region_map                         plots/21b_region_map.png

  21c   21c_payout_curves                      plots/21c_payout_curves.png
"""
import sys
import argparse
import importlib.util
from pathlib import Path

SCRIPTS_DIR  = Path(__file__).parent
PIPELINE_DIR = SCRIPTS_DIR / "pipeline"

sys.path.insert(0, str(PIPELINE_DIR))

STEPS = {
    "01":  "01_wrangling_hycom_resample_daily",
    "02":  "02_wrangling_hycom_temp_anomalies",
    "03":  "03_wrangling_sst_modis_anomalies",
    "04":  "04_wrangling_chl_modis_anomalies",
    "05":  "05_wrangling_calas_consolidation",
    "06":  "06_wrangling_calas_daily_aggregation",
    "07":  "07_wrangling_calas_enrichment",
    "08":  "08_wrangling_catch_summary_by_region",
    "09":  "09_analysis_hycom_modis_crossval",
    "10":  "10_analysis_vessel_density_map",
    "11":  "11_analysis_sst_seasonal_maps",
    "12":  "12_analysis_sst_spatial_homogeneity",
    "13":  "13_analysis_sst_catch_scatter",
    "14":  "14_analysis_sst_catch_ols",
    "15":  "15_pricing_bootstrap_aep",
    "16":  "16_pricing_trigger_design",
    "17":  "17_pricing_baseline_report",
    "18":  "18_pricing_document",
    "19":  "19_analysis_betas_by_region",
    "20":  "20_pricing_empresa_report",
    "21":  "21_pricing_reinsurance",
    "21b": "21b_region_map",
    "21c": "21c_payout_curves",
}

STEP_ORDER = [
    "01", "02", "03", "04", "05", "06", "07", "08",
    "09", "10", "11", "12", "13", "14", "15", "16",
    "17", "18", "19", "20", "21", "21b", "21c",
]


def _load_step(step_id):
    filename = STEPS[step_id] + ".py"
    file_path = PIPELINE_DIR / filename
    spec = importlib.util.spec_from_file_location(f"_step_{step_id}", file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_step(step_id, force=False):
    filename = STEPS[step_id]
    print(f"\n{'='*65}")
    print(f"  STEP {step_id:>3s}  {filename}")
    print(f"{'='*65}")
    if force:
        print("  --force: step will overwrite existing outputs if applicable")
    mod = _load_step(step_id)
    mod.main()


def _normalise(s):
    """Accept '1' -> '01', '21b' -> '21b', etc."""
    if s in STEPS:
        return s
    padded = s.zfill(2)
    if padded in STEPS:
        return padded
    return None


def main():
    parser = argparse.ArgumentParser(description="Peru catch modeling pipeline")
    parser.add_argument(
        "--steps", nargs="+", metavar="STEP",
        help="Steps to run (e.g. --steps 5 6 7). Defaults to all steps.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Hint to steps that they should re-run (individual steps may ignore this).",
    )
    parser.add_argument(
        "--list", action="store_true", help="List all steps and exit.",
    )
    args = parser.parse_args()

    if args.list:
        print("Available steps:")
        for sid in STEP_ORDER:
            print(f"  {sid:>3s}  {STEPS[sid]}")
        return

    if args.steps:
        selected = set()
        for s in args.steps:
            norm = _normalise(s)
            if norm:
                selected.add(norm)
            else:
                print(f"Unknown step: {s} - skipping")
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
