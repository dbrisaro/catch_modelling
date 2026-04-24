"""
Pipeline configuration - all shared paths in one place.
"""
from pathlib import Path
from pyproj import datadir as _pj

_pj.set_data_dir("/home/jupyter-daniela/.conda/envs/peru_environment/share/proj")

# --- source data ---
HYCOM_SRC = Path("/home/jupyter-daniela/suyana/sources/hycom")
CHL_SRC   = Path("/home/jupyter-daniela/suyana/sources/chl_peru")
SST_SRC   = Path("/home/jupyter-daniela/suyana/sources/sst_peru")

# --- processed data ---
FEATURES  = Path("/home/jupyter-daniela/suyana/peru_production/features")
INPUTS    = Path("/home/jupyter-daniela/suyana/peru_production/inputs")
OUTPUTS   = Path("/home/jupyter-daniela/suyana/peru_production/outputs")

# --- plots ---
PLOTS     = Path("/home/jupyter-daniela/peru_catch_modeling/outputs")

FEATURES.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

YEARS     = list(range(2015, 2026))
SST_YEARS = list(range(2002, 2027))   # full MODIS AQUA record (2002 onwards)
