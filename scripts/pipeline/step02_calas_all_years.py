"""
Step 02 - Calas All Years

Loads raw anchoveta fishing records (calas) from IHMA CSV files for all
available years and concatenates them into a single flat table.

Inputs:
  INPUTS/ihma_data/{year}/*Calas*anchoveta*{Primera|Segunda}temporada*.csv

Outputs:
  OUTPUTS/calas_all_data.csv

Skip logic: skipped entirely if output already exists.
"""
import re
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pathlib import Path

from config import INPUTS, OUTPUTS, YEARS


def read_csv_auto(path):
    for enc in ("utf-8-sig", "latin1"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {path}")


def load_year(base_path, anio):
    carpeta = Path(base_path) / "ihma_data" / str(anio)
    archivos = [
        a for a in carpeta.glob("*.csv")
        if re.search(r"Calas", a.name, re.IGNORECASE)
        and (
            re.search(r"Primera temporada", a.name, re.IGNORECASE)
            or re.search(r"Segunda temporada", a.name, re.IGNORECASE)
        )
        and re.search(r"anchoveta", a.name, re.IGNORECASE)
    ]
    if not archivos:
        raise FileNotFoundError(f"No files found in {carpeta}")

    print(f"  {anio}: {[a.name for a in archivos]}")
    df = pd.concat([read_csv_auto(a) for a in archivos], ignore_index=True)
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated()]
    df["fecha_cala"] = pd.to_datetime(df["fecha_cala"], format="%d/%m/%Y")
    df["declarado_tm"] = pd.to_numeric(df["declarado_tm"], errors="coerce")
    if "porcentaje_juvenil" in df.columns:
        df["porcentaje_juvenil"] = pd.to_numeric(df["porcentaje_juvenil"], errors="coerce")
    df["semana"] = df["fecha_cala"].dt.isocalendar().week
    print(f"    -> {len(df):,} rows")
    return df


def main():
    out = OUTPUTS / "calas_all_data.csv"
    if out.exists():
        print(f"Output already exists: {out} -- skipping")
        return

    dfs = []
    for anio in YEARS:
        try:
            dfs.append(load_year(INPUTS, anio))
        except FileNotFoundError:
            print(f"  Skipping {anio}: no files found")

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows: {len(df_all):,}")
    df_all.to_csv(out, index=False)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
