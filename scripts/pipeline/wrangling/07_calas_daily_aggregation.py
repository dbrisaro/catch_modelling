"""
Step 03 - Calas Daily Aggregation

Aggregates individual fishing events to daily totals per vessel/company/season,
keeping mean lat/lon coordinates.

Inputs:
  OUTPUTS/calas_all_data.csv   (from step 02)

Outputs:
  OUTPUTS/calas_daily_with_coordinates.csv

Skip logic: skipped entirely if output already exists.
"""
import pandas as pd
from config import OUTPUTS


def main():
    out = OUTPUTS / "calas_daily_with_coordinates.csv"
    if out.exists():
        print(f"Output already exists: {out} -- skipping")
        return

    df = pd.read_csv(OUTPUTS / "calas_all_data.csv", low_memory=False)
    df["fecha_cala"] = pd.to_datetime(df["fecha_cala"], errors="coerce")
    df = df.rename(columns={"latitud ": "latitud"})

    df_daily = (
        df.groupby(["fecha_cala", "temporada", "empresa", "nave"])
        .agg(
            declarado_tm=("declarado_tm", "sum"),
            latitud=("latitud", "mean"),
            longitud=("longitud", "mean"),
        )
        .reset_index()
        .rename(columns={
            "fecha_cala":   "date",
            "temporada":    "season",
            "empresa":      "company",
            "nave":         "vessel",
            "declarado_tm": "catch_tm",
            "latitud":      "lat",
            "longitud":     "lon",
        })
    )

    df_daily.to_csv(out, index=False)
    print(f"Saved {len(df_daily):,} rows -> {out}")


if __name__ == "__main__":
    main()
