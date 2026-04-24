"""
Step 10 - Nino 1+2 Loading and Time Series Plot with Catch

Inputs:
  OUTPUTS/calas_enriched.csv   (from step 05)
  NOAA CPC sstoi.indices URL   (falls back to embedded values)

Outputs:
  PLOTS/step09_nino12_catch.png

Skip logic: skipped if PLOTS/step09_nino12_catch.png already exists.
"""
import warnings
warnings.filterwarnings("ignore")

import io
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import OUTPUTS, PLOTS

TAB = plt.cm.tab10.colors

# ---------------------------------------------------------------------------
# Nino 1+2 fallback values (SST anomaly, region 0-10S 80-90W).
# Used only if the NOAA download fails.
# ---------------------------------------------------------------------------
NINO12_EMBEDDED = [
    ("2010-01",-0.5),("2010-02",-0.9),("2010-03",-1.4),("2010-04",-1.5),
    ("2010-05",-1.1),("2010-06",-1.2),("2010-07",-1.6),("2010-08",-2.0),
    ("2010-09",-2.0),("2010-10",-1.6),("2010-11",-1.3),("2010-12",-1.2),
    ("2011-01",-1.3),("2011-02",-1.2),("2011-03",-0.9),("2011-04",-0.6),
    ("2011-05",-0.3),("2011-06",-0.4),("2011-07",-0.8),("2011-08",-0.9),
    ("2011-09",-0.9),("2011-10",-1.0),("2011-11",-1.1),("2011-12",-1.0),
    ("2012-01",-0.8),("2012-02",-0.5),("2012-03",-0.4),("2012-04", 0.1),
    ("2012-05", 0.4),("2012-06", 0.3),("2012-07", 0.1),("2012-08", 0.0),
    ("2012-09",-0.1),("2012-10",-0.3),("2012-11",-0.5),("2012-12",-0.5),
    ("2013-01",-0.3),("2013-02",-0.2),("2013-03", 0.1),("2013-04", 0.1),
    ("2013-05",-0.1),("2013-06",-0.3),("2013-07",-0.4),("2013-08",-0.3),
    ("2013-09",-0.1),("2013-10",-0.1),("2013-11",-0.2),("2013-12",-0.3),
    ("2014-01",-0.2),("2014-02", 0.1),("2014-03", 0.5),("2014-04", 0.7),
    ("2014-05", 0.7),("2014-06", 0.4),("2014-07", 0.1),("2014-08", 0.1),
    ("2014-09", 0.3),("2014-10", 0.5),("2014-11", 0.6),("2014-12", 0.7),
    ("2015-01", 0.8),("2015-02", 0.9),("2015-03", 1.2),("2015-04", 1.6),
    ("2015-05", 2.0),("2015-06", 2.3),("2015-07", 2.4),("2015-08", 2.7),
    ("2015-09", 3.1),("2015-10", 3.5),("2015-11", 3.8),("2015-12", 4.0),
    ("2016-01", 3.9),("2016-02", 3.2),("2016-03", 2.2),("2016-04", 0.8),
    ("2016-05",-0.1),("2016-06",-0.8),("2016-07",-1.1),("2016-08",-1.1),
    ("2016-09",-0.9),("2016-10",-0.8),("2016-11",-0.8),("2016-12",-0.7),
    ("2017-01",-0.5),("2017-02",-0.2),("2017-03", 0.2),("2017-04", 0.4),
    ("2017-05", 0.5),("2017-06", 0.2),("2017-07",-0.1),("2017-08",-0.3),
    ("2017-09",-0.6),("2017-10",-0.8),("2017-11",-0.9),("2017-12",-0.9),
    ("2018-01",-0.7),("2018-02",-0.4),("2018-03",-0.2),("2018-04", 0.1),
    ("2018-05", 0.5),("2018-06", 0.8),("2018-07", 0.9),("2018-08", 1.0),
    ("2018-09", 1.1),("2018-10", 1.0),("2018-11", 0.8),("2018-12", 0.6),
    ("2019-01", 0.7),("2019-02", 0.8),("2019-03", 0.9),("2019-04", 0.8),
    ("2019-05", 0.6),("2019-06", 0.3),("2019-07", 0.1),("2019-08",-0.1),
    ("2019-09",-0.3),("2019-10",-0.5),("2019-11",-0.6),("2019-12",-0.5),
    ("2020-01", 0.4),("2020-02", 0.5),("2020-03", 0.2),("2020-04",-0.1),
    ("2020-05",-0.4),("2020-06",-0.8),("2020-07",-1.1),("2020-08",-1.3),
    ("2020-09",-1.4),("2020-10",-1.4),("2020-11",-1.3),("2020-12",-1.2),
    ("2021-01",-1.1),("2021-02",-0.9),("2021-03",-0.7),("2021-04",-0.5),
    ("2021-05",-0.3),("2021-06",-0.1),("2021-07",-0.2),("2021-08",-0.4),
    ("2021-09",-0.6),("2021-10",-0.9),("2021-11",-1.1),("2021-12",-1.2),
    ("2022-01",-1.2),("2022-02",-1.3),("2022-03",-1.1),("2022-04",-0.9),
    ("2022-05",-0.8),("2022-06",-0.8),("2022-07",-0.9),("2022-08",-1.0),
    ("2022-09",-1.0),("2022-10",-1.0),("2022-11",-0.9),("2022-12",-0.7),
    ("2023-01",-0.4),("2023-02", 0.1),("2023-03", 0.7),("2023-04", 1.3),
    ("2023-05", 1.8),("2023-06", 2.1),("2023-07", 2.5),("2023-08", 2.8),
    ("2023-09", 2.9),("2023-10", 2.8),("2023-11", 2.5),("2023-12", 2.1),
    ("2024-01", 1.7),("2024-02", 1.2),("2024-03", 0.6),("2024-04", 0.1),
    ("2024-05",-0.3),("2024-06",-0.7),("2024-07",-1.0),("2024-08",-1.1),
    ("2024-09",-1.1),("2024-10",-1.0),("2024-11",-0.9),("2024-12",-0.8),
]


def load_nino12():
    """Load monthly Nino 1+2 SST anomaly (region 0-10S, 80-90W) from NOAA CPC.
    Falls back to embedded values if the download fails.
    ENSO phase threshold: +-1.0 C (Nino 1+2 is more variable than Nino 3.4).
    """
    NINO12_URL = "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices"
    try:
        with urllib.request.urlopen(NINO12_URL, timeout=15) as r:
            raw = r.read().decode()
        records = []
        for line in raw.strip().split("\n"):
            parts = line.split()
            if len(parts) < 4 or not parts[0].isdigit():
                continue
            yr, mo = int(parts[0]), int(parts[1])
            anom = float(parts[3])   # column 3 = Nino 1+2 anomaly
            records.append({"date": pd.Timestamp(year=yr, month=mo, day=1),
                             "nino12_anom": anom})
        nino12_df = pd.DataFrame(records).dropna()
        print(f"Nino 1+2 loaded from NOAA: {len(nino12_df)} months")
    except Exception as e:
        print(f"NOAA download failed ({e}). Using embedded values.")
        nino12_df = pd.DataFrame(NINO12_EMBEDDED, columns=["ym", "nino12_anom"])
        nino12_df["date"] = pd.to_datetime(nino12_df["ym"])
        nino12_df = nino12_df[["date", "nino12_anom"]]

    THRESH = 1.0   # Nino 1+2 is more variable; use 1.0 C threshold
    nino12_df = nino12_df.sort_values("date").reset_index(drop=True)
    raw_phase = pd.Series("neutral", index=nino12_df.index)
    raw_phase[nino12_df["nino12_anom"] >= THRESH]  = "nino"
    raw_phase[nino12_df["nino12_anom"] <= -THRESH] = "nina"

    enso_phase = raw_phase.copy()
    for i in range(4, len(raw_phase)):
        window = raw_phase.iloc[i - 4:i + 1]
        if (window == "nino").all():
            enso_phase.iloc[i - 4:i + 1] = "nino"
        elif (window == "nina").all():
            enso_phase.iloc[i - 4:i + 1] = "nina"
        else:
            if enso_phase.iloc[i] not in ("nino", "nina"):
                enso_phase.iloc[i] = "neutral"

    nino12_df["enso_phase"] = enso_phase
    return nino12_df


def load_calas():
    df = pd.read_csv(OUTPUTS / "calas_enriched.csv")
    rename_map = {
        "fecha_cala": "date", "fecha": "date",
        "temporada": "season", "declarado_tm": "catch_tm",
        "latitud": "lat", "longitud": "lon",
        "modis_sst_anomaly": "modis_sst_anom",
        "chlor_a": "chl",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df["date"] = pd.to_datetime(df["date"])
    return df


def main():
    if (PLOTS / "step09_nino12_catch.png").exists():
        print("step09_nino12_catch.png exists -- skipping")
        return

    df = load_calas()
    daily_all = (
        df[df["catch_tm"] > 0]
        .groupby("date")
        .agg(total_catch_tm=("catch_tm", "sum"), season=("season", "first"))
        .reset_index()
    )
    print(f"Full catch panel: {len(daily_all):,} days | "
          f"{daily_all['date'].min().date()} - {daily_all['date'].max().date()}")

    nino12_df = load_nino12()

    fig, ax1 = plt.subplots(figsize=(16, 4))
    ax2 = ax1.twinx()
    nino12_plot = nino12_df[nino12_df["date"] >= "1980-01-01"].copy()
    for phase, col in [("nino", TAB[1]), ("nina", TAB[9])]:
        sub = nino12_plot[nino12_plot["enso_phase"] == phase]
        for _, row in sub.iterrows():
            ax1.axvspan(row["date"], row["date"] + pd.offsets.MonthEnd(1),
                        alpha=0.15, color=col, zorder=0)
    ax2.plot(nino12_plot["date"], nino12_plot["nino12_anom"], color="dimgrey", lw=1.2)
    ax2.axhline( 1.0, color=TAB[1], lw=0.7, ls="--", alpha=0.8)
    ax2.axhline(-1.0, color=TAB[9], lw=0.7, ls="--", alpha=0.8)
    ax1.fill_between(daily_all["date"], daily_all["total_catch_tm"] / 1e3,
                     alpha=0.55, color=TAB[0])
    ax1.set_xlabel("Date"); ax1.set_ylabel("Catch (kt)", color=TAB[0])
    ax2.set_ylabel("Nino 1+2 anomaly (C)", color="dimgrey")
    ax1.set_title("Nino 1+2 (1980-present) with ENSO phase shading and anchovy catch", loc="left")
    handles = [mpatches.Patch(color=TAB[1], alpha=0.5, label="El Nino (>=+1.0 C)"),
               mpatches.Patch(color=TAB[9], alpha=0.5, label="La Nina (<=-1.0 C)"),
               mpatches.Patch(color=TAB[0], alpha=0.6, label="Catch (kt)")]
    ax1.legend(handles=handles, loc="upper left", fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(PLOTS / "step09_nino12_catch.png", dpi=120)
    plt.close()

    print(f"step09_nino12_catch.png saved to {PLOTS}")


if __name__ == "__main__":
    main()
