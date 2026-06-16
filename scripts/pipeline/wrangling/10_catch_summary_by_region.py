"""
Step 00 - Catch Summary: serie temporal de captura por temporada y region

Carga todos los archivos de Calas de anchoveta (2015-2025) y calcula:
  - Captura total por temporada (1ra / 2da) para todo el mar y por region:
      Norte        (lat > -7.1)
      Centro Norte (-11.0 < lat <= -7.1)
      Centro Sur   (-15.8 < lat <= -11.0)
      Centro       (-15.8 < lat <= -7.1)

Usa declarado_tm de los archivos crudos de Calas, independiente del
paso de enriquecimiento, para incluir siempre el anio mas reciente.

Inputs:
  INPUTS/ihma_data/{year}/*anchoveta*Calas*.csv

Outputs:
  OUTPUTS/catch_summary_by_region.csv
  OUTPUTS/catch_summary_by_region.md
  PLOTS/08_wrangling_catch_timeseries.png
"""
import warnings
warnings.filterwarnings("ignore")

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from suyana_style import apply_style, SGREEN, SBLUE, SGRAY, SMIDGRAY, SBLACK, SORANGE, SRED

from config import INPUTS, OUTPUTS, PLOTS

REGIONS = [
    ("Todo el mar",  None,   None),
    ("Norte",        -7.1,   None),
    ("Centro Norte", -11.0,  -7.1),
    ("Centro Sur",   -15.8,  -11.0),
    ("Centro",       -15.8,  -7.1),
]

REG_COLORS = {
    "Todo el mar":  SBLACK,
    "Norte":        SBLUE,
    "Centro Norte": SGREEN,
    "Centro Sur":   SORANGE,
    "Centro":       SGRAY,
}

T1_COLOR = SGREEN
T2_COLOR = SBLUE


def load_all_calas():
    base = INPUTS / "ihma_data"
    dfs = []
    for f in sorted(base.rglob("*.csv")):
        if not re.search(r"Calas", f.name, re.IGNORECASE):
            continue
        if not re.search(r"anchoveta", f.name, re.IGNORECASE):
            continue
        nombre = f.name.lower()
        if "primera" in nombre:
            temporada = "1ra"
        elif "segunda" in nombre:
            temporada = "2da"
        else:
            continue
        for enc in ("utf-8-sig", "latin1"):
            try:
                df = pd.read_csv(f, low_memory=False, encoding=enc)
                df.columns = df.columns.str.strip()
                df = df.loc[:, ~df.columns.duplicated()]
                if "declarado_tm" not in df.columns:
                    break
                df["declarado_tm"] = pd.to_numeric(df["declarado_tm"], errors="coerce")
                df["latitud"]  = pd.to_numeric(df.get("latitud",  pd.Series(dtype=float)), errors="coerce")
                df["longitud"] = pd.to_numeric(df.get("longitud", pd.Series(dtype=float)), errors="coerce")
                df["anio"]      = int(f.parts[-2])
                df["temporada"] = temporada
                df["temporada_key"] = f"{temporada} {f.parts[-2]}"
                dfs.append(df[["anio", "temporada", "temporada_key", "declarado_tm", "latitud", "longitud"]])
                break
            except UnicodeDecodeError:
                continue
    if not dfs:
        raise RuntimeError("No se encontraron archivos de calas de anchoveta.")
    return pd.concat(dfs, ignore_index=True)


def region_mask(df, lat_min, lat_max):
    mask = pd.Series(True, index=df.index)
    if lat_min is not None:
        mask &= df["latitud"] > lat_min
    if lat_max is not None:
        mask &= df["latitud"] <= lat_max
    return mask


def build_series(df):
    """Devuelve dict {region: DataFrame con columnas temporada_key, anio, temporada, captura_tm}"""
    series = {}
    for reg_name, lat_min, lat_max in REGIONS:
        sub = df[region_mask(df, lat_min, lat_max)]
        agg = (sub.groupby(["temporada_key", "anio", "temporada"])["declarado_tm"]
               .sum().reset_index(name="captura_tm"))
        agg = agg.sort_values(["anio", "temporada"])
        series[reg_name] = agg
    return series


def build_summary_table(series):
    """Tabla de medias por region, separando T1 y T2."""
    rows = []
    for reg_name, agg in series.items():
        t1 = agg[agg["temporada"] == "1ra"]["captura_tm"]
        t2 = agg[agg["temporada"] == "2da"]["captura_tm"]
        anual = agg.groupby("anio")["captura_tm"].sum()
        rows.append({
            "Region":         reg_name,
            "Media 1ra (tm)": int(t1.mean()) if len(t1) else 0,
            "Media 2da (tm)": int(t2.mean()) if len(t2) else 0,
            "Media anual (tm)": int(anual.mean()),
            "Min anual (tm)": int(anual.min()),
            "Anio min":       int(anual.idxmin()),
            "Max anual (tm)": int(anual.max()),
            "Anio max":       int(anual.idxmax()),
        })
    return pd.DataFrame(rows)


def save_markdown(summary_df, series, out_path):
    lines = ["# Resumen de captura de anchoveta por region (2015-2025)", ""]
    lines += ["## Medias de captura", ""]

    # tabla de medias
    cols = list(summary_df.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in summary_df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, (int, np.integer)):
                vals.append(f"{v:,}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    lines += ["", "## Serie temporal por temporada y region", ""]
    cols2 = ["Temporada", "Todo el mar", "Norte", "Centro Norte", "Centro Sur", "Centro"]
    lines.append("| " + " | ".join(cols2) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols2)) + " |")

    keys = sorted(series["Todo el mar"]["temporada_key"].unique(),
                  key=lambda s: (int(s.split()[1]), s.split()[0]))
    for tk in keys:
        row_vals = [tk]
        for reg_name in ["Todo el mar", "Norte", "Centro Norte", "Centro Sur", "Centro"]:
            sub = series[reg_name]
            match = sub[sub["temporada_key"] == tk]["captura_tm"]
            row_vals.append(f"{int(match.values[0]):,}" if len(match) else "-")
        lines.append("| " + " | ".join(row_vals) + " |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Guardado -> {out_path}")


def plot_timeseries(series):
    apply_style()
    fig, axes = plt.subplots(len(REGIONS), 1, figsize=(13, 3.2 * len(REGIONS)),
                             sharex=True)

    all_keys = sorted(
        series["Todo el mar"]["temporada_key"].unique(),
        key=lambda s: (int(s.split()[1]), s.split()[0])
    )
    x = np.arange(len(all_keys))

    for ax, (reg_name, *_) in zip(axes, REGIONS):
        agg = series[reg_name]
        color = REG_COLORS[reg_name]

        vals = []
        colors_bar = []
        for tk in all_keys:
            match = agg[agg["temporada_key"] == tk]["captura_tm"]
            v = match.values[0] / 1e3 if len(match) else 0.0
            vals.append(v)
            colors_bar.append(T1_COLOR if "1ra" in tk else T2_COLOR)

        bars = ax.bar(x, vals, color=colors_bar, alpha=0.80, width=0.75, zorder=3)

        media = np.mean([v for v, tk in zip(vals, all_keys) if v > 0])
        ax.axhline(media, color=color, lw=1.4, ls="--", zorder=4,
                   label=f"Media {media:,.0f} kt")

        ax.set_title(reg_name, loc="left", fontsize=11, fontweight="bold", color=SBLACK)
        ax.set_ylabel("Captura (kt)", fontsize=9, color=SGRAY)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.legend(fontsize=8)

        for bar, tk in zip(bars, all_keys):
            bar.set_edgecolor("none")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(all_keys, rotation=45, ha="right", fontsize=8)

    # leyenda global de colores de barra
    from matplotlib.patches import Patch
    fig.legend(
        handles=[Patch(color=T1_COLOR, alpha=0.8, label="1ra temporada"),
                 Patch(color=T2_COLOR, alpha=0.8, label="2da temporada")],
        loc="upper right", fontsize=9, frameon=False,
        bbox_to_anchor=(0.99, 0.99)
    )

    fig.suptitle("Captura de anchoveta por temporada y region -- Peru 2015-2025",
                 fontsize=12, fontweight="bold", x=0.01, ha="left", color=SBLACK)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = PLOTS / "08_wrangling_catch_timeseries.png"
    plt.savefig(out)
    plt.close()
    print(f"Guardado -> {out}")


def main():
    print("Cargando calas de anchoveta...")
    df = load_all_calas()
    df = df[df["declarado_tm"] > 0]
    print(f"  {len(df):,} calas validas | anios: {sorted(df['anio'].unique())}\n")

    series = build_series(df)
    summary_df = build_summary_table(series)

    # CSV
    rows = []
    for reg_name, agg in series.items():
        for _, r in agg.iterrows():
            rows.append({"region": reg_name, "temporada_key": r["temporada_key"],
                         "anio": r["anio"], "temporada": r["temporada"],
                         "captura_tm": r["captura_tm"]})
    pd.DataFrame(rows).to_csv(OUTPUTS / "catch_summary_by_region.csv", index=False)
    print(f"Guardado -> {OUTPUTS / 'catch_summary_by_region.csv'}")

    # Markdown
    save_markdown(summary_df, series, OUTPUTS / "catch_summary_by_region.md")

    # Print resumen
    print("\n--- RESUMEN MEDIAS ---")
    for _, row in summary_df.iterrows():
        print(f"  {row['Region']:<20}: {row['Media anual (tm)']:>12,} tm/anio")

    # Plot
    plot_timeseries(series)


if __name__ == "__main__":
    main()
