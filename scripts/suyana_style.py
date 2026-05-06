"""
Suyana plotting style — colors, formatters, rcParams.
Import and call apply_style() once at the top of each plot script.
"""
import matplotlib.pyplot as plt

SGREEN      = "#43A047"
SDARKGREEN  = "#2E7D32"
SBLACK      = "#141414"
SGRAY       = "#555555"
SMIDGRAY    = "#9E9E9E"
SLIGHTGRAY  = "#E0E0E0"
SRED        = "#d62728"
SORANGE     = "#E6550D"
SBLUE       = "#2171B5"

COLORMAP = "Greens"


def apply_style() -> None:
    plt.rcParams.update({
        "axes.prop_cycle": plt.cycler(
            color=[SGREEN, SDARKGREEN, SGRAY, SBLUE, SORANGE, SRED, SMIDGRAY]
        ),
        "axes.edgecolor":   SBLACK,
        "axes.labelcolor":  SBLACK,
        "xtick.color":      SBLACK,
        "ytick.color":      SBLACK,
        "axes.grid":        True,
        "grid.color":       SLIGHTGRAY,
        "grid.alpha":       0.35,
        "grid.linewidth":   0.6,
        "axes.axisbelow":   True,
        "axes.spines.top":  False,
        "axes.spines.right": False,
        "font.size":        10,
        "legend.frameon":   False,
        "legend.fontsize":  9,
        "savefig.dpi":      170,
        "savefig.bbox":     "tight",
    })


def pct_formatter(decimals: int = 0, already_percent: bool = False):
    scale = 1.0 if already_percent else 100.0
    return plt.FuncFormatter(lambda v, _: f"{v * scale:.{decimals}f}%")


def dollar_formatter(unit: str = "auto", decimals=None):
    if unit == "raw":
        d = 0 if decimals is None else decimals
        return plt.FuncFormatter(lambda v, _: f"${v:,.{d}f}")
    if unit == "K":
        d = 0 if decimals is None else decimals
        return plt.FuncFormatter(lambda v, _: f"${v / 1e3:,.{d}f}K")
    if unit == "M":
        d = 1 if decimals is None else decimals
        return plt.FuncFormatter(lambda v, _: f"${v / 1e6:,.{d}f}M")
    def fmt(v, _):
        av = abs(v)
        if av >= 1e6:
            return f"${v / 1e6:,.1f}M"
        if av >= 1e3:
            return f"${v / 1e3:,.0f}K"
        return f"${v:,.0f}"
    return plt.FuncFormatter(fmt)


def physical_formatter(unit: str = "", decimals: int = 1, signed: bool = False):
    sign = "+" if signed else ""
    return plt.FuncFormatter(lambda v, _: f"{v:{sign}.{decimals}f}{unit}")


def year_formatter():
    return plt.FuncFormatter(lambda v, _: f"{int(v)}")


def count_formatter():
    return plt.FuncFormatter(lambda v, _: f"{int(v):,}")
