"""
Step 16c - SSS payout curves (Part B analog for SSS)

Analogo a la Part B de 16_pricing_trigger_design.py pero con SSS anomaly
como predictor. El trigger SSS es INVERTIDO: SSS bajo (agua dulce, El Nino)
genera el pago, no SSS alto.

Cuantiles estacionales SSS (climatologia 2015-2024, de analisis 15c):
  p10 = -0.109 PSU  (entrada ramp, trigger step)
  p05 = -0.151 PSU
  p01 = -0.176 PSU  (salida ramp, pago maximo)

Curvas de pago:
  - OLS-implied (exponencial): max(0, 1 - exp(beta_sss * sss)) normalizada
  - Ramp lineal: 0 por encima de entry (p10), lineal hasta 1 en exit (p01)
  - Step (binario): 0 por encima de trigger (p10), 1 en o por debajo

Beta de referencia SSS -> captura (Norte, positivo): BETA_SSS = +0.35
(valor aproximado del analisis 14b/16b region Norte)

Inputs:
  (ninguno -- curvas calculadas analitica/parametricamente)

Outputs:
  PLOTS/16c_pricing_sss_payout_curves.png

Skip logic: skipped si el output ya existe.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

from config import PLOTS

# ── payout parameters ─────────────────────────────────────────────────────────
# Calibrados a cuantiles estacionales SSS (climatologia 2015-2024, analisis 15c)
# Trigger es INVERTIDO: pago cuando SSS < umbral (agua mas dulce = El Nino)
HIST_QUANTILES = {
    "p10": -0.109,   # entrada ramp / trigger step
    "p05": -0.151,
    "p01": -0.176,   # salida ramp / pago maximo
}

ENTRY_SSS    = HIST_QUANTILES["p10"]   # ramp entry = p10
EXIT_SSS     = HIST_QUANTILES["p01"]   # ramp exit  = p01
STEP_TRIGGER = HIST_QUANTILES["p10"]   # step trigger = p10

# Beta OLS SSS -> log(captura): positivo (mas salinidad = mas captura).
# Placeholder +0.35 aproximado de analisis 14b/16b region Norte.
BETA_SSS = +0.35


# ── payout functions (inverted vs SST) ────────────────────────────────────────
def payout_ols_sss(sss, beta=BETA_SSS, exit_=EXIT_SSS):
    """OLS-implied fractional loss: max(0, 1 - exp(beta * sss)) para sss < 0.

    Para SSS positivo (agua salada normal) no hay pago.
    La curva crece a medida que sss se vuelve mas negativo (mas fresco).
    Normalizada al valor en exit_ (p01) para que el maximo sea 1.
    """
    raw = np.where(sss < 0, np.maximum(0.0, 1 - np.exp(beta * sss)), 0.0)
    return raw


def payout_linear(sss, entry=ENTRY_SSS, exit_=EXIT_SSS):
    """Ramp lineal invertida.

    entry = -0.109 (p10): pago comienza cuando sss cae por debajo de este valor.
    exit_ = -0.176 (p01): pago maximo cuando sss cae hasta aqui o mas.
    frac va de 0 en entry a 1 en exit_ (sss mas negativo -> mas pago).
    """
    return np.clip((entry - sss) / (entry - exit_), 0.0, 1.0)


def payout_step(sss, trigger=STEP_TRIGGER):
    """Step binario: 1 cuando sss <= trigger, 0 en caso contrario."""
    return np.where(sss <= trigger, 1.0, 0.0)


# ── plot ──────────────────────────────────────────────────────────────────────
def plot_payout_curves(outpath):
    # Rango de anomalia SSS: +0.2 a -0.25 PSU
    # El eje X va de izquierda (positivo = normal) a derecha (negativo = El Nino)
    sss_range = np.linspace(0.20, -0.25, 500)

    # Normalizar curva OLS al valor en EXIT_SSS (p01)
    max_ref = payout_ols_sss(np.array([EXIT_SSS]))[0]
    if max_ref <= 0:
        max_ref = 1.0   # fallback para evitar division por cero

    y_ols    = payout_ols_sss(sss_range) / max_ref
    y_linear = payout_linear(sss_range)
    y_step   = payout_step(sss_range)

    # Colores para cuantiles (tonos de azul: baja salinidad = El Nino = azul)
    q_colors = {
        "p10": "#9ecae1",   # azul claro  (entrada / trigger)
        "p05": "#3182bd",   # azul medio
        "p01": "#08519c",   # azul oscuro (salida / pago maximo)
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, (title, y_curve, col, lbl_curve) in zip(axes, [
        ("Ramp lineal", y_linear, "#2166ac", "Ramp lineal"),
        ("Step (binario)", y_step, "#4393c3", "Step (disparador unico)"),
    ]):
        # Curva OLS de referencia
        ax.plot(sss_range, y_ols, color="#888888", lw=1.8, ls="--",
                label=f"OLS (exponencial, beta={BETA_SSS:+.2f})", zorder=2)
        # Curva principal
        ax.plot(sss_range, y_curve, color=col, lw=2.5, label=lbl_curve, zorder=3)
        # Relleno bajo la curva
        ax.fill_between(sss_range, 0, y_curve, color=col, alpha=0.12)

        # Lineas de referencia de cuantiles historicos
        for qname, qval in HIST_QUANTILES.items():
            ax.axvline(qval, color=q_colors[qname], lw=1.2, ls=":",
                       label=f"{qname} = {qval} PSU")

        ax.axhline(1.0, color="black", lw=0.7, ls="--", alpha=0.4)
        ax.axvline(0,   color="grey",  lw=0.7, ls=":",  alpha=0.5)

        # Eje X: positivo a la izquierda, negativo a la derecha
        ax.set_xlim(0.20, -0.25)
        ax.set_ylim(-0.05, 1.25)
        ax.set_xlabel("Anomalia SSS estacional (PSU)", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Fraccion del pago maximo", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Anotaciones de entrada/salida para ramp; trigger para step
        if "Ramp" in title:
            ax.annotate(
                f"p10 = {ENTRY_SSS} PSU\n(inicio pago)",
                xy=(ENTRY_SSS, 0.0),
                xytext=(ENTRY_SSS + 0.08, 0.18),
                fontsize=7.5, color="#2166ac",
                arrowprops=dict(arrowstyle="->", color="#2166ac", lw=0.8),
            )
            ax.annotate(
                f"p01 = {EXIT_SSS} PSU\n(pago maximo)",
                xy=(EXIT_SSS, 1.0),
                xytext=(EXIT_SSS + 0.06, 1.10),
                fontsize=7.5, color="#2166ac",
                arrowprops=dict(arrowstyle="->", color="#2166ac", lw=0.8),
            )
        else:
            ax.annotate(
                f"Trigger\n(p10 = {STEP_TRIGGER} PSU)",
                xy=(STEP_TRIGGER, 0.5),
                xytext=(STEP_TRIGGER + 0.07, 0.60),
                fontsize=7.5, color="#4393c3",
                arrowprops=dict(arrowstyle="->", color="#4393c3", lw=0.8),
            )

    fig.suptitle(
        "Curvas de pago SSS - trigger invertido (SSS bajo = El Nino = pago)",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {outpath}")

    # Imprimir valores clave de referencia
    print("\nPayout values at reference SSS anomalies:")
    print(f"  {'SSS':>7}  {'OLS':>8}  {'Ramp':>8}  {'Step':>8}")
    for sss in [0.10, 0.00, -0.05, -0.109, -0.130, -0.151, -0.176, -0.20, -0.25]:
        ols_raw = payout_ols_sss(np.array([sss]))[0] / max_ref
        ramp    = payout_linear(np.array([sss]))[0]
        step    = payout_step(np.array([sss]))[0]
        print(f"  {sss:>7.3f}  {ols_raw:>8.3f}  {ramp:>8.3f}  {step:>8.3f}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    out_payout = PLOTS / "16c_pricing_sss_payout_curves.png"

    if out_payout.exists():
        print("16c output exists -- skipping")
        return

    print("16c: SSS payout curves...")
    plot_payout_curves(out_payout)


if __name__ == "__main__":
    main()
