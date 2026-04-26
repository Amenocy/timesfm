"""
Visualize EUR/USD session forecast with covariates.

Generates a 3-panel visualization:
  1. EUR/USD context + forecast with quantile bands
  2. USD/JPY covariate
  3. GBP/USD covariate
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def visualize_forecast(json_path: str, output_path: str = None):
    """Create a 3-panel visualization from a forecast JSON file."""
    with open(json_path) as f:
        data = json.load(f)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"EUR/USD Session Forecast — {data['date']}", fontsize=14, fontweight="bold")

    # ── Panel 1: EUR/USD ──────────────────────────────────────────────
    ax1 = axes[0]
    ctx_vals = np.array(data["context_values"])
    fc_vals = np.array(data["forecast_point"])
    q_vals = np.array(data["forecast_quantiles"])

    ctx_len = len(ctx_vals)
    x_ctx = range(ctx_len)
    x_fc = range(ctx_len, ctx_len + len(fc_vals))

    ax1.plot(x_ctx, ctx_vals, label="Context (Asia + EU)", color="tab:blue", linewidth=1.5)
    ax1.plot(x_fc, fc_vals, label="Forecast (US)", color="tab:orange", linewidth=1.5)

    # Quantile bands
    if q_vals.ndim == 2:
        ax1.fill_between(x_fc, q_vals[:, 1], q_vals[:, 9],
                         alpha=0.2, color="tab:orange", label="80% PI")
        ax1.fill_between(x_fc, q_vals[:, 2], q_vals[:, 8],
                         alpha=0.3, color="tab:orange", label="60% PI")

    # Actual values if available
    if data.get("actual_values"):
        actual = np.array(data["actual_values"])
        ax1.plot(x_fc, actual, label="Actual", color="tab:green",
                 linewidth=1.5, linestyle="--")
        mae = np.mean(np.abs(actual - fc_vals))
        ax1.set_title(f"EUR/USD — MAE: {mae:.5f}")
    else:
        ax1.set_title("EUR/USD")

    ax1.set_ylabel("Price")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: USD/JPY ──────────────────────────────────────────────
    ax2 = axes[1]
    if data.get("covariate_values") and data["covariate_values"].get("usdjpy"):
        usdjpy = np.array(data["covariate_values"]["usdjpy"])
        ctx_len_cov = data.get("covariates_used", False) and "context_len" in data.get("covariate_values", {})
        
        ax2.plot(x_ctx, usdjpy[:ctx_len], label="USD/JPY (Context)", color="tab:purple", linewidth=1.2)
        ax2.plot(x_fc, usdjpy[ctx_len:], label="USD/JPY (Forecast)", color="tab:purple",
                 linewidth=1.2, linestyle="--")
        ax2.set_ylabel("USD/JPY")
        ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: GBP/USD ──────────────────────────────────────────────
    ax3 = axes[2]
    if data.get("covariate_values") and data["covariate_values"].get("gbpusd"):
        gbpusd = np.array(data["covariate_values"]["gbpusd"])
        ax3.plot(x_ctx, gbpusd[:ctx_len], label="GBP/USD (Context)", color="tab:red", linewidth=1.2)
        ax3.plot(x_fc, gbpusd[ctx_len:], label="GBP/USD (Forecast)", color="tab:red",
                 linewidth=1.2, linestyle="--")
        ax3.set_ylabel("GBP/USD")
        ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── Session markers ───────────────────────────────────────────────
    for ax in axes:
        ax.axvline(x=28, color="gray", linestyle=":", alpha=0.5)  # Asia → EU
        ax.axvline(x=64, color="gray", linestyle=":", alpha=0.5)  # EU → US
        ax.text(14, ax.get_ylim()[1], "Asia", ha="center", fontsize=8, alpha=0.6)
        ax.text(46, ax.get_ylim()[1], "EU", ha="center", fontsize=8, alpha=0.6)
        ax.text(74, ax.get_ylim()[1], "US", ha="center", fontsize=8, alpha=0.6)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"💾 Saved visualization to {output_path}")
    else:
        out = Path(json_path).parent / "forecast_visualization.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"💾 Saved visualization to {out}")

    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_session.py <forecast.json> [output.png]")
        sys.exit(1)

    json_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    visualize_forecast(json_path, output_path)
