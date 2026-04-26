"""
EUR/USD Session Forecasting — Backtester

Runs the forecasting pipeline over a date range and produces a comprehensive
backtest report with:
  - Directional accuracy (% of days correct)
  - Mean of Error (MoE)
  - MAE / RMSE / MAPE
  - +pips / -pips breakdown
  - Per-day and aggregate summary tables

Usage:
    python backtest_session.py --start 2024-01-01 --end 2024-03-31
    python backtest_session.py --start 2024-01-01 --end 2024-03-31 --no-covariates
    python backtest_session.py --start 2024-01-01 --end 2024-03-31 --output results/backtest
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime

# Add parent directory to path so we can import forecast_session
sys.path.insert(0, str(Path(__file__).resolve().parent))

from forecast_session import (
    load_15m_data,
    extract_session,
    build_covariate_arrays,
    check_timesfm_installed,
    CONTEXT_CANDLES,
    FORECAST_CANDLES,
    DATA_DIR,
)

# ── Pip value for EUR/USD (0.0001 = 1 pip for most pairs) ────────────────
PIP_SIZE = 0.0001


def prices_to_pips(forecast: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Convert price differences to pips."""
    return (forecast - actual) / PIP_SIZE


def compute_daily_metrics(result: dict) -> dict:
    """
    Compute per-day metrics from a single forecast result.

    Returns dict with:
        - date
        - mae, rmse, mape, moe
        - directional_accuracy (1 if mean direction correct, else 0)
        - total_pips, positive_pips, negative_pips
        - num_candles
    """
    actual = np.array(result["actual_values"], dtype=np.float64)
    forecast = np.array(result["forecast_point"], dtype=np.float64)

    if len(actual) == 0 or len(forecast) == 0:
        return {}

    # Ensure same length
    min_len = min(len(actual), len(forecast))
    actual = actual[:min_len]
    forecast = forecast[:min_len]

    errors = forecast - actual
    abs_errors = np.abs(errors)
    pips = prices_to_pips(forecast, actual)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    moe = float(np.mean(errors))  # Mean of Error (signed)

    # MAPE (avoid division by zero)
    safe_actual = np.where(np.abs(actual) < 1e-10, 1e-10, actual)
    mape = float(np.mean(np.abs(errors / safe_actual)) * 100)

    # Directional accuracy: compare mean of context last N vs forecast vs actual
    context_vals = np.array(result["context_values"], dtype=np.float64)
    if len(context_vals) >= 4:
        ctx_mean = np.mean(context_vals[-4:])  # last hour of context
    else:
        ctx_mean = context_vals[-1]

    forecast_direction = 1 if np.mean(forecast) > ctx_mean else 0
    actual_direction = 1 if np.mean(actual) > ctx_mean else 0
    directional_correct = int(forecast_direction == actual_direction)

    # Pip metrics
    total_pips = float(np.sum(pips))
    positive_pips = float(np.sum(pips[pips > 0]))
    negative_pips = float(np.sum(pips[pips < 0]))

    return {
        "date": result["date"],
        "num_candles": min_len,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "moe": moe,
        "moe_pips": moe / PIP_SIZE,
        "directional_correct": directional_correct,
        "total_pips": total_pips,
        "positive_pips": positive_pips,
        "negative_pips": negative_pips,
        "abs_total_pips": float(np.sum(np.abs(pips))),
    }


def run_backtest(
    start_date: str,
    end_date: str,
    eur_df: pd.DataFrame,
    usdjpy_df: pd.DataFrame,
    gbpusd_df: pd.DataFrame,
    use_covariates: bool = False,
    project_covariates: bool = True,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Run backtest over a date range.

    Parameters
    ----------
    start_date, end_date : str
        Date range in YYYY-MM-DD format.
    eur_df, usdjpy_df, gbpusd_df : pd.DataFrame
        15-minute OHLCV data.
    use_covariates : bool
        Whether to use covariates.
    project_covariates : bool
        If True, project covariate means (live mode).
        If False, use actual values (backtest mode).
    output_dir : Path, optional
        Directory to save results.

    Returns
    -------
    dict with per-day results and aggregate metrics.
    """
    import torch
    import timesfm

    # ── 1. Load model once ──────────────────────────────────────────────
    torch.set_float32_matmul_precision("high")
    print("🔄 Loading TimesFM 2.5 model...")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    model.compile(timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=256,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=False,
        fix_quantile_crossing=True,
        return_backcast=use_covariates,
    ))
    print("✅ Model loaded and compiled.")

    # ── 2. Build date range ─────────────────────────────────────────────
    dates = pd.date_range(start=start_date, end=end_date, freq="B")  # business days
    print(f"\n📅 Backtest range: {start_date} → {end_date} ({len(dates)} business days)")

    daily_results = []
    skipped = []

    # ── 3. Loop over dates ──────────────────────────────────────────────
    for i, date in enumerate(dates):
        print(f"\n[{i+1}/{len(dates)}] Processing {date.strftime('%Y-%m-%d')}...")

        # Extract session
        session = extract_session(eur_df, date)
        if not session or session["context_candles"] < 32:
            print(f"  ⚠️  Insufficient context, skipping.")
            skipped.append(date.strftime("%Y-%m-%d"))
            continue

        context_values = session["context"]["close"].values.astype(np.float32)
        actual_values = session["target"]["close"].values.astype(np.float32) if len(session["target"]) > 0 else None

        if actual_values is None or len(actual_values) == 0:
            print(f"  ⚠️  No target data, skipping.")
            skipped.append(date.strftime("%Y-%m-%d"))
            continue

        # Build covariates
        # NOTE: For realistic backtesting, we project covariate means from context
        # rather than using actual future values (which would be data leakage).
        # Set project_covariates=True to simulate live trading conditions.
        covariates = {}
        if use_covariates:
            covariates = build_covariate_arrays(
                eur_df, usdjpy_df, gbpusd_df, date,
                include_future=not project_covariates,  # False = project means (no leak)
            )

        # Forecast
        try:
            if use_covariates and covariates:
                point, quantiles = model.forecast_with_covariates(
                    inputs=[context_values],
                    dynamic_numerical_covariates={
                        "usdjpy": [covariates["usdjpy_close"]],
                        "gbpusd": [covariates["gbpusd_close"]],
                    },
                    xreg_mode="xreg + timesfm",
                )
            else:
                horizon = FORECAST_CANDLES
                point, quantiles = model.forecast(
                    horizon=horizon,
                    inputs=[context_values],
                )

            # Build result dict
            result = {
                "date": session["date"],
                "context_values": context_values.tolist(),
                "forecast_point": point[0].tolist(),
                "forecast_quantiles": quantiles[0].tolist(),
                "actual_values": actual_values.tolist(),
                "covariates_used": use_covariates and len(covariates) > 0,
            }

            # Compute metrics
            metrics = compute_daily_metrics(result)
            if metrics:
                daily_results.append(metrics)
                dir_symbol = "✅" if metrics["directional_correct"] else "❌"
                print(f"  {dir_symbol} MAE={metrics['mae']:.5f} | "
                      f"MoE={metrics['moe']:.5f} | "
                      f"Pips=+{metrics['positive_pips']:.1f}/"
                      f"{metrics['negative_pips']:.1f} | "
                      f"Dir={'✓' if metrics['directional_correct'] else '✗'}")

            # Save individual result if output_dir specified
            if output_dir:
                out_path = output_dir / f"backtest_{session['date']}.json"
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)

        except Exception as e:
            print(f"  ❌ Error: {e}")
            skipped.append(date.strftime("%Y-%m-%d"))
            continue

    # ── 4. Aggregate metrics ────────────────────────────────────────────
    if not daily_results:
        print("\n❌ No valid results to aggregate.")
        return {}

    df_metrics = pd.DataFrame(daily_results)

    aggregate = {
        "start_date": start_date,
        "end_date": end_date,
        "total_days": len(dates),
        "valid_days": len(daily_results),
        "skipped_days": len(skipped),
        "skipped_dates": skipped,
        "use_covariates": use_covariates,
        "covariates_projected": project_covariates,
        # Error metrics
        "mean_mae": float(df_metrics["mae"].mean()),
        "median_mae": float(df_metrics["mae"].median()),
        "mean_rmse": float(df_metrics["rmse"].mean()),
        "mean_mape": float(df_metrics["mape"].mean()),
        "mean_moe": float(df_metrics["moe"].mean()),
        "mean_moe_pips": float(df_metrics["moe_pips"].mean()),
        # Directional accuracy
        "directional_accuracy_pct": float(df_metrics["directional_correct"].mean() * 100),
        "days_correct": int(df_metrics["directional_correct"].sum()),
        "days_incorrect": int(len(daily_results) - df_metrics["directional_correct"].sum()),
        # Pip metrics
        "total_pips": float(df_metrics["total_pips"].sum()),
        "total_positive_pips": float(df_metrics["positive_pips"].sum()),
        "total_negative_pips": float(df_metrics["negative_pips"].sum()),
        "total_abs_pips": float(df_metrics["abs_total_pips"].sum()),
        "avg_pips_per_day": float(df_metrics["total_pips"].mean()),
        "avg_positive_pips_per_day": float(df_metrics["positive_pips"].mean()),
        "avg_negative_pips_per_day": float(df_metrics["negative_pips"].mean()),
        # Best / worst days
        "best_day": df_metrics.loc[df_metrics["total_pips"].idxmax(), "date"],
        "best_day_pips": float(df_metrics["total_pips"].max()),
        "worst_day": df_metrics.loc[df_metrics["total_pips"].idxmin(), "date"],
        "worst_day_pips": float(df_metrics["total_pips"].min()),
        # Win rate (days with net positive pips)
        "win_rate_pct": float((df_metrics["total_pips"] > 0).mean() * 100),
        "winning_days": int((df_metrics["total_pips"] > 0).sum()),
        "losing_days": int((df_metrics["total_pips"] <= 0).sum()),
    }

    return {
        "aggregate": aggregate,
        "daily_metrics": daily_results,
        "daily_df": df_metrics,
    }


def print_backtest_report(report: dict, output_dir: Optional[Path] = None):
    """Print a formatted backtest report to console and optionally to file."""
    if not report or "aggregate" not in report:
        print("❌ No report data to display.")
        return

    agg = report["aggregate"]
    df_metrics = report["daily_df"]

    # ── Header ──────────────────────────────────────────────────────────
    width = 80
    print("\n" + "=" * width)
    print("  EUR/USD SESSION FORECAST — BACKTEST REPORT".center(width))
    print("=" * width)

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n📅 Period:        {agg['start_date']} → {agg['end_date']}")
    print(f"📊 Total days:    {agg['total_days']}")
    print(f"✅ Valid days:    {agg['valid_days']}")
    print(f"⚠️  Skipped days: {agg['skipped_days']}")
    print(f"🔧 Covariates:    {'USD/JPY + GBP/USD' if agg['use_covariates'] else 'None'}")
    if agg['use_covariates']:
        cov_mode = "Projected (no leak)" if agg.get('covariates_projected') else "Actual future (LEAK)"
        print(f"🔒 Cov mode:      {cov_mode}")

    # ── Error Metrics ───────────────────────────────────────────────────
    print("\n" + "─" * width)
    print("  ERROR METRICS".center(width))
    print("─" * width)
    print(f"  Mean MAE:           {agg['mean_mae']:.5f}")
    print(f"  Median MAE:         {agg['median_mae']:.5f}")
    print(f"  Mean RMSE:          {agg['mean_rmse']:.5f}")
    print(f"  Mean MAPE:          {agg['mean_mape']:.4f}%")
    print(f"  Mean MoE (signed):  {agg['mean_moe']:.5f}  ({agg['mean_moe_pips']:.1f} pips)")

    # ── Directional Accuracy ────────────────────────────────────────────
    print("\n" + "─" * width)
    print("  DIRECTIONAL ACCURACY".center(width))
    print("─" * width)
    print(f"  Accuracy:           {agg['directional_accuracy_pct']:.1f}%")
    print(f"  Days correct:       {agg['days_correct']} / {agg['valid_days']}")
    print(f"  Days incorrect:     {agg['days_incorrect']}")

    # ── Pip Analysis ────────────────────────────────────────────────────
    print("\n" + "─" * width)
    print("  PIP ANALYSIS".center(width))
    print("─" * width)
    print(f"  Total net pips:     {agg['total_pips']:+.1f}")
    print(f"  Total +pips:        {agg['total_positive_pips']:+.1f}")
    print(f"  Total -pips:        {agg['total_negative_pips']:+.1f}")
    print(f"  Total |pips|:       {agg['total_abs_pips']:.1f}")
    print(f"  Avg pips/day:       {agg['avg_pips_per_day']:+.1f}")
    print(f"  Avg +pips/day:      {agg['avg_positive_pips_per_day']:+.1f}")
    print(f"  Avg -pips/day:      {agg['avg_negative_pips_per_day']:+.1f}")

    # ── Win/Loss ────────────────────────────────────────────────────────
    print("\n" + "─" * width)
    print("  WIN / LOSS BREAKDOWN".center(width))
    print("─" * width)
    print(f"  Win rate:           {agg['win_rate_pct']:.1f}%")
    print(f"  Winning days:       {agg['winning_days']}")
    print(f"  Losing days:        {agg['losing_days']}")
    print(f"  Best day:           {agg['best_day']} ({agg['best_day_pips']:+.1f} pips)")
    print(f"  Worst day:          {agg['worst_day']} ({agg['worst_day_pips']:+.1f} pips)")

    # ── Per-Day Table ───────────────────────────────────────────────────
    print("\n" + "─" * width)
    print("  PER-DAY RESULTS".center(width))
    print("─" * width)

    # Format table
    table_df = df_metrics[["date", "mae", "moe", "moe_pips", "directional_correct",
                            "total_pips", "positive_pips", "negative_pips"]].copy()
    table_df["date"] = table_df["date"].str[:10]
    table_df["dir"] = table_df["directional_correct"].map({1: "✓", 0: "✗"})
    table_df = table_df.drop(columns=["directional_correct"])
    table_df.columns = ["Date", "MAE", "MoE", "MoE(pips)", "Dir",
                        "Net Pips", "+Pips", "-Pips"]

    # Print table
    print(table_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # ── Cumulative Pips ─────────────────────────────────────────────────
    print("\n" + "─" * width)
    print("  CUMULATIVE PIPS OVER TIME".center(width))
    print("─" * width)

    df_metrics["cumulative_pips"] = df_metrics["total_pips"].cumsum()
    cum_table = df_metrics[["date", "total_pips", "cumulative_pips"]].copy()
    cum_table["date"] = cum_table["date"].str[:10]
    cum_table.columns = ["Date", "Daily Pips", "Cumulative Pips"]
    print(cum_table.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    # ── Footer ──────────────────────────────────────────────────────────
    print("\n" + "=" * width)
    print(f"  Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(width))
    print("=" * width + "\n")

    # ── Save to file ────────────────────────────────────────────────────
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save full report as JSON
        report_json = {
            "aggregate": agg,
            "daily_metrics": report["daily_metrics"],
        }
        report_path = output_dir / "backtest_report.json"
        with open(report_path, "w") as f:
            json.dump(report_json, f, indent=2, default=str)
        print(f"💾 Full report saved to {report_path}")

        # Save CSV
        csv_path = output_dir / "backtest_daily.csv"
        df_metrics.to_csv(csv_path, index=False)
        print(f"💾 Daily metrics saved to {csv_path}")

        # Save formatted text report
        txt_path = output_dir / "backtest_report.txt"
        with open(txt_path, "w") as f:
            # Capture print output
            import io
            import contextlib

            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                print_backtest_report_to_file(report, f)
            f.write(buffer.getvalue())
        print(f"💾 Text report saved to {txt_path}")


def print_backtest_report_to_file(report: dict, file_handle):
    """Print report to a file handle (used for saving to .txt)."""
    # Re-use the same logic but write to file instead of stdout
    import sys
    old_stdout = sys.stdout
    sys.stdout = file_handle
    try:
        print_backtest_report(report)
    finally:
        sys.stdout = old_stdout


# ── CLI entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EUR/USD Session Forecasting Backtester")
    parser.add_argument("--start", type=str, required=True,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True,
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--covariates", action="store_true",
                        help="Enable USD/JPY + GBP/USD covariate forecasting.")
    parser.add_argument("--use-future-covariates", action="store_true",
                        help="⚠️  Use actual future covariate values (DATA LEAK — for comparison only).")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for results.")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR),
                        help="Directory containing CSV data files.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output) if args.output else None

    # Check TimesFM
    if not check_timesfm_installed():
        sys.exit(1)

    # Load data
    print("📂 Loading data...")
    eur_df = load_15m_data("EURUSD", data_dir)
    usdjpy_df = load_15m_data("USDJPY", data_dir)
    gbpusd_df = load_15m_data("GBPUSD", data_dir)

    print(f"  EUR/USD: {len(eur_df)} candles ({eur_df['datetime'].min()} → {eur_df['datetime'].max()})")
    print(f"  USD/JPY: {len(usdjpy_df)} candles")
    print(f"  GBP/USD: {len(gbpusd_df)} candles")

    # Run backtest
    # Default: project_covariates=True (no data leak — simulates live trading)
    # Use --use-future-covariates to allow actual future values (for comparison only)
    project_covs = not args.use_future_covariates
    if project_covs:
        print("🔒 Covariate mode: PROJECTED (no data leak — simulates live trading)")
    else:
        print("⚠️  Covariate mode: ACTUAL FUTURE VALUES (DATA LEAK — for comparison only)")

    report = run_backtest(
        start_date=args.start,
        end_date=args.end,
        eur_df=eur_df,
        usdjpy_df=usdjpy_df,
        gbpusd_df=gbpusd_df,
        use_covariates=args.covariates,
        project_covariates=project_covs,
        output_dir=output_dir,
    )

    # Print report
    if report:
        print_backtest_report(report, output_dir)
