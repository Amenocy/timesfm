"""
EUR/USD Session Forecasting — Backtester V2

Runs the forecasting pipeline over a date range with extended context:
  - Context: Previous day + Asia session + EU session
  - Forecast: US session (rest of market)

This provides the model with more historical context for better predictions.

Session times (UTC):
  - Asia:  00:00 – 07:00
  - EU:    07:00 – 16:00
  - US:    16:00 – 21:00

Context window:  Prev day full + Asia open → EU close (~112 candles)
Forecast window: US session            (20 × 15m candles)

Usage:
    python backtest_session_v2.py --start 2024-01-01 --end 2024-03-31
    python backtest_session_v2.py --start 2024-01-01 --end 2024-03-31 --no-covariates
    python backtest_session_v2.py --start 2024-01-01 --end 2024-03-31 --output results/backtest_v2
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
    check_timesfm_installed,
    FORECAST_CANDLES,
    DATA_DIR,
)

# ── Session definitions (UTC) ──────────────────────────────────────────────
ASIA_START = 0   # 00:00 UTC
ASIA_END   = 7   # 07:00 UTC
EU_START   = 7   # 07:00 UTC
EU_END     = 16  # 16:00 UTC
US_START   = 16  # 16:00 UTC
US_END     = 21  # 21:00 UTC

# Context: prev day (96 candles) + Asia+EU (64 candles) = ~160 candles
# But we'll dynamically build it, so no fixed constant needed

# ── Pip value for EUR/USD (0.0001 = 1 pip for most pairs) ────────────────
PIP_SIZE = 0.0001


def extract_session_v2(df: pd.DataFrame, date: pd.Timestamp) -> dict:
    """
    Extract prev day + Asia+EU (context) and US (target) sessions for a given date.
    
    Context includes:
    - Previous trading day: full day (00:00 - 23:45)
    - Current day: Asia start → EU end (00:00 - 16:00)
    
    Target:
    - Current day: US session (16:00 - 21:00)
    
    Returns dict with context/target DataFrames and candle counts.
    """
    day_str = date.strftime("%Y-%m-%d")
    
    # Find previous business day
    prev_date = date - pd.Timedelta(days=1)
    while prev_date.weekday() >= 5:  # Skip weekends
        prev_date -= pd.Timedelta(days=1)
    prev_str = prev_date.strftime("%Y-%m-%d")
    
    # Get current day data
    day_data = df[df["datetime"].dt.strftime("%Y-%m-%d") == day_str].copy()
    prev_data = df[df["datetime"].dt.strftime("%Y-%m-%d") == prev_str].copy()
    
    if day_data.empty:
        return {}
    
    # Context part 1: Previous day full session
    prev_context = prev_data.copy()
    
    # Context part 2: Current day Asia + EU
    current_context = day_data[
        (day_data["datetime"].dt.hour >= ASIA_START)
        & (day_data["datetime"].dt.hour < EU_END)
    ].copy()
    
    # Combine contexts
    if not prev_context.empty and not current_context.empty:
        context = pd.concat([prev_context, current_context], ignore_index=True)
    elif not current_context.empty:
        context = current_context.copy()
    elif not prev_context.empty:
        context = prev_context.copy()
    else:
        return {}
    
    # Target: US session
    target = day_data[
        (day_data["datetime"].dt.hour >= US_START)
        & (day_data["datetime"].dt.hour < US_END)
    ].copy()
    
    return {
        "date": day_str,
        "prev_date": prev_str,
        "context": context,
        "target": target,
        "prev_candles": len(prev_context),
        "current_context_candles": len(current_context),
        "total_context_candles": len(context),
        "target_candles": len(target),
    }


def build_covariate_arrays_v2(
    eur_df: pd.DataFrame,
    usdjpy_df: pd.DataFrame,
    gbpusd_df: pd.DataFrame,
    date: pd.Timestamp,
    use_actual_future: bool = False,
) -> dict:
    """
    Build aligned covariate arrays for extended context + forecast windows.
    
    Context: Previous day + Asia + EU
    Forecast: US session
    
    Parameters
    ----------
    use_actual_future : bool
        If True, uses actual US-session values from USD/JPY and GBP/USD
        (DATA LEAK — for analysis only). If False (default), projects
        context mean for the forecast window (realistic backtesting).
    """
    day_str = date.strftime("%Y-%m-%d")
    
    # Find previous business day
    prev_date = date - pd.Timedelta(days=1)
    while prev_date.weekday() >= 5:
        prev_date -= pd.Timedelta(days=1)
    prev_str = prev_date.strftime("%Y-%m-%d")
    
    # Get context timestamps (prev day + current Asia+EU)
    prev_ts = eur_df[
        eur_df["datetime"].dt.strftime("%Y-%m-%d") == prev_str
    ]["datetime"].values
    
    current_context_ts = eur_df[
        (eur_df["datetime"].dt.strftime("%Y-%m-%d") == day_str)
        & (eur_df["datetime"].dt.hour >= ASIA_START)
        & (eur_df["datetime"].dt.hour < EU_END)
    ]["datetime"].values
    
    context_ts = np.concatenate([prev_ts, current_context_ts])
    
    if len(context_ts) == 0:
        return {}
    
    # Get forecast window timestamps
    forecast_ts = eur_df[
        (eur_df["datetime"].dt.strftime("%Y-%m-%d") == day_str)
        & (eur_df["datetime"].dt.hour >= US_START)
        & (eur_df["datetime"].dt.hour < US_END)
    ]["datetime"].values
    
    # Build full timestamp array (context + forecast)
    all_ts = np.concatenate([context_ts, forecast_ts])
    
    # Helper: align a symbol's close prices to the timestamp array
    def align_close(sym_df: pd.DataFrame) -> np.ndarray:
        sym_df = sym_df.copy()
        sym_df = sym_df.set_index("datetime")
        prices = []
        for ts in all_ts:
            mask = sym_df.index <= ts
            if mask.any():
                prices.append(sym_df.loc[mask, "close"].iloc[-1])
            else:
                prices.append(sym_df.iloc[0]["close"])
        return np.array(prices, dtype=np.float32)
    
    usdjpy_close = align_close(usdjpy_df)
    gbpusd_close = align_close(gbpusd_df)
    
    # If using actual future covariates, keep them; otherwise project context mean
    if not use_actual_future:
        ctx_len = len(context_ts)
        usdjpy_mean = usdjpy_close[:ctx_len].mean()
        gbpusd_mean = gbpusd_close[:ctx_len].mean()
        usdjpy_close[ctx_len:] = usdjpy_mean
        gbpusd_close[ctx_len:] = gbpusd_mean
    
    return {
        "usdjpy_close": usdjpy_close,
        "gbpusd_close": gbpusd_close,
        "context_len": len(context_ts),
        "forecast_len": len(forecast_ts),
    }


def prices_to_pips(forecast: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Convert absolute price differences to pips (always positive)."""
    return np.abs(forecast - actual) / PIP_SIZE


def compute_daily_metrics(result: dict) -> dict:
    """
    Compute per-day metrics from a single forecast result.
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
    pips = prices_to_pips(forecast, actual)  # absolute pip errors per candle
    
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
    
    # Pip metrics (all in absolute terms)
    total_pips = float(np.sum(pips))  # total absolute pip error
    avg_pips = float(np.mean(pips))   # mean pip error per candle
    max_pips = float(np.max(pips))    # worst single candle error
    min_pips = float(np.min(pips))    # best single candle error
    
    return {
        "date": result["date"],
        "num_candles": min_len,
        "context_candles": result.get("total_context_candles", 0),
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "moe": moe,
        "moe_pips": moe / PIP_SIZE,
        "directional_correct": directional_correct,
        "total_pips": total_pips,
        "avg_pips": avg_pips,
        "max_pips": max_pips,
        "min_pips": min_pips,
    }


def run_backtest_v2(
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
    Run backtest V2 over a date range with extended context.
    
    Context: Previous day + Asia + EU sessions
    Forecast: US session
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
    print(f"\n📅 Backtest V2 range: {start_date} → {end_date} ({len(dates)} business days)")
    print(f"   Context: Prev day + Asia (00:00-07:00) + EU (07:00-16:00)")
    print(f"   Forecast: US session (16:00-21:00)")
    
    daily_results = []
    skipped = []
    
    # ── 3. Loop over dates ──────────────────────────────────────────────
    for i, date in enumerate(dates):
        print(f"\n[{i+1}/{len(dates)}] Processing {date.strftime('%Y-%m-%d')}...")
        
        # Extract session with extended context
        session = extract_session_v2(eur_df, date)
        if not session or session["total_context_candles"] < 32:
            print(f"  ⚠️  Insufficient context, skipping.")
            skipped.append(date.strftime("%Y-%m-%d"))
            continue
        
        context_values = session["context"]["close"].values.astype(np.float32)
        actual_values = session["target"]["close"].values.astype(np.float32) if len(session["target"]) > 0 else None
        
        if actual_values is None or len(actual_values) == 0:
            print(f"  ⚠️  No target data, skipping.")
            skipped.append(date.strftime("%Y-%m-%d"))
            continue
        
        print(f"  📊 Context: {session['total_context_candles']} candles "
              f"(prev: {session['prev_candles']}, current: {session['current_context_candles']})")
        print(f"  🎯 Target: {len(actual_values)} candles")
        
        # Build covariates
        covariates = {}
        if use_covariates:
            covariates = build_covariate_arrays_v2(
                eur_df, usdjpy_df, gbpusd_df, date,
                use_actual_future=False,  # Always project means — no data leakage
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
                "prev_date": session["prev_date"],
                "context_values": context_values.tolist(),
                "total_context_candles": session["total_context_candles"],
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
                out_path = output_dir / f"backtest_v2_{session['date']}.json"
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
        "context_type": "prev_day + asia + eu",
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
        "avg_pips_per_candle": float(df_metrics["avg_pips"].mean()),
        "avg_pips_per_day": float(df_metrics["total_pips"].mean()),
        "max_pips_single_candle": float(df_metrics["max_pips"].max()),
        "min_pips_single_candle": float(df_metrics["min_pips"].min()),
        # Best / worst days
        "best_day": df_metrics.loc[df_metrics["total_pips"].idxmax(), "date"],
        "best_day_pips": float(df_metrics["total_pips"].max()),
        "worst_day": df_metrics.loc[df_metrics["total_pips"].idxmin(), "date"],
        "worst_day_pips": float(df_metrics["total_pips"].min()),
        # Win rate
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
    
    width = 80
    print("\n" + "=" * width)
    print("  EUR/USD SESSION FORECAST — BACKTEST V2 REPORT".center(width))
    print("=" * width)
    
    print(f"\n📅 Period:        {agg['start_date']} → {agg['end_date']}")
    print(f"📊 Total days:    {agg['total_days']}")
    print(f"✅ Valid days:    {agg['valid_days']}")
    print(f"⚠️  Skipped days: {agg['skipped_days']}")
    print(f"🔧 Context:       {agg.get('context_type', 'N/A')}")
    print(f"🔧 Covariates:    {'USD/JPY + GBP/USD' if agg['use_covariates'] else 'None'}")
    if agg['use_covariates']:
        cov_mode = "Projected (no leak)" if agg.get('covariates_projected') else "Actual future (LEAK)"
        print(f"🔒 Cov mode:      {cov_mode}")
    
    print("\n" + "─" * width)
    print("  ERROR METRICS".center(width))
    print("─" * width)
    print(f"  Mean MAE:           {agg['mean_mae']:.5f}")
    print(f"  Median MAE:         {agg['median_mae']:.5f}")
    print(f"  Mean RMSE:          {agg['mean_rmse']:.5f}")
    print(f"  Mean MAPE:          {agg['mean_mape']:.4f}%")
    print(f"  Mean MoE (signed):  {agg['mean_moe']:.5f}  ({agg['mean_moe_pips']:.1f} pips)")
    
    print("\n" + "─" * width)
    print("  DIRECTIONAL ACCURACY".center(width))
    print("─" * width)
    print(f"  Accuracy:           {agg['directional_accuracy_pct']:.1f}%")
    print(f"  Days correct:       {agg['days_correct']} / {agg['valid_days']}")
    print(f"  Days incorrect:     {agg['days_incorrect']}")
    
    print("\n" + "─" * width)
    print("  PIP ANALYSIS (Absolute Forecast Error)".center(width))
    print("─" * width)
    print(f"  Total pip error:    {agg['total_pips']:.1f} pips")
    print(f"  Avg pip error/day:  {agg['avg_pips_per_day']:.1f} pips")
    print(f"  Avg pip error/cndl: {agg['avg_pips_per_candle']:.2f} pips")
    print(f"  Max single candle:  {agg['max_pips_single_candle']:.1f} pips")
    print(f"  Min single candle:  {agg['min_pips_single_candle']:.1f} pips")
    
    print("\n" + "─" * width)
    print("  WIN / LOSS BREAKDOWN".center(width))
    print("─" * width)
    print(f"  Win rate:           {agg['win_rate_pct']:.1f}%")
    print(f"  Winning days:       {agg['winning_days']}")
    print(f"  Losing days:        {agg['losing_days']}")
    print(f"  Best day:           {agg['best_day']} ({agg['best_day_pips']:+.1f} pips)")
    print(f"  Worst day:          {agg['worst_day']} ({agg['worst_day_pips']:+.1f} pips)")
    
    print("\n" + "─" * width)
    print("  PER-DAY RESULTS".center(width))
    print("─" * width)
    
    table_df = df_metrics[["date", "mae", "moe", "moe_pips", "directional_correct",
                            "total_pips", "avg_pips", "max_pips"]].copy()
    table_df["date"] = table_df["date"].str[:10]
    table_df["dir"] = table_df["directional_correct"].map({1: "✓", 0: "✗"})
    table_df = table_df.drop(columns=["directional_correct"])
    table_df.columns = ["Date", "MAE", "MoE", "MoE(pips)", "Dir",
                        "Total Pips", "Avg Pips", "Max Pips"]
    
    print(table_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    
    print("\n" + "─" * width)
    print("  CUMULATIVE PIPS OVER TIME".center(width))
    print("─" * width)
    
    df_metrics["cumulative_pips"] = df_metrics["total_pips"].cumsum()
    cum_table = df_metrics[["date", "total_pips", "cumulative_pips"]].copy()
    cum_table["date"] = cum_table["date"].str[:10]
    cum_table.columns = ["Date", "Daily Pips", "Cumulative Pips"]
    print(cum_table.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    
    print("\n" + "=" * width)
    
    # Save report to file
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save aggregate metrics
        agg_path = output_dir / "backtest_v2_aggregate.json"
        with open(agg_path, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"\n💾 Aggregate metrics saved to {agg_path}")
        
        # Save daily metrics
        daily_path = output_dir / "backtest_v2_daily.csv"
        df_metrics.to_csv(daily_path, index=False)
        print(f"💾 Daily metrics saved to {daily_path}")


# ── CLI entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EUR/USD Session Forecasting — Backtester V2")
    parser.add_argument("--start", type=str, required=True,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True,
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--no-covariates", action="store_true",
                        help="Disable covariate forecasting.")
    parser.add_argument("--leak-covariates", action="store_true",
                        help="Use actual future covariate values (DATA LEAK — analysis only).")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for results.")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR),
                        help="Directory containing CSV data files.")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output) if args.output else None
    
    # Load data
    print("📂 Loading data...")
    eur_df = load_15m_data("EURUSD", data_dir)
    usdjpy_df = load_15m_data("USDJPY", data_dir)
    gbpusd_df = load_15m_data("GBPUSD", data_dir)
    
    print(f"  EUR/USD: {len(eur_df)} candles ({eur_df['datetime'].min()} → {eur_df['datetime'].max()})")
    print(f"  USD/JPY: {len(usdjpy_df)} candles")
    print(f"  GBP/USD: {len(gbpusd_df)} candles")
    
    # Run backtest
    report = run_backtest_v2(
        start_date=args.start,
        end_date=args.end,
        eur_df=eur_df,
        usdjpy_df=usdjpy_df,
        gbpusd_df=gbpusd_df,
        use_covariates=not args.no_covariates,
        project_covariates=not args.leak_covariates,
        output_dir=output_dir,
    )
    
    # Print report
    print_backtest_report(report, output_dir)
