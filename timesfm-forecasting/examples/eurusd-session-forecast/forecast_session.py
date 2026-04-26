"""
EUR/USD Session Forecasting with Covariates (USD/JPY, GBP/USD)

Forecasts the US session using Asia + EU session as context,
with USD/JPY and GBP/USD as dynamic covariates.

Session times (UTC):
  - Asia:  00:00 – 07:00
  - EU:    07:00 – 16:00
  - US:    16:00 – 21:00

Context window:  Asia open → EU close  (64 × 15m candles)
Forecast window: US session            (20 × 15m candles)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# ── Session definitions (UTC) ──────────────────────────────────────────────
ASIA_START = 0   # 00:00 UTC
ASIA_END   = 7   # 07:00 UTC
EU_START   = 7   # 07:00 UTC
EU_END     = 16  # 16:00 UTC
US_START   = 16  # 16:00 UTC
US_END     = 21  # 21:00 UTC

CONTEXT_CANDLES = (EU_END - ASIA_START) * 4   # 64 candles (15m)
FORECAST_CANDLES = (US_END - US_START) * 4     # 20 candles (15m)

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "data"


def load_15m_data(symbol: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load 15-minute OHLCV CSV and return a clean DataFrame."""
    csv_path = data_dir / f"{symbol}15.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    # CSV format: "YYYY-MM-DD HH:MM open high low close volume"
    # The whitespace separator splits date and time into separate columns
    df = pd.read_csv(
        csv_path,
        sep=r"\s+",
        header=None,
        names=["date", "time", "open", "high", "low", "close", "volume"],
    )
    # Combine date and time into a single datetime column
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y-%m-%d %H:%M")
    df = df.drop(columns=["date", "time"])
    df = df.sort_values("datetime").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(np.float32)
    return df


def check_timesfm_installed() -> bool:
    """Check if timesfm 2.5 is installed and provide installation instructions."""
    try:
        import timesfm
        # Verify it has the 2.5 class
        if not hasattr(timesfm, "TimesFM_2p5_200M_torch"):
            print("\n❌ timesfm is installed but NOT version 2.5.")
            print("   pip install timesfm gives v1.3.0 (old models only).")
            print("   TimesFM 2.5 must be installed from source:")
            print("   git clone https://github.com/google-research/timesfm.git")
            print("   cd timesfm")
            print("   pip install -e .[torch]")
            print("   pip install -e .[xreg]  # for covariate support")
            return False
        return True
    except ImportError:
        print("\n❌ timesfm is not installed.")
        print("   TimesFM 2.5 is NOT on PyPI — install from source:")
        print("   git clone https://github.com/google-research/timesfm.git")
        print("   cd timesfm")
        print("   pip install -e .[torch]")
        print("   pip install -e .[xreg]  # for covariate support")
        print("\n   Or with uv (faster):")
        print("   uv venv && source .venv/bin/activate")
        print("   uv pip install -e .[torch]")
        print("   uv pip install -e .[xreg]")
        return False


def extract_session(df: pd.DataFrame, date: pd.Timestamp) -> dict:
    """
    Extract Asia+EU (context) and US (target) sessions for a given date.
    Returns dict with context/target DataFrames and candle counts.
    """
    day_str = date.strftime("%Y-%m-%d")
    day_data = df[df["datetime"].dt.strftime("%Y-%m-%d") == day_str].copy()

    if day_data.empty:
        return {}

    # Context: Asia start → EU end
    context = day_data[
        (day_data["datetime"].dt.hour >= ASIA_START)
        & (day_data["datetime"].dt.hour < EU_END)
    ].copy()

    # Target: US session
    target = day_data[
        (day_data["datetime"].dt.hour >= US_START)
        & (day_data["datetime"].dt.hour < US_END)
    ].copy()

    return {
        "date": day_str,
        "context": context,
        "target": target,
        "context_candles": len(context),
        "target_candles": len(target),
    }


def build_covariate_arrays(
    eur_df: pd.DataFrame,
    usdjpy_df: pd.DataFrame,
    gbpusd_df: pd.DataFrame,
    date: pd.Timestamp,
    include_future: bool = True,
) -> dict:
    """
    Build aligned covariate arrays for context + forecast windows.

    Parameters
    ----------
    include_future : bool
        If True, uses actual US-session values from USD/JPY and GBP/USD
        (for backtesting). If False, projects context mean for the forecast
        window (for live forecasting when future covariates are unknown).
    """
    day_str = date.strftime("%Y-%m-%d")

    # Get context window timestamps
    context_ts = eur_df[
        (eur_df["datetime"].dt.strftime("%Y-%m-%d") == day_str)
        & (eur_df["datetime"].dt.hour >= ASIA_START)
        & (eur_df["datetime"].dt.hour < EU_END)
    ]["datetime"].values

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
            # Get all rows up to and including this timestamp
            mask = sym_df.index <= ts
            if mask.any():
                prices.append(sym_df.loc[mask, "close"].iloc[-1])
            else:
                # Fallback: use first available price
                prices.append(sym_df.iloc[0]["close"])
        return np.array(prices, dtype=np.float32)

    usdjpy_close = align_close(usdjpy_df)
    gbpusd_close = align_close(gbpusd_df)

    # If not including future, replace forecast window with context mean
    if not include_future:
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


def forecast_eurusd_session(
    date: pd.Timestamp,
    eur_df: pd.DataFrame,
    usdjpy_df: pd.DataFrame,
    gbpusd_df: pd.DataFrame,
    use_covariates: bool = True,
    project_covariates: bool = False,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Main forecasting function.

    Parameters
    ----------
    date : pd.Timestamp
        The trading date to forecast.
    eur_df, usdjpy_df, gbpusd_df : pd.DataFrame
        15-minute OHLCV DataFrames loaded via `load_15m_data()`.
    use_covariates : bool
        Whether to use USD/JPY and GBP/USD as covariates.
    project_covariates : bool
        If True, projects covariate means for the forecast window
        (live mode). If False, uses actual values (backtest mode).
    output_dir : Path, optional
        Directory to save results. If None, results are returned only.

    Returns
    -------
    dict with keys:
        - date: trading date
        - context_values: EUR/USD close prices (Asia → EU)
        - forecast_point: predicted US session close prices
        - forecast_quantiles: quantile predictions (10 quantiles)
        - covariates: USD/JPY and GBP/USD arrays used
        - actual: actual US session close prices (if available)
    """
    if not check_timesfm_installed():
        sys.exit(1)

    import torch
    import timesfm

    # ── 1. Extract session data ─────────────────────────────────────────
    session = extract_session(eur_df, date)
    if not session or session["context_candles"] < 32:
        print(f"⚠️  Insufficient context for {date}: {session.get('context_candles', 0)} candles")
        return {}

    context_values = session["context"]["close"].values.astype(np.float32)
    actual_values = session["target"]["close"].values.astype(np.float32) if len(session["target"]) > 0 else None

    print(f"📊 {session['date']}: context={len(context_values)} candles, "
          f"target={len(actual_values) if actual_values is not None else 0} candles")

    # ── 2. Build covariates ─────────────────────────────────────────────
    covariates = {}
    if use_covariates:
        covariates = build_covariate_arrays(
            eur_df, usdjpy_df, gbpusd_df, date,
            include_future=not project_covariates,
        )
        if not covariates:
            print("⚠️  Could not build covariate arrays, falling back to no-covariate mode")
            use_covariates = False

    # ── 3. Load model ───────────────────────────────────────────────────
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
        infer_is_positive=False,  # FX prices can have negative returns
        fix_quantile_crossing=True,
        return_backcast=use_covariates,  # required for covariate mode
    ))

    # ── 4. Forecast ─────────────────────────────────────────────────────
    horizon = FORECAST_CANDLES if len(covariates) == 0 else covariates["forecast_len"]
    if horizon == 0:
        horizon = FORECAST_CANDLES

    print(f"🎯 Forecasting {horizon} candles...")

    if use_covariates and covariates:
        # Covariate forecasting mode
        # forecast_with_covariates infers horizon from covariate array length
        # beyond the context window. Covariate arrays must span context + forecast.
        point, quantiles = model.forecast_with_covariates(
            inputs=[context_values],
            dynamic_numerical_covariates={
                "usdjpy": [covariates["usdjpy_close"]],
                "gbpusd": [covariates["gbpusd_close"]],
            },
            xreg_mode="xreg + timesfm",
        )
    else:
        # Standard forecasting (no covariates)
        point, quantiles = model.forecast(
            horizon=horizon,
            inputs=[context_values],
        )

    # ── 5. Build result ─────────────────────────────────────────────────
    result = {
        "date": session["date"],
        "context_values": context_values.tolist(),
        "context_timestamps": session["context"]["datetime"].dt.strftime(
            "%Y-%m-%d %H:%M"
        ).tolist(),
        "forecast_point": point[0].tolist(),
        "forecast_quantiles": quantiles[0].tolist(),
        "forecast_timestamps": session["target"]["datetime"].dt.strftime(
            "%Y-%m-%d %H:%M"
        ).tolist() if len(session["target"]) > 0 else [],
        "actual_values": actual_values.tolist() if actual_values is not None else None,
        "covariates_used": use_covariates,
        "covariate_values": {
            "usdjpy": covariates.get("usdjpy_close", []).tolist(),
            "gbpusd": covariates.get("gbpusd_close", []).tolist(),
        } if covariates else None,
    }

    # ── 6. Save output ──────────────────────────────────────────────────
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"forecast_{session['date']}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"💾 Saved forecast to {out_path}")

    return result


def forecast_multiple_days(
    dates: list[pd.Timestamp],
    eur_df: pd.DataFrame,
    usdjpy_df: pd.DataFrame,
    gbpusd_df: pd.DataFrame,
    use_covariates: bool = True,
    project_covariates: bool = False,
    output_dir: Optional[Path] = None,
) -> list[dict]:
    """Forecast multiple days in sequence."""
    results = []
    for date in dates:
        result = forecast_eurusd_session(
            date, eur_df, usdjpy_df, gbpusd_df,
            use_covariates=use_covariates,
            project_covariates=project_covariates,
            output_dir=output_dir,
        )
        if result:
            results.append(result)
    return results


# ── CLI entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EUR/USD session forecasting with covariates")
    parser.add_argument("--date", type=str, default=None,
                        help="Trading date to forecast (YYYY-MM-DD). Default: last available date.")
    parser.add_argument("--no-covariates", action="store_true",
                        help="Disable covariate forecasting.")
    parser.add_argument("--project-covariates", action="store_true",
                        help="Project covariate means for forecast window (live mode).")
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

    # Determine date
    if args.date:
        target_date = pd.Timestamp(args.date)
    else:
        # Use the last date with sufficient data
        last_date = eur_df["datetime"].max().normalize()
        target_date = last_date

    print(f"\n🎯 Forecasting for: {target_date.strftime('%Y-%m-%d')}")
    print(f"   Context: {ASIA_START:02d}:00 – {EU_END:02d}:00 UTC")
    print(f"   Forecast: {US_START:02d}:00 – {US_END:02d}:00 UTC")
    print(f"   Covariates: {'USD/JPY + GBP/USD' if not args.no_covariates else 'None'}")
    print(f"   Covariate mode: {'projected (live)' if args.project_covariates else 'actual (backtest)'}")

    result = forecast_eurusd_session(
        target_date, eur_df, usdjpy_df, gbpusd_df,
        use_covariates=not args.no_covariates,
        project_covariates=args.project_covariates,
        output_dir=output_dir,
    )

    if result:
        print(f"\n✅ Forecast complete for {result['date']}")
        print(f"   Context: {len(result['context_values'])} candles")
        print(f"   Forecast: {len(result['forecast_point'])} candles")
        if result["actual_values"]:
            mae = np.mean(np.abs(
                np.array(result["actual_values"]) - np.array(result["forecast_point"])
            ))
            print(f"   MAE vs actual: {mae:.5f}")
