# EUR/USD Session Forecasting with Covariates

Forecasts the US trading session for EUR/USD using Asia + EU session data as context,
with USD/JPY and GBP/USD as dynamic covariates.

## Session Times (UTC)

| Session | Hours | Candles (15m) |
|---------|-------|---------------|
| Asia    | 00:00 – 07:00 | 28 |
| EU      | 07:00 – 16:00 | 36 |
| **Context (Asia+EU)** | **00:00 – 16:00** | **64** |
| **US (Forecast)** | **16:00 – 21:00** | **20** |

## Usage

### Basic Forecast (Backtest Mode)

Uses actual USD/JPY and GBP/USD values for the US session:

```bash
python forecast_session.py --date 2022-04-14 --output output
```

### Live Forecast Mode

Projects covariate means for the forecast window (when future covariates are unknown):

```bash
python forecast_session.py --date 2022-04-14 --project-covariates --output output
```

### Without Covariates

```bash
python forecast_session.py --date 2022-04-14 --no-covariates --output output
```

### Programmatic Usage

```python
from forecast_session import load_15m_data, forecast_eurusd_session
import pandas as pd

# Load data
eur_df = load_15m_data("EURUSD")
usdjpy_df = load_15m_data("USDJPY")
gbpusd_df = load_15m_data("GBPUSD")

# Forecast a specific date
result = forecast_eurusd_session(
    date=pd.Timestamp("2022-04-14"),
    eur_df=eur_df,
    usdjpy_df=usdjpy_df,
    gbpusd_df=gbpusd_df,
    use_covariates=True,
    project_covariates=False,  # True for live mode
    output_dir="output",
)

# Access results
print(f"Date: {result['date']}")
print(f"Context candles: {len(result['context_values'])}")
print(f"Forecast candles: {len(result['forecast_point'])}")
print(f"MAE: {result.get('mae', 'N/A')}")
```

## Output

Results are saved as JSON files in the `output/` directory with:
- Context values and timestamps
- Forecast point predictions
- Quantile predictions (10 quantiles for prediction intervals)
- Covariate values used
- Actual values (if available for backtesting)

## Requirements

- **TimesFM 2.5** (NOT on PyPI — install from source):
  ```bash
  git clone https://github.com/google-research/timesfm.git
  cd timesfm
  pip install -e .[torch]
  pip install -e .[xreg]  # for covariate support
  ```
- PyTorch: `pip install torch`
- pandas, numpy

> ⚠️ `pip install timesfm` only installs v1.3.0 (old models). TimesFM 2.5 must be installed from the GitHub repo.
