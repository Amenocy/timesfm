#!/usr/bin/env python3
"""Fine-tune TimesFM 2.5 with LoRA on 15-min forex candle data.

Trains LoRA adapters on OHLCV forex data (15-minute candles) for close-price
forecasting.  Supports multiple currency pairs loaded from CSV files.

Data format (no header, tab/space separated):
    datetime  open  high  low  close  volume

Each series has ~100K data points spanning Apr 2022 – Apr 2026.

Usage:
    python finetune_forex_lora.py [OPTIONS]

    Options:
        --model_id       HuggingFace model ID (default: google/timesfm-2.5-200m-transformers)
        --data_dir       Directory containing forex CSV files (default: ../../data)
        --context_len    Context length in 15-min candles (default: 672 = 7 days)
        --horizon_len    Forecast horizon in candles (default: 96 = 1 day)
        --epochs         Number of training epochs (default: 10)
        --batch_size     Training batch size (default: 32)
        --lr             Learning rate (default: 1e-4)
        --lora_r         LoRA rank (default: 4)
        --lora_alpha     LoRA alpha (default: 8)
        --lora_dropout   LoRA dropout (default: 0.05)
        --num_samples    Number of random training windows to pre-sample (default: 10000)
        --output_dir     Directory to save the LoRA adapter (default: timesfm2_5-forex-lora)
        --seed           Random seed (default: 42)
        --test_split     Fraction of each series to reserve for testing (default: 0.1)
        --weight_decay   Weight decay for AdamW optimizer (default: 0.01)
        --patience       Early stopping patience in epochs (default: 5, 0 to disable)
        --warmup_epochs  Learning rate warmup epochs (default: 1)
"""

import argparse
import glob
import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TimeSeriesRandomWindowDataset(Dataset):
    """Random-window dataset for time series fine-tuning.

    Pre-samples random (series, split-point) windows.  Each window has a full
    *context_len* context (no zero-padding) to avoid corrupting TimesFM's
    internal RevIN normalisation statistics.
    """

    def __init__(
        self,
        series_list: list[np.ndarray],
        context_len: int,
        horizon_len: int,
        num_samples: int = 10000,
        seed: int = 42,
    ):
        self.series_list = series_list
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.samples: list[tuple[int, int]] = []

        rng = np.random.default_rng(seed)
        min_len = context_len + horizon_len
        valid = [i for i, s in enumerate(series_list) if len(s) >= min_len]
        if not valid:
            raise ValueError(
                f"No series long enough for context_len={context_len} + "
                f"horizon_len={horizon_len}. Shortest series: "
                f"{min(len(s) for s in series_list)}"
            )

        for _ in range(num_samples):
            idx = rng.choice(valid)
            series = series_list[idx]
            max_start = len(series) - min_len
            start = rng.integers(0, max_start + 1)
            self.samples.append((idx, start))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        idx, start = self.samples[i]
        series = self.series_list[idx]
        end = start + self.context_len + self.horizon_len

        context = torch.tensor(
            series[start : start + self.context_len], dtype=torch.float32
        )
        target = torch.tensor(
            series[start + self.context_len : end], dtype=torch.float32
        )
        return context, target


class TimeSeriesLastWindowDataset(Dataset):
    """Validation dataset using the last window of each series."""

    def __init__(
        self,
        series_list: list[np.ndarray],
        context_len: int,
        horizon_len: int,
    ):
        self.items: list[tuple[torch.Tensor, torch.Tensor]] = []
        min_len = context_len + horizon_len
        for s in series_list:
            if len(s) >= min_len:
                ctx = torch.tensor(s[-min_len:-horizon_len], dtype=torch.float32)
                tgt = torch.tensor(s[-horizon_len:], dtype=torch.float32)
                self.items.append((ctx, tgt))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        return self.items[i]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_forex_data(
    data_dir: str,
    context_len: int,
    horizon_len: int,
    num_samples: int,
    seed: int,
    test_split: float = 0.1,
    max_series_len: int | None = None,
) -> tuple[TimeSeriesRandomWindowDataset, TimeSeriesLastWindowDataset]:
    """Load forex CSV files and prepare train/test datasets.

    Each CSV file is treated as a separate time series.  The close price
    (column index 4) is used as the target value.  No covariates — purely
    univariate.  The last *test_split* fraction of each series is reserved
    for testing; the rest is used for training.

    Args:
        data_dir: Directory containing forex CSV files.
        context_len: Context window length in candles.
        horizon_len: Forecast horizon in candles.
        num_samples: Number of random training windows to sample.
        seed: Random seed for reproducibility.
        test_split: Fraction of each series to reserve for testing (default 0.1).
        max_series_len: If set, truncate each series to this many candles.

    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    logger.info("Found %d forex CSV files: %s", len(csv_files),
                [os.path.basename(f) for f in csv_files])

    train_series: list[np.ndarray] = []
    test_series: list[np.ndarray] = []

    for fpath in tqdm(csv_files, desc="Loading data", unit="file"):
        name = os.path.basename(fpath)
        # No header: datetime, open, high, low, close, volume
        df = pd.read_csv(
            fpath,
            sep=r"\s+",
            header=None,
            names=["datetime", "open", "high", "low", "close", "volume"],
            engine="python",
        )
        close = df["close"].values.astype(np.float32)

        # Optional truncation for quick testing
        if max_series_len is not None and len(close) > max_series_len:
            close = close[:max_series_len]
            logger.info("  %s: truncated to %d candles", name, len(close))

        # Last test_split fraction for testing, rest for training
        split_idx = int(len(close) * (1 - test_split))
        train_part = close[:split_idx]
        test_part = close[split_idx:]

        logger.info(
            "  %s: total=%d, train=%d, test=%d",
            name, len(close), len(train_part), len(test_part),
        )

        if len(train_part) >= context_len + horizon_len:
            train_series.append(train_part)
        if len(test_part) >= context_len + horizon_len:
            test_series.append(test_part)

    logger.info(
        "Valid train series: %d | Valid test series: %d",
        len(train_series), len(test_series),
    )

    train_ds = TimeSeriesRandomWindowDataset(
        train_series, context_len, horizon_len,
        num_samples=num_samples, seed=seed,
    )
    test_ds = TimeSeriesLastWindowDataset(
        test_series, context_len, horizon_len,
    )
    return train_ds, test_ds


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    from peft import LoraConfig, get_peft_model
    from transformers import TimesFm2_5ModelForPrediction

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    logger.info("Loading model: %s", args.model_id)
    model = TimesFm2_5ModelForPrediction.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    horizon_len = args.horizon_len
    context_len = min(args.context_len, model.config.context_length)

    # Round context_len down to nearest multiple of 32 (patch size)
    context_len = (context_len // 32) * 32
    logger.info("Using context_len=%d, horizon_len=%d", context_len, horizon_len)

    # ------------------------------------------------------------------
    # Apply LoRA
    # ------------------------------------------------------------------
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules="all-linear",
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Prepare data
    # ------------------------------------------------------------------
    train_ds, test_ds = load_forex_data(
        args.data_dir, context_len, horizon_len,
        num_samples=args.num_samples,
        seed=args.seed,
        test_split=args.test_split,
        max_series_len=args.max_series_len,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    logger.info(
        "Train samples: %d (%d batches) | Test samples: %d",
        len(train_ds), len(train_loader), len(test_ds),
    )

    # ------------------------------------------------------------------
    # Optimiser & scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Warmup + cosine decay
    warmup_steps = args.warmup_epochs * len(train_loader)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        anneal_strategy="cos",
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_test_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for context, target_vals in pbar:
            context = context.to(device)
            target_vals = target_vals.to(device)

            outputs = model(
                past_values=context,
                future_values=target_vals,
                forecast_context_len=context_len,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Test
        model.eval()
        test_loss = 0.0
        test_batches = 0
        with torch.no_grad():
            for context, target_vals in tqdm(test_loader, desc=f"  Testing {epoch}", unit="batch"):
                context = context.to(device)
                target_vals = target_vals.to(device)
                outputs = model(
                    past_values=context,
                    future_values=target_vals,
                    forecast_context_len=context_len,
                )
                test_loss += outputs.loss.item()
                test_batches += 1

        avg_test_loss = test_loss / max(test_batches, 1)

        logger.info(
            "Epoch %d/%d (%d steps) — train loss: %.4f, test loss: %.4f",
            epoch, args.epochs, n_batches,
            avg_train_loss, avg_test_loss,
        )

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            patience_counter = 0
            model.save_pretrained(args.output_dir)
            logger.info("  ✓ saved best adapter → %s", args.output_dir)
        else:
            patience_counter += 1
            logger.info(
                "  ⏳ no improvement (patience %d/%d)",
                patience_counter, args.patience,
            )
            if args.patience > 0 and patience_counter >= args.patience:
                logger.info(
                    "Early stopping at epoch %d (best test loss: %.4f)",
                    epoch, best_test_loss,
                )
                break

    logger.info("Training complete. Best test loss: %.4f", best_test_loss)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    """Compare zero-shot vs fine-tuned on forex validation data."""
    from peft import PeftModel
    from transformers import TimesFm2_5ModelForPrediction

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading base model …")
    base_model = TimesFm2_5ModelForPrediction.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    base_model.eval()
    horizon_len = args.horizon_len
    context_len = min(args.context_len, base_model.config.context_length)
    context_len = (context_len // 32) * 32

    logger.info("Loading LoRA adapter from %s …", args.output_dir)
    ft_model = PeftModel.from_pretrained(base_model, args.output_dir)
    ft_model.eval()

    # Load forex data for evaluation
    csv_files = sorted(glob.glob(os.path.join(args.data_dir, "*.csv")))
    base_maes: list[float] = []
    ft_maes: list[float] = []

    for fpath in csv_files:
        name = os.path.basename(fpath)
        df = pd.read_csv(
            fpath, sep=r"\s+", header=None,
            names=["datetime", "open", "high", "low", "close", "volume"],
            engine="python",
        )
        close = df["close"].values.astype(np.float32)

        if len(close) < context_len + horizon_len:
            logger.info("  %s: too short for evaluation, skipping", name)
            continue

        # Use last window of the full series
        min_len = context_len + horizon_len
        test_input = torch.tensor(
            close[-min_len:-horizon_len], dtype=torch.float32, device=device
        ).unsqueeze(0)
        ground_truth = close[-horizon_len:]

        with torch.no_grad():
            base_out = base_model(past_values=test_input)
            ft_out = ft_model(past_values=test_input)

        base_forecast = base_out.mean_predictions[0, :horizon_len].float().cpu().numpy()
        ft_forecast = ft_out.mean_predictions[0, :horizon_len].float().cpu().numpy()

        base_mae = float(np.abs(base_forecast - ground_truth).mean())
        ft_mae = float(np.abs(ft_forecast - ground_truth).mean())
        base_maes.append(base_mae)
        ft_maes.append(ft_mae)

        logger.info(
            "  %s — zero-shot MAE: %.6f, LoRA MAE: %.6f",
            name, base_mae, ft_mae,
        )

    if base_maes:
        avg_base = np.mean(base_maes)
        avg_ft = np.mean(ft_maes)
        improvement = (avg_base - avg_ft) / avg_base * 100
        logger.info("Average zero-shot MAE: %.6f", avg_base)
        logger.info("Average LoRA MAE:      %.6f", avg_ft)
        logger.info("Improvement:           %.1f%%", improvement)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune TimesFM 2.5 with LoRA on 15-min forex data"
    )
    p.add_argument(
        "--model_id", type=str,
        default="google/timesfm-2.5-200m-transformers",
    )
    p.add_argument(
        "--data_dir", type=str,
        default="/home/aminabb/workspace/data",
        help="Directory containing forex CSV files",
    )
    p.add_argument(
        "--context_len", type=int, default=672,
        help="Context length in 15-min candles (672 = 7 days, must be multiple of 32)",
    )
    p.add_argument(
        "--horizon_len", type=int, default=96,
        help="Forecast horizon in candles (96 = 1 day)",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora_r", type=int, default=4)
    p.add_argument("--lora_alpha", type=int, default=8)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--num_samples", type=int, default=10000)
    p.add_argument("--output_dir", type=str, default="timesfm2_5-forex-lora")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--test_split", type=float, default=0.1,
        help="Fraction of each series to reserve for testing (default: 0.1)",
    )
    p.add_argument(
        "--weight_decay", type=float, default=0.01,
        help="Weight decay for AdamW optimizer (default: 0.01)",
    )
    p.add_argument(
        "--patience", type=int, default=5,
        help="Early stopping patience in epochs (default: 5, 0 to disable)",
    )
    p.add_argument(
        "--warmup_epochs", type=int, default=1,
        help="Learning rate warmup epochs (default: 1)",
    )
    p.add_argument(
        "--evaluate", action="store_true",
        help="Run evaluation instead of training",
    )
    p.add_argument(
        "--max_series_len", type=int, default=None,
        help="Truncate each series to this many candles (for quick testing)",
    )
    p.add_argument(
        "--quick", action="store_true",
        help="Quick test mode: --num_samples=64 --epochs=2 --context_len=64 --horizon_len=32 --max_series_len=5000",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Quick test mode overrides
    if args.quick:
        logger.info("Quick test mode enabled")
        args.num_samples = 64
        args.epochs = 2
        args.context_len = 64
        args.horizon_len = 32
        args.max_series_len = 5000
        args.batch_size = 8

    if args.evaluate:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
