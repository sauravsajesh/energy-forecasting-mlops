import os
from pathlib import Path

import numpy as np
import pandas as pd


RAW_PATH = Path("data/raw/germany_energy_consumption.csv")
PROCESSED_PATH = Path("data/processed/germany_energy_processed.csv")


def load_raw_opsd(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load raw OPSD Germany time-series data."""
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at {path}. Run download script first.")
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    df = df.sort_index()
    return df


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning + feature engineering for energy forecasting."""
    df = df.copy()

    # Keep one main target column: actual load (adjust name if needed)
    # Common OPSD load column example: 'DE_load_actual_entsoe_transparency' [web:45][web:51]
    candidates = [c for c in df.columns if "load" in c.lower() and "DE_" in c]
    if not candidates:
        # If no specific load column, just use first column
        target_col = df.columns[0]
    else:
        target_col = candidates[0]

    df = df[[target_col]].rename(columns={target_col: "load_MW"})

    # Drop duplicates
    df = df[~df.index.duplicated(keep="first")]

    # Uniform hourly frequency (fill gaps)
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_index)

    # Handle missing values: linear interpolation then ffill/bfill
    missing_before = df["load_MW"].isna().mean()
    df["load_MW"] = df["load_MW"].interpolate(limit_direction="both")
    df["load_MW"] = df["load_MW"].ffill().bfill()
    missing_after = df["load_MW"].isna().mean()

    # Time-based features
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Rolling stats (simple ones for now)
    df["load_rolling_24h_mean"] = df["load_MW"].rolling(window=24, min_periods=1).mean()
    df["load_rolling_168h_mean"] = df["load_MW"].rolling(window=168, min_periods=1).mean()  # weekly

    print(f"Missing ratio before: {missing_before:.4f}, after: {missing_after:.4f}")
    return df


def save_processed(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    print(f"Processed data saved to {path} with shape {df.shape}")


def run_preprocessing():
    df_raw = load_raw_opsd()
    df_processed = clean_and_engineer_features(df_raw)
    save_processed(df_processed)


if __name__ == "__main__":
    run_preprocessing()
