from pathlib import Path

import pandas as pd

from src.data.preprocess_opsd import run_preprocessing, PROCESSED_PATH


def test_preprocessing_creates_file():
    # Run pipeline
    run_preprocessing()

    assert PROCESSED_PATH.exists(), "Processed file was not created"

    df = pd.read_csv(PROCESSED_PATH, parse_dates=[0], index_col=0)

    # Basic sanity checks
    assert len(df) > 1000, "Too few rows after preprocessing"
    assert "load_MW" in df.columns, "Target column missing"
    assert df["load_MW"].isna().mean() == 0.0, "Missing values remain in target"
    for col in ["hour", "day_of_week", "month", "is_weekend"]:
        assert col in df.columns, f"Missing time feature: {col}"
