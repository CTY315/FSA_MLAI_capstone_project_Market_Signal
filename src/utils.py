"""Shared constants, data loading utilities, and helper functions."""

import pandas as pd
from pathlib import Path

# === Project Paths ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# === Temporal Split Dates ===
TRAIN_END = "2021-01-01"       # Train: everything before this
VAL_END = "2022-07-01"         # Val: TRAIN_END to this; Test: VAL_END onward (~18 months)
# Test: everything from VAL_END onward

# === Data Parameters ===
DATA_START = "2008-01-01"
DATA_END = "2023-12-31"        # Kaggle headlines end 2024-03-04; cut all data here
SPY_TICKER = "SPY"
VIX_TICKER = "^VIX"

# === Model Parameters ===
PCA_VARIANCE_THRESHOLD = 0.70  # Keep 70% of variance
NEWS_SHOCK_WINDOW = 20         # Rolling window for z-scores
NEWS_SHOCK_THRESHOLD = 2.0     # |z| > 2 = shock

# === Embedding Model ===
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def get_temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame by temporal boundaries. Index must be DatetimeIndex."""
    train = df[df.index < TRAIN_END]
    val = df[(df.index >= TRAIN_END) & (df.index < VAL_END)]
    test = df[df.index >= VAL_END]
    return train, val, test


def load_spy_data() -> pd.DataFrame:
    """Load SPY daily data from raw CSV."""
    path = DATA_RAW / "spy_daily.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def load_vix_data() -> pd.DataFrame:
    """Load VIX daily data from raw CSV."""
    path = DATA_RAW / "vix_daily.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def load_headlines() -> pd.DataFrame:
    """Load Kaggle headlines CSV. Normalizes column names to lowercase."""
    path = DATA_RAW / "sp500_headlines.csv"
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    return df
