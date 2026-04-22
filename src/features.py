"""Feature engineering functions for technical indicators and news shock features."""

import numpy as np
import pandas as pd
from .utils import NEWS_SHOCK_WINDOW, NEWS_SHOCK_THRESHOLD


def compute_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log return features at multiple horizons.
    
    Args:
        df: DataFrame with 'Close' column.
    
    Returns:
        DataFrame with return_1d, return_5d, return_10d columns.
    """
    close = df["Close"]
    features = pd.DataFrame(index=df.index)

    for period in [1, 5, 10]:
        features[f"return_{period}d"] = np.log(close / close.shift(period))

    return features


def compute_ma_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute price-to-moving-average ratio features.
    
    Args:
        df: DataFrame with 'Close' column.
    
    Returns:
        DataFrame with price_to_maX columns.
    """
    close = df["Close"]
    features = pd.DataFrame(index=df.index)

    for window in [10, 20, 50, 100, 200]:
        ma = close.rolling(window).mean()
        features[f"price_to_ma{window}"] = close / ma - 1

    return features


def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling volatility measures.
    
    Args:
        df: DataFrame with 'Close' column.
    
    Returns:
        DataFrame with volatility_Xd columns.
    """
    returns = np.log(df["Close"] / df["Close"].shift(1))
    features = pd.DataFrame(index=df.index)

    for window in [5, 10, 20]:
        features[f"volatility_{window}d"] = returns.rolling(window).std()

    return features


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute volume change and relative volume features.
    
    Args:
        df: DataFrame with 'Volume' column.
    
    Returns:
        DataFrame with volume_change and volume_to_avg20 columns.
    """
    vol = df["Volume"]
    features = pd.DataFrame(index=df.index)

    features["volume_change"] = vol.pct_change()
    features["volume_to_avg20"] = vol / vol.rolling(20).mean()

    return features


def compute_vix_features(vix_df: pd.DataFrame) -> pd.DataFrame:
    """Compute VIX-based features.
    
    Args:
        vix_df: VIX DataFrame with 'Close' column.
    
    Returns:
        DataFrame with vix_level, vix_change_1d, vix_change_5d, vix_to_ma20 columns.
    """
    close = vix_df["Close"]
    features = pd.DataFrame(index=vix_df.index)

    features["vix_level"] = close
    features["vix_change_1d"] = close.pct_change(1)
    features["vix_change_5d"] = close.pct_change(5)
    features["vix_to_ma20"] = close / close.rolling(20).mean() - 1

    return features


def compute_all_technical_features(spy_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """Build complete technical feature matrix from SPY and VIX data.
    
    Args:
        spy_df: SPY daily OHLCV data.
        vix_df: VIX daily data.
    
    Returns:
        Combined DataFrame with all technical features and target variable.
    """
    spy_features = pd.concat([
        compute_return_features(spy_df),
        compute_ma_features(spy_df),
        compute_volatility_features(spy_df),
        compute_volume_features(spy_df),
    ], axis=1)

    vix_features = compute_vix_features(vix_df)

    # Merge on date index
    tech_features = spy_features.join(vix_features, how="inner")

    # Target: next-day direction (1 = up, 0 = down)
    tech_features["target"] = (
        spy_df["Close"].pct_change().shift(-1) > 0
    ).astype(int)

    # Drop rows with NaN from rolling calculations
    tech_features = tech_features.dropna()

    return tech_features


def compute_news_shock_features(
    df: pd.DataFrame,
    window: int = NEWS_SHOCK_WINDOW,
    threshold: float = NEWS_SHOCK_THRESHOLD,
) -> pd.DataFrame:
    """Compute news shock features using rolling z-scores.
    
    Args:
        df: DataFrame with 'headline_count' and 'embedding_magnitude' columns.
        window: Rolling window size for z-score calculation.
        threshold: Absolute z-score threshold for flagging shocks.
    
    Returns:
        DataFrame with z-score and binary shock columns.
    """
    features = pd.DataFrame(index=df.index)

    for col, name in [("headline_count", "volume"), ("embedding_magnitude", "magnitude")]:
        rolling_mean = df[col].rolling(window, min_periods=5).mean()
        rolling_std = df[col].rolling(window, min_periods=5).std()

        z_score = (df[col] - rolling_mean) / (rolling_std + 1e-8)
        features[f"news_{name}_zscore"] = z_score
        features[f"news_{name}_shock"] = (z_score.abs() > threshold).astype(int)

    return features
