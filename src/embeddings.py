"""Embedding generation and daily aggregation pipeline.

NOTE: The heavy embedding generation (Step 1) should be run on Google Colab
with GPU. This module provides the aggregation and merging logic for local use.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .utils import EMBEDDING_DIM, PCA_VARIANCE_THRESHOLD


def aggregate_daily_embeddings(headlines_df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """Aggregate headline embeddings by day using mean pooling.
    
    Args:
        headlines_df: DataFrame with 'date' and 'headline' columns.
        embeddings: Numpy array of shape (n_headlines, embedding_dim).
    
    Returns:
        DataFrame with one row per date: mean embedding, headline_count, embedding_magnitude.
    """
    headlines_df = headlines_df.copy()
    headlines_df["embedding"] = list(embeddings)

    def _agg(group: pd.DataFrame) -> pd.Series:
        emb_matrix = np.stack(group["embedding"].values)
        mean_emb = emb_matrix.mean(axis=0)
        return pd.Series({
            "mean_embedding": mean_emb,
            "headline_count": len(group),
            "embedding_magnitude": np.linalg.norm(mean_emb),
        })

    daily = headlines_df.groupby("date").apply(_agg).reset_index()
    return daily


def expand_embeddings_to_columns(daily_df: pd.DataFrame, dim: int = EMBEDDING_DIM) -> pd.DataFrame:
    """Expand mean_embedding arrays into separate columns.
    
    Args:
        daily_df: DataFrame with 'mean_embedding' column containing arrays.
        dim: Embedding dimension (default 384 for all-MiniLM-L6-v2).
    
    Returns:
        DataFrame with emb_0 through emb_{dim-1} columns plus metadata.
    """
    emb_df = pd.DataFrame(
        daily_df["mean_embedding"].tolist(),
        columns=[f"emb_{i}" for i in range(dim)],
        index=daily_df.index,
    )

    result = pd.concat([
        daily_df[["date", "headline_count", "embedding_magnitude"]],
        emb_df,
    ], axis=1)

    return result


def fit_pca_on_train(
    merged_df: pd.DataFrame,
    train_mask: pd.Series,
    variance_threshold: float = PCA_VARIANCE_THRESHOLD,
) -> tuple[PCA, pd.DataFrame]:
    """Fit PCA on training embedding columns only, transform all data.
    
    Args:
        merged_df: Full DataFrame with emb_0..emb_N columns.
        train_mask: Boolean Series indicating training rows.
        variance_threshold: Cumulative variance to retain (default 0.70).
    
    Returns:
        Tuple of (fitted PCA object, DataFrame with PCA columns replacing raw embeddings).
    """
    emb_cols = [c for c in merged_df.columns if c.startswith("emb_")]

    pca = PCA(n_components=variance_threshold)
    pca.fit(merged_df.loc[train_mask, emb_cols])

    n_components = pca.n_components_
    print(f"PCA: {n_components} components retain {pca.explained_variance_ratio_.sum():.3f} variance")

    # Transform all data
    pca_features = pca.transform(merged_df[emb_cols])
    pca_df = pd.DataFrame(
        pca_features,
        columns=[f"news_pca_{i}" for i in range(n_components)],
        index=merged_df.index,
    )

    # Replace raw embeddings with PCA
    result = merged_df.drop(columns=emb_cols).join(pca_df)

    return pca, result
