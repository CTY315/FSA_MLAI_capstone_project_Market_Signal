"""Model training, evaluation, and walk-forward validation."""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss,
    classification_report,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from statsmodels.stats.contingency_tables import mcnemar
from scipy import stats


# Default hyperparameter search space
PARAM_GRID = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 200, 500],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "scale_pos_weight": [0.8, 1.0, 1.2, 1.5],
    "min_child_weight": [1, 3, 5, 10],
    "gamma": [0, 0.1, 0.3],
    "reg_alpha": [0, 0.1, 1.0],
}


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 50,
    random_state: int = 42,
) -> xgb.XGBClassifier:
    """Train XGBoost with randomized hyperparameter search using temporal CV.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        n_iter: Number of random search iterations.
        random_state: Random seed.
    
    Returns:
        Best fitted XGBClassifier.
    """
    tscv = TimeSeriesSplit(n_splits=5)

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        use_label_encoder=False,
    )

    search = RandomizedSearchCV(
        model,
        PARAM_GRID,
        n_iter=n_iter,
        cv=tscv,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=random_state,
        verbose=1,
    )

    search.fit(X_train, y_train)

    print(f"Best params: {search.best_params_}")
    print(f"Best CV AUC: {search.best_score_:.4f}")

    return search.best_estimator_


def find_best_threshold(
    model: xgb.XGBClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    thresholds: list[float] | None = None,
) -> float:
    """Find the decision threshold that maximises Youden's J on the validation set.

    Youden's J = sensitivity + specificity - 1 = TPR - FPR. Optimising it
    balances the true-positive and true-negative rates without being thrown
    off by class imbalance, making it a better criterion than raw accuracy
    when the model is biased toward one class.

    Args:
        model: Trained classifier.
        X_val: Validation features (must match model feature names).
        y_val: Validation labels.
        thresholds: Candidate thresholds to search. Defaults to 0.30–0.70 in
            steps of 0.01.

    Returns:
        Best threshold (float between 0 and 1).
    """
    if thresholds is None:
        thresholds = np.arange(0.30, 0.71, 0.01).tolist()

    feat_names = model.get_booster().feature_names
    y_prob = model.predict_proba(X_val[feat_names])[:, 1]
    y_true = y_val.values

    best_thresh, best_j = 0.5, -1.0
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        tn = np.sum((pred == 0) & (y_true == 0))
        fp = np.sum((pred == 1) & (y_true == 0))
        fn = np.sum((pred == 0) & (y_true == 1))
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = sensitivity + specificity - 1.0
        if j > best_j:
            best_j, best_thresh = j, t

    print(f"  Best threshold: {best_thresh:.2f}  (Youden's J = {best_j:.4f})")
    return float(best_thresh)


def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
) -> dict:
    """Evaluate a trained model on test data.
    
    Args:
        model: Trained classifier.
        X_test: Test features.
        y_test: Test labels.
        model_name: Label for printing.
    
    Returns:
        Dictionary of metric names to values.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "log_loss": log_loss(y_test, y_prob),
        "f1_up": report["1"]["f1-score"],
        "f1_down": report["0"]["f1-score"],
    }

    print(f"\n=== {model_name} Test Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return metrics


def mcnemar_test(
    y_test: pd.Series,
    pred_baseline: np.ndarray,
    pred_enhanced: np.ndarray,
) -> dict:
    """Run McNemar's test comparing two models' predictions.
    
    Args:
        y_test: True labels.
        pred_baseline: Baseline model predictions.
        pred_enhanced: Enhanced model predictions.
    
    Returns:
        Dictionary with statistic and p-value.
    """
    bl_correct = (pred_baseline == y_test.values)
    en_correct = (pred_enhanced == y_test.values)

    table = np.array([
        [(bl_correct & en_correct).sum(), (bl_correct & ~en_correct).sum()],
        [(~bl_correct & en_correct).sum(), (~bl_correct & ~en_correct).sum()],
    ])

    result = mcnemar(table, exact=True)

    print(f"\nMcNemar's Test:")
    print(f"  Statistic: {result.statistic:.4f}")
    print(f"  p-value: {result.pvalue:.4f}")
    print(f"  Significant (p<0.05): {result.pvalue < 0.05}")

    return {"statistic": result.statistic, "pvalue": result.pvalue}


def walk_forward_validation(
    df: pd.DataFrame,
    tech_cols: list[str],
    all_cols: list[str],
    min_train_years: int = 3,
) -> pd.DataFrame:
    """Expanding-window walk-forward validation.
    
    Trains fresh models for each month using all data up to that point,
    then tests on the next month.
    
    Args:
        df: Full feature DataFrame with DatetimeIndex and 'target' column.
        tech_cols: Baseline feature column names.
        all_cols: Enhanced model feature column names.
        min_train_years: Minimum training window before starting evaluation.
    
    Returns:
        DataFrame with monthly AUC results for both models.
    """
    results = []
    df = df.copy()
    df["year_month"] = df.index.to_period("M")
    months = df["year_month"].unique()

    min_train_end = df.index.min() + pd.DateOffset(years=min_train_years)

    for month in months:
        month_start = month.start_time
        month_end = month.end_time

        if month_start < min_train_end:
            continue

        train_data = df[df.index < month_start]
        test_data = df[(df.index >= month_start) & (df.index <= month_end)]

        if len(test_data) < 5:
            continue

        try:
            # Baseline
            bl = xgb.XGBClassifier(
                random_state=42, use_label_encoder=False, eval_metric="logloss"
            )
            bl.fit(train_data[tech_cols], train_data["target"])
            bl_auc = roc_auc_score(
                test_data["target"],
                bl.predict_proba(test_data[tech_cols])[:, 1],
            )

            # Enhanced
            en = xgb.XGBClassifier(
                random_state=42, use_label_encoder=False, eval_metric="logloss"
            )
            en.fit(train_data[all_cols], train_data["target"])
            en_auc = roc_auc_score(
                test_data["target"],
                en.predict_proba(test_data[all_cols])[:, 1],
            )

            results.append({
                "month": str(month),
                "baseline_auc": bl_auc,
                "enhanced_auc": en_auc,
                "diff": en_auc - bl_auc,
                "n_test": len(test_data),
            })
        except Exception as e:
            print(f"  Skipping {month}: {e}")
            continue

    wf_df = pd.DataFrame(results)

    # Paired t-test
    if len(wf_df) > 1:
        t_stat, p_value = stats.ttest_rel(
            wf_df["enhanced_auc"], wf_df["baseline_auc"]
        )
        print(f"\nWalk-Forward Summary ({len(wf_df)} months):")
        print(f"  Mean Baseline AUC: {wf_df['baseline_auc'].mean():.4f}")
        print(f"  Mean Enhanced AUC: {wf_df['enhanced_auc'].mean():.4f}")
        print(f"  Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")

    return wf_df
