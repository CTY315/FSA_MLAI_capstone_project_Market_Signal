# AI Market Signal: Multimodal Market Prediction Using Transformer News Embeddings

**Author:** Shirley Cheung  
**Program:** FullStack Academy AI/ML — Cohort 2510-FTB-CT-AIM-PT  
**Instructor:** Dr. George Perdrizet  
**Teaching Assistant:** Andrew Thomas  
**Date:** April 2026

---

## Abstract

This project investigates whether transformer-derived financial news embeddings can improve next-day SPY (S&P 500 ETF) direction prediction over a purely technical baseline. Daily S&P 500 financial headlines (2008–2023) were encoded using the `all-MiniLM-L6-v2` sentence transformer model, mean-pooled into daily vectors, and compressed to 79 principal components via PCA fitted on training data only. An XGBoost classifier trained on these embeddings alongside 17 technical indicators achieved a test-set AUC of 0.561 and accuracy of 54.9%, compared to 0.496 AUC and 47.8% accuracy for the technical-only baseline. McNemar's test (p = 0.035) confirmed the models make significantly different prediction errors. However, walk-forward validation across 146 monthly windows showed the enhanced model's mean AUC of 0.491 was lower than the baseline's 0.507, with a paired t-test p-value of 0.256 — indicating the improvement does not hold consistently across market regimes. The project includes a five-tab Streamlit dashboard and a RAG-powered market assistant built with ChromaDB and Claude.

---

## 1. Introduction

### 1.1 Problem Statement

Predicting the direction of equity market returns is one of the most studied problems in quantitative finance. Despite decades of research, financial markets remain largely efficient — prices incorporate publicly available information quickly, leaving little systematic edge for prediction models. Yet the efficient market hypothesis does not preclude all predictability: structural behavioral patterns, market microstructure dynamics, and time-varying risk premia have been shown to create weak, transient predictable components in short-horizon returns.

The question this project asks is specific: **does the textual content of daily financial news headlines contain information about next-day SPY direction that is not already captured by technical price and volatility indicators?**

### 1.2 Motivation

Two developments in recent years make this question newly tractable:

1. **Transformer language models** have dramatically improved the quality of text representations. Models like `all-MiniLM-L6-v2` produce dense 384-dimensional embeddings that capture semantic relationships between financial concepts — relationships that term-frequency methods (TF-IDF, bag-of-words) miss entirely.

2. **Large public datasets** of labeled financial headlines now cover over a decade of market history, providing enough data to train and evaluate models with proper temporal holdout sets.

Prior work on news-based prediction has relied heavily on sentiment scoring (positive/negative/neutral labels), which discards most of the semantic content of headlines. This project takes a different approach: use the full embedding vector as input features, letting the model discover which semantic dimensions of news are predictive.

### 1.3 Research Question

**Primary:** Does adding transformer news embeddings to technical indicators improve next-day SPY direction prediction on a held-out 2022–2023 test set?

**Secondary:** Are any improvements statistically significant, and do they hold consistently across different market regimes?

---

## 2. Data

### 2.1 SPY Daily Price Data

- **Source:** Yahoo Finance via `yfinance` API
- **Columns:** Open, High, Low, Close (adjusted), Volume
- **Raw range:** 2008-01-02 → 2024-12-30 (4,278 trading days)
- **After project cutoff:** 2008-01-02 → 2023-12-29 (4,027 rows)
- **Missing values:** None

The SPY adjusted close price rose from approximately $103 in January 2008 to $475 by end of 2023, spanning the 2008–2009 financial crisis, the 2020 COVID crash, and the 2022 Federal Reserve rate-hiking cycle — three qualitatively different market regimes.

**Target variable:** Binary next-day direction derived from `sign(Close[t+1] - Close[t])`. The full dataset is approximately balanced: 55.0% Up days, 45.0% Down days.

### 2.2 VIX Daily Data

- **Source:** Yahoo Finance via `yfinance` API (`^VIX`)
- **Columns:** Open, High, Low, Close, Volume (Volume is always 0 — VIX is an index, not a traded asset)
- **Range:** 2008-01-02 → 2023-12-29 (4,027 rows, aligned with SPY)
- **Missing values:** None

VIX (the CBOE Volatility Index) measures the market's expectation of 30-day implied volatility derived from S&P 500 options. It is a direct measure of market fear and tends to spike during crashes — making it a natural input feature for market direction prediction.

### 2.3 Financial Headlines Dataset

- **Source:** [Kaggle — S&P 500 with Financial News Headlines 2008–2024](https://www.kaggle.com/datasets/dyutidasmahaptra/s-and-p-500-with-financial-news-headlines20082024/data)
- **Columns:** `title` (headline text), `date`, `cp` (S&P 500 close at that date)
- **Raw size:** 19,127 rows (2008-01-02 → 2024-03-04)

**Data quality issues and decisions:**

1. **Coverage gap at end of 2024:** The dataset's headline coverage drops sharply after March 2024. To ensure consistent coverage across the full test window, all data was cut off at `2023-12-31`, reducing the dataset to 18,294 rows.

2. **Deduplication:** Same-day duplicate headlines were removed, reducing the count to **17,326 unique headline-date pairs**.

3. **Trading day coverage:** Headline coverage is not uniform over time. Of the 4,174 trading days in the 2008–2024 window, only 3,464 (83.0%) have at least one headline — leaving **710 trading days with no news coverage**. Coverage improved significantly over time:

| Split | Trading Days | Days with Headlines | Coverage |
|-------|-------------|---------------------|----------|
| Train (2008–2020) | 3,394 | 2,719 | 80.1% |
| Val (2021–2022) | 391 | 374 | 95.7% |
| Test (2022–2023) | 391 | 371 | 94.9% |

The average number of headlines per covered day also increased over time: 3.6/day in training, 6.6/day in validation, and 15.9/day in the test period — reflecting growth in financial media coverage.

4. **Zero-vector imputation:** Rather than dropping no-news days (which would exclude years of valid price history) or mean-imputing (which would inject spurious sentiment), days with no headlines received a zero embedding vector. A binary `has_news` flag was added so the model can learn to discount these days.

### 2.4 Temporal Split

Given the time-series nature of the prediction task, strict temporal splits were used throughout. No random shuffling was applied at any stage.

| Split | Date Range | Rows | Purpose |
|-------|-----------|------|---------|
| **Train** | 2008-10-15 → 2020-12-31 | 3,075 | Model training and hyperparameter search |
| **Val** | 2021-01-04 → 2022-06-30 | 376 | Threshold tuning; model selection |
| **Test** | 2022-07-01 → 2023-12-29 | 377 | Final one-shot evaluation only |

The training set starts on 2008-10-15 (not 2008-01-02) because the 200-day moving average requires 199 prior trading days of warmup before producing a valid value.

---

## 3. Methodology

### 3.1 Feature Engineering: Technical Indicators

Seventeen technical features were computed from SPY and VIX daily data. These features were chosen because they capture the key dimensions of market state that quantitative practitioners use: recent momentum, trend context (price relative to moving averages), volatility regime, volume conditions, and fear/greed (VIX).

**Return features (3):** Log returns at 1, 5, and 10-day horizons capture short-term momentum at different time scales. Log returns are preferred over percentage returns for their additive property and better statistical behavior.

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| `return_1d` | log(Close[t] / Close[t-1]) | Overnight momentum |
| `return_5d` | log(Close[t] / Close[t-5]) | Weekly momentum |
| `return_10d` | log(Close[t] / Close[t-10]) | Bi-weekly momentum |

**Moving average ratio features (5):** Price divided by its N-day moving average (minus 1), measuring how far price has deviated from its long-run trend. These are mean-reversion signals — large deviations may predict reversals.

| Feature | Window | Interpretation |
|---------|--------|----------------|
| `price_to_ma10` | 10-day | Ultra-short-term trend context |
| `price_to_ma20` | 20-day | Short-term trend context |
| `price_to_ma50` | 50-day | Medium-term trend context |
| `price_to_ma100` | 100-day | Intermediate trend context |
| `price_to_ma200` | 200-day | Long-term trend context (the "golden cross") |

**Volatility features (3):** Rolling standard deviation of log returns, measuring the speed of price movement. High volatility often precedes further volatility (volatility clustering) or regime shifts.

| Feature | Window | Interpretation |
|---------|--------|----------------|
| `volatility_5d` | 5-day | Short-term realized volatility |
| `volatility_10d` | 10-day | Medium-term realized volatility |
| `volatility_20d` | 20-day | Month-scale realized volatility |

**Volume features (2):**

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| `volume_change` | (Vol[t] - Vol[t-1]) / Vol[t-1] | Day-over-day volume change |
| `volume_to_avg20` | Vol[t] / mean(Vol[t-20:t]) | Relative volume vs. recent average |

**VIX features (4):**

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| `vix_level` | VIX close | Absolute fear level |
| `vix_change_1d` | (VIX[t] - VIX[t-1]) / VIX[t-1] | Day-over-day change in fear |
| `vix_change_5d` | (VIX[t] - VIX[t-5]) / VIX[t-5] | Weekly change in fear |
| `vix_to_ma20` | VIX[t] / MA20(VIX) - 1 | VIX vs. its own rolling average |

These features were chosen over traditional indicators like RSI and MACD because they are more interpretable, lower in parameter count, and directly correspond to well-understood market quantities. RSI and MACD are transformations of price and moving averages that add no new information beyond what the price-to-MA ratios and return features already capture.

### 3.2 Transformer Embeddings

**Model:** `sentence-transformers/all-MiniLM-L6-v2`  
**Embedding dimension:** 384  
**Normalization:** L2-normalized output vectors

`all-MiniLM-L6-v2` was selected for three reasons:

1. **Speed:** At 6 layers and 22.7M parameters, it encodes ~17,000 headlines in under 7 seconds on a T4 GPU — practical for an educational project.
2. **Quality:** Despite its small size, it consistently ranks in the top tier of the MTEB (Massive Text Embedding Benchmark) for semantic similarity tasks.
3. **Domain transfer:** Its multilingual BERT pre-training and sentence-pair fine-tuning generalize reasonably to financial text without domain-specific fine-tuning.

**Encoding process:** All 17,326 cleaned headlines were encoded in a single batch of size 256 on a Google Colab T4 GPU. Total encoding time was 6.15 seconds (wall clock). Outputs were L2-normalized, meaning each embedding vector has unit length. This normalization ensures that cosine similarity equals dot product — a useful property for downstream PCA.

**Computational note:** Embedding generation is the only GPU-dependent step. All other pipeline stages (PCA, model training, evaluation, dashboard) run on CPU.

### 3.3 Daily Aggregation: Mean Pooling

Multiple headlines per trading day are aggregated into a single daily embedding vector via **mean pooling** (element-wise average of all headline embeddings for that day).

**Rationale for mean pooling over alternatives:**
- *Attention-weighted pooling* requires supervised training of the attention weights — which would introduce a risk of leakage if trained on the same target variable.
- *Max pooling* amplifies the most extreme embedding dimension, which may not reflect the dominant news theme of the day.
- *Mean pooling* is simple, deterministic, and equivalent to computing the centroid of that day's news in embedding space — a reasonable proxy for average daily news sentiment/topic.

Along with the mean embedding, two metadata features are recorded per day: `headline_count` (how many headlines) and `embedding_magnitude` (L2 norm of the mean vector — measures how much the day's news aligns in one direction).

### 3.4 PCA Dimensionality Reduction

Raw 384-dimensional embedding vectors would dominate the feature matrix numerically and cause severe overfitting. PCA was applied to compress the embedding space while retaining the most predictive variance structure.

**Variance threshold:** 70% of training-set embedding variance (`PCA_VARIANCE_THRESHOLD = 0.70`)  
**Result:** 79 principal components  
**Leakage prevention:** PCA was fit **exclusively on the 3,075 training rows** (dates before 2021-01-01). The same fitted PCA object was then used to transform validation and test rows. This prevents information from future dates from influencing the PCA basis vectors.

The 70% threshold was chosen as a balance between information retention and overfitting risk. At 70%, 79 components are retained from the original 384 — a 79% reduction in dimensionality. Higher thresholds (e.g., 90%) would retain ~150 components, increasing overfitting risk with a training set of 3,075 rows.

### 3.5 News Shock Features

Beyond the compressed embeddings, four interpretable news-shock features were engineered to capture anomalous news days:

**Volume z-score** (`news_volume_zscore`): A rolling 20-day z-score of the headline count:
```
z_vol(t) = (count(t) - mean(count[t-20:t])) / std(count[t-20:t])
```

**Magnitude z-score** (`news_magnitude_zscore`): Rolling 20-day z-score of the embedding magnitude, measuring whether today's news is unusually directionally coherent:
```
z_mag(t) = (magnitude(t) - mean(magnitude[t-20:t])) / std(magnitude[t-20:t])
```

**Binary shock flags:** `news_volume_shock = 1` when `|z_vol| > 2`; `news_magnitude_shock = 1` when `|z_mag| > 2`.

These z-scores are computed on the zero-imputed series — meaning no-news days with embedding magnitude of 0 pull the rolling mean down, making genuine high-magnitude news days stand out more strongly. This is intentional: the signal being captured is "today's news is unusually extreme relative to the recent norm."

**Important caveat:** Zero-imputation means that extreme silent periods (many consecutive no-news days) can trigger shock flags as the rolling mean adapts. This is an acknowledged limitation discussed in Section 9.

### 3.6 Model Selection

**XGBoost** was chosen as the primary classifier for several reasons:

- **Gradient-boosted trees handle tabular data well:** Financial features have irregular distributions, non-linear interactions (e.g., VIX-level × momentum), and mixed scales. Tree-based models handle these natively without normalization.
- **Built-in regularization:** Parameters like `gamma`, `min_child_weight`, `reg_alpha`, and `subsample` directly control overfitting — important given the low signal-to-noise ratio in financial data.
- **Feature importance:** XGBoost provides interpretable gain-based importance scores, allowing post-hoc analysis of which features the model actually uses.
- **Proven track record:** XGBoost consistently dominates Kaggle competitions on tabular datasets.

**Logistic Regression** was included as a linear comparison baseline to separate feature quality from model capacity. A significant gap between LR and XGBoost on the same feature set would indicate the signal requires non-linear modeling. A small gap would indicate the signal ceiling is in the features.

### 3.7 Validation Strategy

**Why temporal splits only:** Standard random cross-validation would create data leakage in time-series prediction. If a test fold includes data from, say, March 2020, and the training fold includes April 2020 data, the model is trained on the "future" relative to what it is being tested on. Temporal splits ensure training data always precedes validation and test data.

**Expanding-window walk-forward validation:** In addition to the fixed temporal holdout, a walk-forward validation was run over 146 monthly windows starting from November 2011 (3 years of warmup from the training start). For each month, a fresh XGBoost model is trained on all data up to that month and evaluated on the following month. This tests whether the model performs consistently across different market regimes, not just in a single held-out period.

### 3.8 Hyperparameter Tuning

**Search method:** RandomizedSearchCV with `n_iter=50` (50 random parameter combinations), 5-fold `TimeSeriesSplit` cross-validation, scored by ROC-AUC.

**Parameter grid:**

| Parameter | Search Space | Role |
|-----------|-------------|------|
| `max_depth` | [3, 5, 7] | Tree depth; controls capacity |
| `learning_rate` | [0.01, 0.05, 0.1] | Step size; smaller = more regularized |
| `n_estimators` | [100, 200, 500] | Number of trees |
| `subsample` | [0.7, 0.8, 1.0] | Row subsampling per tree |
| `colsample_bytree` | [0.7, 0.8, 1.0] | Feature subsampling per tree |
| `scale_pos_weight` | [0.8, 1.0, 1.2, 1.5] | Class imbalance correction |
| `min_child_weight` | [1, 3, 5, 10] | Minimum leaf node weight (regularization) |
| `gamma` | [0, 0.1, 0.3] | Minimum gain to split a node (regularization) |
| `reg_alpha` | [0, 0.1, 1.0] | L1 regularization term |

**Best parameters (both models):**

| Parameter | Value |
|-----------|-------|
| `colsample_bytree` | 0.7 |
| `gamma` | 0.1 |
| `learning_rate` | 0.05 |
| `max_depth` | 7 |
| `min_child_weight` | 5 |
| `n_estimators` | 100 |
| `reg_alpha` | 0 |
| `scale_pos_weight` | 0.8 |
| `subsample` | 0.8 |

**Decision threshold tuning:** Both models default to a 0.50 probability threshold for classification. Because the training set has ~55% Up days, the raw predicted probabilities are skewed upward — both models tend to predict "Up" too often. Thresholds were tuned on the **validation set only** (never test) by maximizing Youden's J statistic:

```
J = Sensitivity + Specificity - 1 = TPR - FPR
```

This criterion balances true-positive and true-negative rates without being thrown off by class imbalance. Tuned thresholds:
- **Baseline:** 0.59 (raises the bar for predicting Up)
- **Enhanced:** 0.55 (slightly less aggressive correction)

---

## 4. Results

### 4.1 Validation Set Results

The validation set (2021-01-04 → 2022-06-30, n = 376 trading days) was used for threshold tuning and model selection. The test set was kept completely sealed.

| Model | Threshold | Accuracy | AUC | Up % |
|-------|-----------|----------|-----|------|
| XGB — Tech only | 0.50 | 51.1% | 0.5064 | 62.8% |
| XGB — Tech + News | 0.50 | 50.3% | 0.4774 | 65.7% |
| LR — Tech only | 0.50 | 55.3% | 0.5487 | 97.9% |
| LR — Tech + News | 0.50 | 52.1% | 0.4964 | 84.0% |

Key observations on validation:
- LR outperforms XGBoost on the validation set (0.5487 vs 0.5064 AUC), suggesting XGBoost may be overfitting to training patterns from 2008–2020.
- Both LR models exhibit severe "Up" bias (84–98% Up predictions), confirming that un-tuned thresholds are unsuitable.
- Adding news features **hurts** on the validation set for both model types. This is partly attributable to the validation window covering COVID recovery (2021) and early rate-hike volatility (2022) — a regime not well represented in training data.

### 4.2 Test Set Results

The test set (2022-07-01 → 2023-12-29, n = 377 trading days) was evaluated exactly once with tuned thresholds.

| Model | Threshold | Accuracy | AUC | F1 (Up) | F1 (Down) | Up % |
|-------|-----------|----------|-----|---------|-----------|------|
| **XGB — Baseline (Tech only)** | 0.59 | 47.8% | 0.496 | 0.278 | 0.590 | 21.2% |
| **XGB — Enhanced (Tech+News)** | 0.55 | 54.9% | 0.561 | 0.509 | 0.583 | 40.6% |
| LR — Tech only | 0.50 | 50.1% | 0.457 | — | — | 95.8% |
| LR — Tech + News | 0.50 | 52.3% | 0.526 | — | — | 80.4% |

**Side-by-side delta (Enhanced vs Baseline, XGB):**

| Metric | Baseline | Enhanced | Δ |
|--------|----------|----------|---|
| Accuracy | 0.4775 | 0.5491 | +0.0716 |
| ROC-AUC | 0.4959 | 0.5609 | +0.0650 |
| Log Loss | 0.7174 | 0.6905 | −0.0269 |
| F1 — Up | 0.2784 | 0.5087 | +0.2303 |
| F1 — Down | 0.5904 | 0.5833 | −0.0071 |

The enhanced XGBoost model shows meaningful improvement on the test set across all metrics. Feature effect (news vs. tech, holding model type fixed) was +0.065 AUC for XGBoost and +0.069 AUC for LR. Model effect (XGBoost vs. LR, holding features fixed) was +0.039 AUC — XGBoost captures non-linear interactions that LR misses.

The baseline's 47.8% accuracy (worse than a coin flip) at threshold 0.59 reflects an over-correction: the aggressive threshold rarely predicts Up (only 21.2% of days), causing many missed Up days.

### 4.3 Statistical Significance

#### McNemar's Test

McNemar's test compares the prediction *disagreements* between the two XGBoost models on the same 377 test observations — specifically, the 63 days where the baseline was right and the enhanced was wrong (cell b), and the 90 days where the enhanced was right and the baseline was wrong (cell c).

**Contingency table:**

|  | Enhanced Correct | Enhanced Wrong |
|--|-----------------|----------------|
| **Baseline Correct** | 117 | 63 |
| **Baseline Wrong** | 90 | 107 |

- McNemar statistic (with Yates' continuity correction): **63.000**
- p-value: **0.0352**
- Conclusion: **Significant at α = 0.05.** The enhanced model makes statistically different errors than the baseline — its improvements are not consistent with random variation.

#### Walk-Forward Validation

Walk-forward validation tested both models across 146 monthly windows (2011-11 → 2023-12), each time training from scratch on all available data up to that month.

| Metric | Baseline | Enhanced |
|--------|----------|----------|
| Mean AUC | 0.5071 | 0.4909 |
| Std AUC | 0.1174 | 0.1425 |
| Min AUC | 0.2024 | 0.0833 |
| Max AUC | 0.8095 | 0.8462 |
| % windows enhanced wins | 52.1% | — |

**Paired t-test (enhanced vs. baseline AUC, 146 windows):**
- t-statistic: −1.1403
- p-value: 0.2560
- Conclusion: **Not significant at α = 0.05.** The enhanced model wins 52.1% of monthly windows, but this advantage is not statistically reliable.

The contradiction between the McNemar result (p = 0.035, significant) and the walk-forward result (p = 0.256, not significant) reflects an important nuance: McNemar asks whether the models disagree differently on a specific fixed test window; walk-forward asks whether the advantage is consistent across time. The walk-forward result is the more stringent test.

#### Bootstrap Confidence Intervals (n = 1,000)

Non-parametric bootstrap CIs provide uncertainty bounds on the test-set point estimates.

| Model | Accuracy (mean) | Accuracy 95% CI | AUC (mean) | AUC 95% CI |
|-------|----------------|-----------------|------------|------------|
| XGB Baseline | 0.4783 | [0.4297, 0.5252] | 0.4962 | [0.4375, 0.5605] |
| XGB Enhanced | 0.5490 | [0.4987, 0.6021] | 0.5612 | [0.5042, 0.6210] |

The 95% AUC CIs do not overlap (Baseline upper: 0.5605, Enhanced lower: 0.5042 — overlapping by 0.056 units). While the intervals technically overlap slightly, the enhanced model's lower CI bound of 0.504 is above the random baseline of 0.50, providing moderate confidence that the enhanced model has some genuine predictive ability.

### 4.4 Feature Importance Analysis

XGBoost feature importance was measured by **gain** (the total improvement in the loss function attributable to a feature across all splits in all trees).

**Baseline model — Top 15 features by gain:**

| Rank | Feature | Gain |
|------|---------|------|
| 1 | `vix_level` | 3.790 |
| 2 | `price_to_ma200` | 3.730 |
| 3 | `volume_to_avg20` | 3.704 |
| 4 | `price_to_ma100` | 3.682 |
| 5 | `price_to_ma50` | 3.677 |
| 6 | `vix_change_5d` | 3.666 |
| 7 | `price_to_ma10` | 3.661 |
| 8 | `vix_change_1d` | 3.640 |
| 9 | `price_to_ma20` | 3.599 |
| 10 | `volatility_5d` | 3.584 |
| 11 | `volatility_10d` | 3.565 |
| 12 | `return_5d` | 3.533 |
| 13 | `return_10d` | 3.498 |
| 14 | `volume_change` | 3.465 |
| 15 | `volatility_20d` | 3.391 |

The baseline model uses all 17 technical features relatively evenly (gain range: 3.39–3.79), with VIX level, long-term MA ratios, and volume leading. Notably, all features contribute positive gain — none are discarded by XGBoost.

**Enhanced model — Top 15 features by gain:**

| Rank | Feature | Gain | Type |
|------|---------|------|------|
| 1 | `news_pca_7` | 6.433 | News PCA |
| 2 | `news_pca_40` | 6.122 | News PCA |
| 3 | `news_pca_2` | 5.987 | News PCA |
| 4 | `news_pca_26` | 5.951 | News PCA |
| 5 | `news_pca_12` | 5.799 | News PCA |
| 6 | `news_pca_35` | 5.772 | News PCA |
| 7 | `news_pca_45` | 5.706 | News PCA |
| 8 | `news_pca_31` | 5.692 | News PCA |
| 9 | `price_to_ma20` | 5.680 | Technical |
| 10 | `news_pca_66` | 5.673 | News PCA |
| 11 | `news_pca_54` | 5.671 | News PCA |
| 12 | `news_pca_74` | 5.650 | News PCA |
| 13 | `news_pca_62` | 5.635 | News PCA |
| 14 | `news_pca_77` | 5.618 | News PCA |
| 15 | `news_pca_57` | 5.612 | News PCA |

**14 of the top 15 enhanced features are news PCA components.** Only `price_to_ma20` (rank 9) breaks into the top 15 from the technical side. The model is genuinely leveraging news content — the PCA components that made it into the top 15 carry more predictive gain per split than any individual technical indicator. The gain values themselves are higher (5.6–6.4 vs. 3.4–3.8 for the baseline), indicating that individual news PCA components are more discriminative than any single technical indicator when they are informative.

---

## 5. Discussion

### 5.1 Did News Features Help?

**Short answer: Yes on the held-out test set, but not consistently across time.**

On the fixed test window (2022-07 to 2023-12), the enhanced model improved AUC by +0.065 (from 0.496 to 0.561), accuracy by +7.2 percentage points, and achieved statistically different prediction errors (McNemar p = 0.035). The bootstrap 95% CI for enhanced AUC of [0.504, 0.621] excludes 0.50, providing evidence of above-chance predictive ability.

However, the walk-forward validation across 146 monthly windows tells a more cautious story: the enhanced model wins only 52.1% of months, its mean walk-forward AUC of 0.491 is actually *below* the baseline's 0.507, and the paired t-test (p = 0.256) does not reach significance. The enhanced model has higher variance (std 0.143 vs. 0.117 for baseline), meaning it produces stronger signals in some regimes and weaker in others — a pattern consistent with news being a regime-dependent, rather than universally predictive, signal source.

### 5.2 Why Did Validation and Test Results Differ?

The validation period (2021-01 to 2022-06) and test period (2022-07 to 2023-12) represent two different market regimes:

- **Validation:** COVID-19 recovery (2021) followed by early Federal Reserve interest rate hikes and the initial market downturn (first half of 2022). News during this period was dominated by COVID vaccination rollout, supply chain disruptions, and early inflation signals — topics with no analog in the 2008–2020 training data.
- **Test:** Continued Fed rate hikes, inflation peak-and-decline narrative, and the beginning of AI sector excitement (late 2023). News patterns here overlap more with the financial-stress and recovery patterns in training data.

This regime mismatch explains why the enhanced model underperformed on validation but outperformed on test. Financial news embeddings appear to carry useful signal about "crisis-recovery" dynamics (well represented in training) but less signal about novel macroeconomic regime transitions like the COVID shock.

### 5.3 Why Did LR Beat XGB on Validation But Not Test?

On the validation set, LR achieved AUC 0.549 vs. XGBoost's 0.506 (tech features). This is a classic **bias-variance tradeoff** effect:

- XGBoost with `max_depth=7` and `n_estimators=100` has high capacity. On a 3,075-row training set, it can memorize specific patterns that do not generalize.
- LR has zero capacity to model non-linear interactions, so it underfits training data but generalizes better to the somewhat similar validation regime.
- On the longer test set, which covers a regime more similar to training, XGBoost's non-linear patterns re-emerge as predictive, closing the gap.

This is not an indictment of XGBoost — it is a reminder that financial time series exhibit non-stationarity, and a model's in-sample vs. out-of-sample performance depends heavily on whether the holdout period matches the training distribution.

### 5.4 The "Always Predict Up" Bias

Without threshold tuning, both XGBoost and LR predict "Up" on the majority of test days:
- XGB Baseline (t=0.50): ~84% Up predictions
- LR Tech only (t=0.50): ~96% Up predictions

This occurs because the training set has 55.7% Up days, causing the model to learn a prior toward Up predictions. The effect is amplified in LR by the StandardScaler pipeline — normalized inputs push probabilities toward the more common class.

Threshold tuning via Youden's J partially corrected this:
- Baseline at t=0.59: 21.2% Up (over-corrected)
- Enhanced at t=0.55: 40.6% Up (better balance)

The baseline's over-correction at t=0.59 contributed to its 47.8% accuracy — the model became too reluctant to predict Up. A more systematic threshold search or Platt scaling might find a better calibration point.

### 5.5 The Zero-Headline Shock Finding

An interesting artifact of the zero-imputation + rolling z-score design: days with zero headlines receive an embedding magnitude of 0, which can be dramatically lower than the rolling average. If several high-magnitude days precede a quiet period, the rolling mean magnitude stays high while no-news days have magnitude 0, triggering `news_magnitude_shock = 1` for silence.

This means the `news_magnitude_shock` flag captures both "the market had extremely coherent news" and "the market had no news" — semantically different events that the model cannot distinguish. This is an acknowledged design flaw that could be corrected by computing shock z-scores only on days with `has_news = 1`.

---

## 6. Dashboard & RAG System

### 6.1 Streamlit Dashboard

The project includes a five-tab Streamlit dashboard (`dashboard/app.py`) built with Plotly for interactive charts.

**Tab 1 — Headlines Browser:** Browse Kaggle headlines by trading date. Includes a random-date button, a collapsible list of all news shock days for quick navigation, shock warning banners, and a headline count metric card.

**Tab 2 — Daily Predictions:** SPY closing price chart with model predictions overlaid as dot markers (green = news shock days). Date range filter, summary metrics (total days, correct predictions, range accuracy), and a prediction table with green/red row coloring for correct/incorrect predictions.

**Tab 3 — Market Assistant:** RAG-powered Q&A over the 2008–2023 headline corpus. Includes four pre-built example question buttons, chat history, and source headline citations for each answer.

**Tab 4 — Model Comparison:** All four models (XGB baseline/enhanced + LR tech/news) compared via summary metrics table, confusion matrices, ROC curves, bootstrap AUC confidence intervals, and McNemar's test.

**Tab 5 — Performance Overview:** Top-level test-set metrics (accuracy, AUC, log loss for baseline and enhanced, with delta indicators), walk-forward AUC chart over time, and feature importance bar charts (top 15 by gain).

### 6.2 RAG Architecture

The Market Assistant tab implements Retrieval-Augmented Generation over the headline corpus.

**Components:**
- **Vector store:** ChromaDB (v1.5+) with ONNX-backed embeddings for fast similarity search
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (same model as the prediction pipeline — ensures embedding consistency)
- **Indexing:** One-time indexing via `scripts/index_headlines.py`; integer `date_int` metadata enables date-range filtering
- **Language model:** Anthropic Claude (configured for fast inference) via the Anthropic SDK
- **Chain:** Custom `_RAGChain` class (not `RetrievalQA`) that formats retrieved headlines into a structured context prompt

**Query flow:**
1. User question is embedded with `all-MiniLM-L6-v2`
2. ChromaDB performs approximate nearest-neighbor search over 17,326 headline embeddings
3. Top-k retrieved headlines (with dates) are formatted as context
4. Claude generates an answer grounded in the retrieved context
5. Answer and source citations are returned to the UI

**Why RAG over fine-tuning:** RAG provides verifiable citations (users can see which headlines the answer came from) and requires no GPU or model fine-tuning. It also generalizes naturally to new date ranges — re-indexing new headlines requires only running `index_headlines.py`.

---

## 7. Limitations

1. **Headline text only:** The model uses only the headline string, not the full article body. A headline like "Fed raises rates" carries far less information than the full article containing the magnitude of the increase, forward guidance language, and dot plot revisions.

2. **Daily granularity:** News sentiment can change multiple times per day. Intraday news events (e.g., a Fed announcement mid-session) are averaged with morning headlines, diluting the signal. Predicting next-day closes from same-day headlines also conflates market-hours and after-hours news effects.

3. **Single asset:** The model predicts only SPY direction. News about individual sectors (technology, energy, financials) may predict sector ETF returns more cleanly than the aggregate index, where news effects are diluted.

4. **Coverage gaps:** 710 trading days (17% of the training set) have no headlines. Zero-imputation is a pragmatic solution but assumes that no-news days are informationally equivalent to low-news days — which may not hold.

5. **Low signal-to-noise ratio:** Financial markets are information-efficient at short horizons. Even with a sophisticated feature set, the test-set AUC of 0.561 is modest. Transaction costs, bid-ask spreads, and implementation slippage would likely consume much of any theoretical edge.

6. **Walk-forward significance:** Despite directional improvement on the fixed test set, the walk-forward paired t-test (p = 0.256) indicates the news advantage is not consistently present across market regimes. Practitioners would require stronger evidence before deploying.

7. **Dataset bias:** The Kaggle headlines dataset likely over-represents large-cap S&P 500 company news and financial media sources, potentially missing macroeconomic or geopolitical signals that move markets.

---

## 8. Future Work

1. **Domain-specific embedding models:** Replace `all-MiniLM-L6-v2` with `FinBERT` (fine-tuned on financial text) or a larger model (e.g., `all-mpnet-base-v2`, 768-dim). Domain-specific fine-tuning may capture financial jargon semantics that general-purpose models miss.

2. **Attention-weighted pooling:** Instead of mean pooling, learn a weighted average of headline embeddings using a small attention layer. The weights could be learned jointly with the prediction model on the training set, potentially identifying which types of headlines (Fed announcements vs. earnings reports) carry more signal.

3. **Fix the zero-headline shock definition:** Compute rolling z-scores only over days with `has_news = 1` to avoid conflating extreme silence with extreme news.

4. **PCA threshold sweep:** Systematically compare 50%, 70%, 85%, and 95% variance thresholds with proper walk-forward evaluation to find the optimal dimensionality for news feature compression.

5. **Multi-asset prediction:** Extend to sector ETFs (XLK, XLE, XLF, XLV) where news sentiment may be more targeted and predictive.

6. **Intraday prediction:** If timestamped headlines are available, predict intraday returns (opening 30-minute move) from pre-market news only, avoiding the same-day mixing problem.

7. **Regime-aware modeling:** Train separate models for high-VIX (VIX > 25) and low-VIX regimes. The walk-forward chart shows enhanced model performance varies substantially across periods — regime conditioning could stabilize this.

8. **Ensemble methods:** Combine the XGBoost and LR predictions with a stacking ensemble. LR and XGBoost make different error patterns (as shown by McNemar), suggesting their predictions may be complementary.

9. **Alternative text sources:** Supplement Kaggle headlines with SEC 8-K filings, FOMC minutes, or earnings call transcripts — all of which have more consistent coverage and higher information density than consumer media headlines.

---

## 9. Conclusion

This project built a multimodal market direction prediction system combining classical technical indicators with transformer-derived news embeddings. The enhanced model (XGBoost, 17 technical + 79 PCA news + 5 shock/coverage features) achieved a test-set AUC of 0.561 and 54.9% accuracy on 377 held-out trading days from 2022–2023, outperforming the technical-only baseline (AUC 0.496, 47.8% accuracy) by a meaningful margin. McNemar's test confirmed the models make statistically different errors (p = 0.035).

However, walk-forward validation across 146 monthly windows revealed that this advantage does not hold consistently over time. The enhanced model won only 52.1% of monthly windows, and the paired t-test on monthly AUC differences was not significant (p = 0.256). The feature importance analysis confirmed the model genuinely uses news content — 14 of the 15 most important features in the enhanced model are news PCA components — but the signal those components carry is regime-dependent and sensitive to the specific market environment of the test period.

The key takeaway is methodological: **rigorous temporal validation reveals that news-based features are not a universally reliable source of edge in equity direction prediction.** The positive test-set result is real but likely reflects a partial alignment between the test-period market regime and the training-data patterns. Markets are adaptive, and any systematic signal embedded in financial news is likely to be arbitraged away as more sophisticated actors exploit it.

Despite the modest predictive performance, the project demonstrates a complete, production-quality pipeline: from raw data ingestion through embedding generation, dimensionality reduction, temporal cross-validation, threshold tuning, statistical testing, a Streamlit dashboard, and a RAG-powered market assistant. The methodology is rigorous, reproducible, and extensible — a solid foundation for further experimentation with richer text sources, larger embedding models, or regime-aware architectures.

---

## 10. References

### Datasets
- **SPY and VIX price data:** Yahoo Finance via [`yfinance`](https://github.com/ranaroussi/yfinance) Python library
- **S&P 500 Financial Headlines:** Kaggle dataset by Dyuti Das Mahapatra — [S&P 500 with Financial News Headlines 2008–2024](https://www.kaggle.com/datasets/dyutidasmahaptra/s-and-p-500-with-financial-news-headlines20082024/data)

### Models and Libraries
- **Sentence embedding model:** `sentence-transformers/all-MiniLM-L6-v2` — [Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Sentence Transformers library:** Reimers & Gurevych (2019) — [GitHub](https://github.com/UKPLab/sentence-transformers)
- **XGBoost:** Chen & Guestrin (2016) — [GitHub](https://github.com/dmlc/xgboost)
- **scikit-learn:** Pedregosa et al. (2011) — [Documentation](https://scikit-learn.org/)
- **pandas:** [Documentation](https://pandas.pydata.org/)
- **NumPy:** [Documentation](https://numpy.org/)
- **Streamlit:** [Documentation](https://streamlit.io/)
- **Plotly:** [Documentation](https://plotly.com/python/)
- **ChromaDB:** [Documentation](https://docs.trychroma.com/)
- **LangChain:** [Documentation](https://python.langchain.com/)
- **Anthropic Claude API:** [Documentation](https://docs.anthropic.com/)
- **statsmodels:** Seabold & Perktold (2010) — [Documentation](https://www.statsmodels.org/)
- **scipy:** Virtanen et al. (2020) — [Documentation](https://scipy.org/)

### Methodological References
- **McNemar's test:** McNemar, Q. (1947). "Note on the sampling error of the difference between correlated proportions or percentages." *Psychometrika*, 12(2), 153–157.
- **Youden's J statistic:** Youden, W. J. (1950). "Index for rating diagnostic tests." *Cancer*, 3(1), 32–35.
- **Walk-forward validation:** Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- **Bootstrap confidence intervals:** Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- **Temporal cross-validation for financial data:** Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.), Chapter 5.