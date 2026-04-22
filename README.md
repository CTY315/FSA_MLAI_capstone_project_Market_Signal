# AI Market Signal: Multimodal Market Prediction Using Transformer News Embeddings

## Overview
A machine learning system that combines market technical indicators and transformer-derived news embeddings (headlines) to predict next-day SPY (S&P 500 ETF) market direction. Includes a 5-tab Streamlit dashboard and a RAG-powered market assistant.

**Author:** Shirley Cheung  
**Program:** FullStack Academy AI/ML — Cohort 2510-FTB-CT-AIM-PT  
**Instructor:** Dr. George Perdrizet | **TA:** Andrew Thomas

---

## Project Structure
```
FSA_AI_capstone_project/
├── data/
│   ├── raw/                        # Original downloaded files (not committed)
│   └── processed/                  # Cleaned & merged feature matrices (not committed)
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb  # Technical indicators → technical_features.csv
│   ├── 03_embedding.ipynb          # Headline embeddings (run on Colab with GPU)
│   ├── 04_pca_and_merge.ipynb      # PCA compression + merge → final_features.csv
│   ├── 05_modeling.ipynb           # Train XGBoost models + threshold tuning
│   └── 06_evaluation.ipynb         # Test-set evaluation, walk-forward, bootstrap CIs
├── src/
│   ├── features.py                 # Technical indicator feature engineering
│   ├── embeddings.py               # Embedding aggregation & PCA pipeline
│   ├── model.py                    # Model training, evaluation, walk-forward validation
│   ├── utils.py                    # Shared constants and data loading helpers
│   └── rag.py                      # ChromaDB indexing + RAG chain
├── dashboard/
│   └── app.py                      # 5-tab Streamlit dashboard
├── scripts/
│   └── index_headlines.py          # One-time RAG indexing script
├── models/                         # Saved .pkl artifacts (not committed)
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites
- Python 3.10+
- Google Colab account (for GPU-accelerated embedding generation in notebook 03)
- Anthropic API key (for RAG market assistant tab)

### Installation
```bash
# Clone the repo
git clone https://github.com/CTY315/FSA_AI_capstone_project.git
cd FSA_MLAI_capstone_project_Market_Signal

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Setup
Place the following files in `data/raw/`:
- `spy_daily.csv` — SPY daily OHLCV (from yfinance)
- `vix_daily.csv` — VIX daily data (from yfinance)
- `sp500_headlines.csv` — [Kaggle S&P 500 Headlines 2008–2024](https://www.kaggle.com/datasets/dyutidasmahaptra/s-and-p-500-with-financial-news-headlines20082024/data)

### Run Notebooks in Order
```bash
# 01–02: Local (EDA and feature engineering)
# 03: Run on Google Colab (GPU required for embedding generation)
# 04–06: Local (PCA, modeling, evaluation)
```

### Running the Dashboard
```bash
streamlit run dashboard/app.py
```

### RAG Market Assistant (Optional)
The 2nd dashboard tab requires:
1. An Anthropic API key in `.streamlit/secrets.toml` (copy from `.streamlit/secrets.toml.example`)
2. One-time headline indexing:
```bash
python scripts/index_headlines.py
# Use --force to rebuild the index from scratch
```

---

## Methodology

### Models
| Model | Features | AUC | Accuracy | Up% |
|---|---|---|---|---|
| XGB Baseline | 17 technical indicators | 0.496 | 47.8% | 21.2% |
| **XGB Enhanced** | Tech + 100 PCA news components + shock features | **0.561** | **54.9%** | **40.6%** |
| LR Tech only | 17 technical indicators | 0.457 | 50.1% | 95.8% |
| LR Tech + News | Tech + news | 0.526 | 52.3% | 80.4% |

*Test set: 2022-07-01 → 2023-12-29 (377 trading days). XGB thresholds tuned on validation set.*

### Key Design Decisions
- **Temporal split only** — no random splits. Train: 2008–2020, Val: 2021–2022, Test: 2022–2023
- **PCA on train data only** — prevents leakage; 70% variance threshold (~100 components)
- **Tuned thresholds** — baseline: 0.59, enhanced: 0.55 (tuned on val set, not test set)
- **McNemar's test**: p = 0.035 — enhanced model makes significantly different errors than baseline

### Dashboard Tabs
1. **Performance Overview** — test metrics, walk-forward AUC chart, feature importance
2. **Headlines Browser** — browse raw headlines by date, news shock flags
3. **Daily Predictions** — SPY close price chart with model predictions and shock overlays
4. **Model Comparison** — all 4 models, confusion matrices, ROC curves, bootstrap CIs, McNemar's test
5. **Market Assistant** — RAG-powered Q&A over 2008–2023 financial headlines (requires API key)

---

## Data Sources
- **SPY & VIX:** yfinance API
- **Financial Headlines:** [Kaggle S&P 500 Headlines 2008–2024](https://www.kaggle.com/datasets/dyutidasmahaptra/s-and-p-500-with-financial-news-headlines20082024/data)
