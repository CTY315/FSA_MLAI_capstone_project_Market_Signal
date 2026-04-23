"""AI Market Signal Dashboard — Streamlit app."""

import pickle
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2 as chi2_dist
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    roc_curve,
)
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Market Signal Dashboard",
    page_icon="📈",
    layout="wide",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"

TRAIN_END = "2021-01-01"
VAL_END = "2022-07-01"

TECH_COLS = [
    "return_1d", "return_5d", "return_10d",
    "price_to_ma10", "price_to_ma20", "price_to_ma50", "price_to_ma100", "price_to_ma200",
    "volatility_5d", "volatility_10d", "volatility_20d",
    "volume_change", "volume_to_avg20",
    "vix_level", "vix_change_1d", "vix_change_5d", "vix_to_ma20",
]

# ── Consistent color palette ───────────────────────────────────────────────────
CLR_ENHANCED = "#1f77b4"   # blue  — XGB Enhanced
CLR_BASELINE = "#888888"   # gray  — XGB Baseline
CLR_LR_TECH  = "#ff7f0e"   # orange — LR Tech only
CLR_LR_NEWS  = "#2ca02c"   # green  — LR Tech+News
CHART_H      = 420

# ── Data / model loaders ───────────────────────────────────────────────────────
@st.cache_data
def load_features() -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED / "final_features.csv", index_col=0, parse_dates=True)
    return df


@st.cache_data
def load_walkforward() -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED / "walkforward_results.csv")
    df["month"] = pd.to_datetime(df["month"])
    return df


@st.cache_data
def load_headlines() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW / "sp500_headlines.csv")
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_spy() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW / "spy_daily.csv", index_col=0, parse_dates=True)
    df.index.name = "Date"
    return df[["Close"]]


@st.cache_resource
def load_models():
    with open(MODELS_DIR / "baseline_model.pkl", "rb") as f:
        baseline = pickle.load(f)
    with open(MODELS_DIR / "enhanced_model.pkl", "rb") as f:
        enhanced = pickle.load(f)
    return baseline, enhanced


@st.cache_resource
def load_thresholds() -> dict:
    thresh_path = MODELS_DIR / "thresholds.pkl"
    if thresh_path.exists():
        import joblib
        return joblib.load(thresh_path)
    return {"baseline": 0.5, "enhanced": 0.5}


@st.cache_resource
def load_lr_models(df: pd.DataFrame):
    """Train LR (tech-only and tech+news) on the train split. Cached across reruns."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    train = df[df.index < TRAIN_END]
    X_tr_tech = train[TECH_COLS]
    X_tr_all  = train[[c for c in df.columns if c != "target"]]
    y_tr      = train["target"]

    def _pipe():
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("lr",  LogisticRegression(max_iter=1000, random_state=42, C=0.1)),
        ])

    lr_tech = _pipe(); lr_tech.fit(X_tr_tech, y_tr)
    lr_all  = _pipe(); lr_all.fit(X_tr_all,  y_tr)
    return lr_tech, lr_all


@st.cache_resource(show_spinner=False)
def get_rag_chain(api_key: str):
    """Load the LangChain RAG chain (cached across reruns)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.rag import build_rag_chain
    return build_rag_chain(api_key)


# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data
def get_test_data(df: pd.DataFrame):
    test = df[df.index >= VAL_END].copy()
    all_cols = [c for c in df.columns if c != "target"]
    X_tech = test[TECH_COLS]
    X_all = test[all_cols]
    y = test["target"]
    return test, X_tech, X_all, y


@st.cache_data
def predict_with_model(_model, X: pd.DataFrame, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_pred, y_prob). Reorders X columns to match model feature names."""
    feat_names = _model.get_booster().feature_names
    X_ordered = X[feat_names]
    y_prob = _model.predict_proba(X_ordered)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return y_pred, y_prob


def bootstrap_auc(y_true: np.ndarray, y_prob: np.ndarray, n: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        try:
            aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
        except ValueError:
            pass
    aucs = np.array(aucs)
    return aucs.mean(), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def metrics(y_true: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict:
    valid = ~(np.isnan(y_true) | np.isnan(prob))
    y, p, d = y_true[valid], prob[valid], pred[valid]
    return {
        "accuracy": accuracy_score(y, d),
        "auc": roc_auc_score(y, p),
        "log_loss": log_loss(y, p),
    }


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 AI Market Signal")
    st.markdown("**Author:** Shirley Cheung")
    st.markdown("**Program:** FullStack Academy AI/ML — Cohort 2510-FTB-CT-AIM-PT")
    st.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-Repo-black?logo=github)]"
        "(https://github.com/CTY315/FSA_AI_capstone_project)"
    )
    st.divider()
    st.markdown(
        "Predicts **next-day SPY direction** using XGBoost trained on technical indicators "
        "and transformer-derived news embeddings (all-MiniLM-L6-v2). "
        "Includes walk-forward validation and a RAG-powered market assistant."
    )
    st.divider()
    with st.expander("ℹ️ How to use"):
        st.markdown(
            "**🤖 Market Assistant**\n"
            "Ask natural-language questions about historical market headlines (2008–2023). "
            "Requires an Anthropic API key in `.streamlit/secrets.toml`.\n\n"
            "**🎯 My Prediction**\n"
            "Pick a test-set day, review the previous day's news signals and technical "
            "indicators, make your Up/Down call, then reveal the answer.\n\n"
            "**📰 Headlines Browser**\n"
            "Browse S&P 500 financial headlines by trading date. Use the shock-date shortcut "
            "or pick any date with the date picker.\n\n"
            "**🔮 Daily Model Predictions**\n"
            "View the enhanced model's predicted vs actual SPY direction over the test window. "
            "Filter by date range and inspect the prediction table.\n\n"
            "**⚖️ Model Comparison**\n"
            "Compare all 4 models (XGB baseline/enhanced + LR tech/news) side-by-side with "
            "ROC curves, confusion matrices, bootstrap CIs, and McNemar's test.\n\n"
            "**📊 Performance Overview**\n"
            "Top-level test-set metrics, walk-forward AUC over time, and feature importance charts."
        )
    st.divider()
    st.markdown(
        "| Split | Period |\n"
        "|---|---|\n"
        "| Train | 2008-10-15 → 2020-12-31 |\n"
        "| Val | 2021-01-01 → 2022-06-30 |\n"
        "| Test | 2022-07-01 → 2023-12-29 |"
    )

# ── Load data ──────────────────────────────────────────────────────────────────
df = load_features()
wf = load_walkforward()
headlines = load_headlines()
spy = load_spy()
baseline_model, enhanced_model = load_models()
thresholds = load_thresholds()
lr_tech_model, lr_all_model = load_lr_models(df)

test_df, X_tech, X_all, y_test = get_test_data(df)

bl_pred, bl_prob = predict_with_model(baseline_model, test_df[TECH_COLS], thresholds["baseline"])
en_pred, en_prob = predict_with_model(enhanced_model, test_df[[c for c in df.columns if c != "target"]], thresholds["enhanced"])

_X_test_tech = test_df[TECH_COLS]
_X_test_all  = test_df[[c for c in df.columns if c != "target"]]
lr_tech_prob = lr_tech_model.predict_proba(_X_test_tech)[:, 1]
lr_tech_pred = (lr_tech_prob >= 0.5).astype(int)
lr_all_prob  = lr_all_model.predict_proba(_X_test_all)[:, 1]
lr_all_pred  = (lr_all_prob >= 0.5).astype(int)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🤖 Market Assistant",
    "🎯 My Prediction",
    "📰 Headlines Browser",
    "🔮 Daily Model Predictions",
    "⚖️ Model Comparison",
    "📊 Performance Overview",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Headlines Browser
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Headlines Browser")
    st.caption(
        "Browse raw S&P 500 financial headlines by trading date. "
        "News shock days are flagged when headline volume z-score |z| > 2 "
        "relative to a rolling 20-day window."
    )

    @st.fragment
    def _headlines_ui() -> None:
        min_hl_date = headlines["date"].dt.date.min()
        max_hl_date = min(headlines["date"].dt.date.max(), pd.Timestamp("2023-12-31").date())

        if "shock_expander_open" not in st.session_state:
            st.session_state["shock_expander_open"] = False

        # ── Random date button ─────────────────────────────────────────────
        dates_with_hl = headlines[
            (headlines["date"].dt.date >= min_hl_date) &
            (headlines["date"].dt.date <= max_hl_date)
        ]["date"].dt.date.unique()

        if st.button("🎲 Random date"):
            rng = np.random.default_rng()
            st.session_state["hl_date"] = rng.choice(dates_with_hl)
            st.session_state["shock_expander_open"] = False
            st.rerun(scope="fragment")

        # ── Shock dates expander ───────────────────────────────────────────
        with st.expander("📅 Show all news shock dates", expanded=st.session_state["shock_expander_open"]):
            st.session_state["shock_expander_open"] = True
            shock_dates = df[df["news_volume_shock"] == 1].index
            if len(shock_dates) == 0:
                st.info("No news shock dates found in the feature matrix.")
            else:
                st.caption(f"{len(shock_dates)} news shock days (|volume z-score| > 2)")
                cols = st.columns(6)
                for i, d in enumerate(sorted(shock_dates)):
                    if cols[i % 6].button(str(d.date()), key=f"shock_{d.date()}"):
                        st.session_state["hl_date"] = d.date()
                        st.session_state["shock_expander_open"] = False
                        st.rerun(scope="fragment")

        _default_date = st.session_state.pop("hl_date", max_hl_date)
        selected_date = st.date_input(
            "Select a date",
            value=_default_date,
            min_value=min_hl_date,
            max_value=max_hl_date,
        )

        ts = pd.Timestamp(selected_date)

        # ── Shock warning banner ───────────────────────────────────────────
        if ts in df.index:
            row = df.loc[ts]
            is_vol_shock  = bool(row.get("news_volume_shock", 0))
            is_mag_shock  = bool(row.get("news_magnitude_shock", 0))
            if is_vol_shock or is_mag_shock:
                shock_type = []
                if is_vol_shock:
                    shock_type.append("headline volume")
                if is_mag_shock:
                    shock_type.append("embedding magnitude")
                st.warning(f"⚡ This was a news shock day — unusually high {' & '.join(shock_type)}.")

        day_hl = headlines[headlines["date"].dt.date == selected_date][
            ["title"]
        ].reset_index(drop=True)
        day_hl.index += 1

        # ── Headline count metric card ─────────────────────────────────────
        count_col, _ = st.columns([1, 3])
        count_col.metric("Headlines on this day", len(day_hl))

        st.divider()

        meta_col, table_col = st.columns([1, 3])
        with meta_col:
            if ts in df.index:
                row = df.loc[ts]
                vol_z = row.get("news_volume_zscore")
                if not pd.isna(vol_z):
                    st.metric("Volume Z-Score", f"{vol_z:.2f}")
                mag_shock = bool(row.get("news_magnitude_shock", 0))
                st.metric("Magnitude Shock", "Yes ⚡" if mag_shock else "No")
            else:
                if ts < df.index.min():
                    st.info(
                        f"Date is before the feature matrix start ({df.index.min().date()}). "
                        "Technical features require a warm-up period for rolling indicators."
                    )
                else:
                    st.info("Date not in feature matrix (market holiday or weekend).")

        with table_col:
            if len(day_hl) == 0:
                st.warning("No headlines found for this date.")
            else:
                st.dataframe(
                    day_hl.rename(columns={"title": "Headline"}),
                    use_container_width=True,
                    height=420,
                )

    _headlines_ui()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — My Prediction
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🎯 Can You Predict the Market?")
    st.caption(
        "Pick a test-set trading day, review the previous day's news signals and "
        "technical indicators, make your call, then reveal the answer."
    )

    @st.fragment
    def _quiz_ui() -> None:
        test_dates = sorted(test_df.index.date)
        valid_dates = [d for d in test_dates if pd.Timestamp(d) in df.index]

        # ── Date selector ──────────────────────────────────────────────────
        quiz_date = st.date_input(
            "Select a trading day to predict",
            value=valid_dates[0],
            min_value=valid_dates[0],
            max_value=valid_dates[-1],
            key="quiz_date",
        )

        ts = pd.Timestamp(quiz_date)

        # Reset guess whenever the date changes
        if st.session_state.get("quiz_last_date") != quiz_date:
            st.session_state["quiz_guess"]    = None
            st.session_state["quiz_revealed"] = False
            st.session_state["quiz_last_date"] = quiz_date

        # ── Previous-day signals ───────────────────────────────────────────
        prior_rows = df[df.index < ts]
        if prior_rows.empty:
            st.info("No prior-day data available for this date.")
            return

        prev_row  = prior_rows.iloc[-1]
        prev_date = prior_rows.index[-1].date()

        # ── Previous day headlines ─────────────────────────────────────────
        prev_headlines = headlines[headlines["date"].dt.date == prev_date][["title"]].reset_index(drop=True)
        prev_headlines.index += 1
        with st.expander(f"📰 Previous day headlines ({len(prev_headlines)} headlines on {prev_date})", expanded=True):
            if prev_headlines.empty:
                st.info("No headlines found for this date.")
            else:
                st.dataframe(
                    prev_headlines.rename(columns={"title": "Headline"}),
                    use_container_width=True,
                    height=min(200, 35 * len(prev_headlines) + 38),
                )

        st.markdown(f"**Previous trading day signals** *(as of {prev_date})*")

        vol_shock = bool(prev_row.get("news_volume_shock", 0))
        mag_shock = bool(prev_row.get("news_magnitude_shock", 0))

        sig1, sig2 = st.columns(2)
        sig1.metric(
            "News Volume Shock",
            "⚡ Yes" if vol_shock else "No",
            help="True when prior-day headline count z-score |z| > 2 vs. rolling 20-day window",
        )
        sig2.metric(
            "Magnitude Shock",
            "⚡ Yes" if mag_shock else "No",
            help="True when prior-day embedding magnitude z-score |z| > 2 — news was unusually directional",
        )

        # ── Technical features table ───────────────────────────────────────
        with st.expander("📊 Previous day technical indicators", expanded=True):
            TECH_GROUPS = {
                "Returns": {
                    "return_1d":  ("1-Day Log Return",        "Yesterday's log return — short-term momentum"),
                    "return_5d":  ("5-Day Log Return",        "5-day log return — weekly momentum"),
                    "return_10d": ("10-Day Log Return",       "10-day log return — bi-weekly momentum"),
                },
                "Price vs Moving Averages": {
                    "price_to_ma10":  ("Price / MA10 − 1",   "Deviation from 10-day average"),
                    "price_to_ma20":  ("Price / MA20 − 1",   "Deviation from 20-day average"),
                    "price_to_ma50":  ("Price / MA50 − 1",   "Deviation from 50-day average"),
                    "price_to_ma100": ("Price / MA100 − 1",  "Deviation from 100-day average"),
                    "price_to_ma200": ("Price / MA200 − 1",  "Deviation from 200-day average (golden cross)"),
                },
                "Volatility": {
                    "volatility_5d":  ("5-Day Volatility",   "Rolling 5-day std of log returns"),
                    "volatility_10d": ("10-Day Volatility",  "Rolling 10-day std of log returns"),
                    "volatility_20d": ("20-Day Volatility",  "Rolling 20-day std of log returns"),
                },
                "Volume": {
                    "volume_change":   ("Volume Change",        "Day-over-day volume % change"),
                    "volume_to_avg20": ("Volume / 20-Day Avg",  "Today's volume relative to 20-day average"),
                },
                "VIX (Fear Index)": {
                    "vix_level":     ("VIX Level",        "Absolute VIX close — market fear level"),
                    "vix_change_1d": ("VIX 1-Day Change", "Day-over-day VIX % change"),
                    "vix_change_5d": ("VIX 5-Day Change", "5-day VIX % change"),
                    "vix_to_ma20":   ("VIX / MA20 − 1",   "VIX deviation from its 20-day average"),
                },
            }

            for group_name, features in TECH_GROUPS.items():
                st.markdown(f"**{group_name}**")
                cols = st.columns(len(features))
                for col, (feat, (label, tooltip)) in zip(cols, features.items()):
                    val = prev_row.get(feat, None)
                    if val is None or (hasattr(val, '__float__') and pd.isna(float(val))):
                        display_val = "N/A"
                    elif feat == "vix_level":
                        display_val = f"{val:.2f}"
                    elif feat == "volume_to_avg20":
                        display_val = f"{val:.2f}x"
                    elif feat.startswith("return_") or feat.startswith("price_to_ma") or \
                         feat.startswith("vix_change") or feat in ("vix_to_ma20", "volume_change"):
                        display_val = f"{val:.2%}"
                    else:
                        display_val = f"{val:.4f}"
                    col.metric(label, display_val, help=tooltip)

        st.divider()

        # ── Model prediction ───────────────────────────────────────────────
        # Look up the enhanced model's prediction for this date
        if ts in test_df.index:
            _idx = list(test_df.index).index(ts)
            model_prob  = float(en_prob[_idx])
            model_pred  = int(en_pred[_idx])
            model_dir   = "Up" if model_pred == 1 else "Down"
            model_icon  = "📈" if model_pred == 1 else "📉"
            mc1, mc2, _ = st.columns([1, 1, 3])
            mc1.metric(
                "Model Prediction",
                f"{model_icon} {model_dir}",
                help=f"XGBoost enhanced model prediction (threshold = {thresholds['enhanced']})",
            )
            mc2.metric(
                "P(Up)" if model_pred == 1 else "P(Down)",
                f"{model_prob:.1%}" if model_pred == 1 else f"{1 - model_prob:.1%}",
                help="Model's predicted probability in the direction of its call",
            )
        else:
            st.info("Model prediction not available for this date (outside test window).")

        st.divider()

        # ── Prediction buttons ─────────────────────────────────────────────
        if st.session_state["quiz_guess"] is None:
            st.markdown("**Your prediction for today's close:**")
            btn_col1, btn_col2, _ = st.columns([1, 1, 3])
            if btn_col1.button("📈 Up", use_container_width=True, key="quiz_btn_up"):
                st.session_state["quiz_guess"] = "Up"
                st.rerun(scope="fragment")
            if btn_col2.button("📉 Down", use_container_width=True, key="quiz_btn_down"):
                st.session_state["quiz_guess"] = "Down"
                st.rerun(scope="fragment")
        else:
            guess = st.session_state["quiz_guess"]
            st.markdown(f"**Your prediction:** {'📈 Up' if guess == 'Up' else '📉 Down'}")

            # ── Reveal answer ──────────────────────────────────────────────
            if not st.session_state["quiz_revealed"]:
                if st.button("👁 Reveal answer", key="quiz_reveal"):
                    st.session_state["quiz_revealed"] = True
                    st.rerun(scope="fragment")
            else:
                actual_val = int(df.loc[ts, "target"]) if ts in df.index else None

                if actual_val is None:
                    st.warning("Actual direction not available for this date.")
                else:
                    actual_dir = "Up" if actual_val == 1 else "Down"
                    correct    = (guess == actual_dir)

                    res_col1, res_col2 = st.columns(2)
                    res_col1.metric("Actual direction", "📈 Up" if actual_val == 1 else "📉 Down")
                    res_col2.metric("Result", "✅ Correct!" if correct else "❌ Incorrect")

                    if correct:
                        st.success(f"Nice call! SPY moved **{actual_dir}** on {quiz_date}.")
                    else:
                        st.error(
                            f"SPY moved **{actual_dir}** on {quiz_date} — "
                            f"you predicted {guess}. Better luck next time!"
                        )

            if st.button("🔄 Try another date", key="quiz_reset"):
                st.session_state["quiz_guess"]    = None
                st.session_state["quiz_revealed"] = False
                st.rerun(scope="fragment")

    _quiz_ui()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — Daily Model Predictions
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Daily Predictions")
    st.info(
        "Enhanced model predicted vs actual market direction over the test period. "
        "Hover over dots for predicted confidence, actual direction, and news shock status."
    )

    test_start = pd.Timestamp(VAL_END).date()
    test_end   = test_df.index.max().date()

    col_s, col_e = st.columns(2)
    with col_s:
        range_start = st.date_input(
            "Start date", value=test_start, min_value=test_start, max_value=test_end, key="pred_start"
        )
    with col_e:
        range_end = st.date_input(
            "End date", value=test_end, min_value=test_start, max_value=test_end, key="pred_end"
        )

    pred_df = test_df[[*TECH_COLS, "target"]].copy()
    pred_df["prob_up"] = en_prob
    pred_df["pred"]    = en_pred
    pred_df["correct"] = (pred_df["pred"] == pred_df["target"]).astype(int)

    mask = (pred_df.index.date >= range_start) & (pred_df.index.date <= range_end)
    view = pred_df[mask].copy()

    if view.empty:
        st.warning("No data in selected range.")
    else:
        view["pred_dir"]   = view["pred"].map({1: "Up", 0: "Down"})
        view["actual_dir"] = view["target"].map({1: "Up", 0: "Down"})
        shock_map          = df["news_volume_shock"].reindex(view.index).fillna(0).astype(bool)
        view["is_shock"]   = shock_map.values
        view["close"]      = spy["Close"].reindex(view.index)

        # ── Summary metrics ────────────────────────────────────────────────
        n_correct  = int(view["correct"].sum())
        n_total    = len(view)
        accuracy   = view["correct"].mean()

        sm1, sm2, sm3 = st.columns(3)
        sm1.metric("Days Shown",           n_total)
        sm2.metric("Correct Predictions",  n_correct)
        sm3.metric("Accuracy (range)",     f"{accuracy:.1%}")

        st.divider()

        hover_text = [
            f"<b>Date:</b> {idx.date()}<br>"
            f"<b>Close:</b> ${c:.2f}<br>"
            f"<b>Predicted:</b> {pd_} ({pu:.1%})<br>"
            f"<b>Actual:</b> {ad}<br>"
            f"<b>Match:</b> {'Yes' if pd_ == ad else 'No'}<br>"
            f"<b>News Shock:</b> {'⚡ Yes' if s else 'No'}"
            for idx, c, pd_, pu, ad, s in zip(
                view.index, view["close"], view["pred_dir"],
                view["prob_up"], view["actual_dir"], view["is_shock"]
            )
        ]
        dot_colors = ["#2ca02c" if s else CLR_ENHANCED for s in view["is_shock"]]

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=view.index, y=view["close"], mode="lines", name="SPY Close",
            line=dict(color=CLR_ENHANCED, width=1.5), hoverinfo="skip",
        ))
        fig_pred.add_trace(go.Scatter(
            x=view.index, y=view["close"], mode="markers", name="Trading Day",
            marker=dict(color=dot_colors, size=7, symbol="circle"),
            hovertext=hover_text, hovertemplate="%{hovertext}<extra></extra>",
        ))
        fig_pred.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name="News Shock Day",
            marker=dict(color="#2ca02c", size=7, symbol="circle"),
        ))
        fig_pred.update_layout(
            xaxis_title="Date", yaxis_title="SPY Close Price ($)",
            height=CHART_H, margin=dict(t=20, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig_pred, use_container_width=True)
        st.caption("Green dots = news shock days (headline volume z-score |z| > 2)")

        st.divider()

        st.subheader("Prediction Table")

        display = view[["prob_up", "pred", "target", "correct"]].copy()
        display.index = display.index.date
        display.index.name = "Date"
        display["P(Up)"]               = display["prob_up"].round(3)
        display["Predicted Direction"] = display["pred"].map({1: "Up", 0: "Down"})
        display["Actual Direction"]    = display["target"].map({1: "Up", 0: "Down"})
        display["Correct"]             = display["correct"]

        out = display[["P(Up)", "Predicted Direction", "Actual Direction", "Correct"]]

        def _row_color(row):
            color = "#d4edda" if row["Correct"] == 1 else "#f8d7da"
            return [f"background-color: {color}"] * len(row)

        st.dataframe(
            out.style.apply(_row_color, axis=1),
            use_container_width=True,
            height=420,
        )
        st.caption(
            f"Showing {n_total} trading days · "
            f"{n_correct} correct ({accuracy:.1%}) · "
            "Green = correct prediction, Red = incorrect"
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Market Assistant (RAG)
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.html("""
    <div style="background:#0e1117; padding: 8px 8px 8px 8px;">
      <p style="color:#4ecdc4; font-size:1.25rem; font-weight:700; margin-bottom:16px;">How It Works</p>
      <div style="display:flex; align-items:center; gap:0;">

        <div style="background:#1a2744; border-radius:8px; padding:18px 20px; flex:1;">
          <div style="display:flex; align-items:flex-start; gap:12px;">
            <span style="color:#4ecdc4; font-size:1.5rem; font-weight:700; line-height:1;">1</span>
            <div>
              <p style="color:#ffffff; font-weight:700; margin:0 0 6px 0;">You ask</p>
              <p style="color:#a0aec0; margin:0; font-size:0.9rem;">"What happened during March 2020?"</p>
            </div>
          </div>
        </div>

        <div style="color:#4ecdc4; font-size:1.6rem; padding:0 12px;">→</div>

        <div style="background:#1a2744; border-radius:8px; padding:18px 20px; flex:1;">
          <div style="display:flex; align-items:flex-start; gap:12px;">
            <span style="color:#4ecdc4; font-size:1.5rem; font-weight:700; line-height:1;">2</span>
            <div>
              <p style="color:#ffffff; font-weight:700; margin:0 0 6px 0;">System retrieves</p>
              <p style="color:#a0aec0; margin:0; font-size:0.9rem;">Finds 15 most relevant headlines from ChromaDB</p>
            </div>
          </div>
        </div>

        <div style="color:#4ecdc4; font-size:1.6rem; padding:0 12px;">→</div>

        <div style="background:#1a2744; border-radius:8px; padding:18px 20px; flex:1;">
          <div style="display:flex; align-items:flex-start; gap:12px;">
            <span style="color:#4ecdc4; font-size:1.5rem; font-weight:700; line-height:1;">3</span>
            <div>
              <p style="color:#ffffff; font-weight:700; margin:0 0 6px 0;">AI answers</p>
              <p style="color:#a0aec0; margin:0; font-size:0.9rem;">Claude reads headlines &amp; responds</p>
            </div>
          </div>
        </div>

      </div>
    </div>
    """)

    st.header("Market Assistant")
    st.caption(
        "⚠️ Answers are generated from historical headlines 2008–2023. Not financial advice."
    )

    st.info(
        "Ask natural-language questions about financial headlines from 2008–2023. "
        "The assistant retrieves the most relevant headlines from ChromaDB and uses "
        "Claude to generate a grounded answer."
    )

    _api_key = None  # type: ignore[assignment]
    _rag_ready = False

    try:
        _api_key = st.secrets["ANTHROPIC_API_KEY"]
    except (KeyError, FileNotFoundError):
        st.error(
            "**API key not found.** "
            "Create `.streamlit/secrets.toml` (see `.streamlit/secrets.toml.example`) "
            "and add your Anthropic API key:\n\n"
            "```toml\nANTHROPIC_API_KEY = \"sk-ant-...\"\n```"
        )

    if _api_key:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.rag import is_indexed

        if not is_indexed():
            with st.spinner("Building headline index for first launch..."):
                result = subprocess.run(
                    ["python3", "scripts/index_headlines.py"],
                    cwd=Path(__file__).resolve().parent.parent,
                    capture_output=True,
                    text=True,
                )
            if is_indexed():
                st.rerun()
            else:
                st.error(result.stderr or "ChromaDB indexing failed with no stderr output.")
        else:
            _rag_ready = True

    if _rag_ready and _api_key:
        if "rag_messages" not in st.session_state:
            st.session_state.rag_messages = []

        _EXAMPLE_QUESTIONS = [
            "What were the major news themes during the 2008 financial crisis?",
            "What happened in financial markets during March 2020?",
            "Were there any major Fed announcements about interest rates in 2022?",
            "What were analysts saying about tech stocks in 2021?",
        ]

        @st.fragment
        def _chat_ui(api_key: str) -> None:
            # ── Example question buttons ───────────────────────────────────
            st.markdown("**Try an example question:**")
            eq_cols = st.columns(2)
            for i, q in enumerate(_EXAMPLE_QUESTIONS):
                if eq_cols[i % 2].button(q, key=f"eq_{i}", use_container_width=True):
                    st.session_state["pending_question"] = q
                    st.rerun(scope="fragment")

            st.divider()

            _history_container = st.container()
            _question = st.chat_input("Ask about financial headlines…")

            # Check for pending question from example buttons
            if "pending_question" in st.session_state:
                _question = st.session_state.pop("pending_question")

            with _history_container:
                for _msg in st.session_state.rag_messages:
                    with st.chat_message(_msg["role"]):
                        st.markdown(_msg["content"])
                        if _msg.get("sources"):
                            with st.expander(f"Source Headlines ({len(_msg['sources'])})"):
                                for _src in _msg["sources"]:
                                    st.caption(f"**[{_src['date']}]** {_src['headline']}")

            if _question:
                st.session_state.rag_messages.append(
                    {"role": "user", "content": _question}
                )
                with _history_container:
                    with st.chat_message("user"):
                        st.markdown(_question)

                    with st.chat_message("assistant"):
                        with st.spinner("Retrieving headlines and generating answer…"):
                            try:
                                from src.rag import query_rag

                                _chain   = get_rag_chain(api_key)
                                _answer, _sources = query_rag(_chain, _question)
                                st.markdown(_answer)
                                if _sources:
                                    with st.expander(f"Source Headlines ({len(_sources)})"):
                                        for _src in _sources:
                                            st.caption(
                                                f"**[{_src['date']}]** {_src['headline']}"
                                            )
                                st.session_state.rag_messages.append(
                                    {
                                        "role": "assistant",
                                        "content": _answer,
                                        "sources": _sources,
                                    }
                                )
                            except Exception as _exc:
                                _err = str(_exc)
                                st.error(f"Error querying the RAG chain: {_err}")
                                st.session_state.rag_messages.append(
                                    {"role": "assistant", "content": f"⚠️ Error: {_err}"}
                                )

        _chat_ui(_api_key)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — Model Comparison
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Model Comparison")

    # ── Key findings summary ───────────────────────────────────────────────
    y_arr = y_test.values

    _auc_base = roc_auc_score(y_arr, bl_prob)
    _auc_enh  = roc_auc_score(y_arr, en_prob)

    _correct_b = bl_pred == y_arr
    _correct_e = en_pred == y_arr
    _b_only    = int(np.sum(_correct_b & ~_correct_e))
    _c_only    = int(np.sum(~_correct_b & _correct_e))
    if (_b_only + _c_only) > 0:
        _chi2  = (abs(_b_only - _c_only) - 1) ** 2 / (_b_only + _c_only)
        _pval  = chi2_dist.sf(_chi2, df=1)
        _sig   = "statistically significant (p < 0.05)" if _pval < 0.05 else "not statistically significant (p ≥ 0.05)"
        _mcn_str = f"McNemar p = {_pval:.4f} — {_sig}."
    else:
        _mcn_str = "McNemar test: identical predictions."

    st.info(
        f"**Key findings —** Enhanced model AUC: **{_auc_enh:.3f}** vs Baseline: **{_auc_base:.3f}** "
        f"(Δ = {_auc_enh - _auc_base:+.3f}). {_mcn_str}"
    )

    st.caption(
        f"Test set: {VAL_END} onward, n = {len(y_test)} trading days. "
        "XGB thresholds tuned on val set (baseline=0.59, enhanced=0.55). LR uses default threshold=0.50."
    )

    # ── 4-model summary table ──────────────────────────────────────────────
    st.subheader("Summary — All 4 Models")
    _summary = []
    for name, prob, pred in [
        ("XGB — Baseline (Tech)",      bl_prob,      bl_pred),
        ("XGB — Enhanced (Tech+News)", en_prob,      en_pred),
        ("LR  — Tech only",            lr_tech_prob, lr_tech_pred),
        ("LR  — Tech + News",          lr_all_prob,  lr_all_pred),
    ]:
        _summary.append({
            "Model":    name,
            "Accuracy":         f"{accuracy_score(y_arr, pred):.3f}",
            "AUC":              f"{roc_auc_score(y_arr, prob):.3f}",
            "Predict Up (%)":   f"{pred.mean():.1%}",
        })
    st.dataframe(pd.DataFrame(_summary).set_index("Model"), use_container_width=True)

    st.divider()

    # ── Confusion matrices ─────────────────────────────────────────────────
    st.subheader("Confusion Matrices")
    _cm_models = [
        ("XGB Baseline", bl_pred,      "Greys"),
        ("XGB Enhanced", en_pred,      "Blues"),
        ("LR Tech only", lr_tech_pred, "Oranges"),
        ("LR Tech+News", lr_all_pred,  "Greens"),
    ]
    cm_cols = st.columns(4)
    for col, (title, pred, cscale) in zip(cm_cols, _cm_models):
        with col:
            st.markdown(f"**{title}**")
            fig_cm = px.imshow(
                confusion_matrix(y_arr, pred),
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Down", "Up"], y=["Down", "Up"],
                text_auto=True,
                color_continuous_scale=cscale,
                zmin=0,
            )
            fig_cm.update_layout(
                height=280,
                margin=dict(l=0, r=0, t=10, b=0),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    st.divider()

    # ── ROC curves ────────────────────────────────────────────────────────
    st.subheader("ROC Curves")
    _roc_models = [
        ("XGB Baseline", bl_prob,      CLR_BASELINE, "solid"),
        ("XGB Enhanced", en_prob,      CLR_ENHANCED, "solid"),
        ("LR Tech only", lr_tech_prob, CLR_LR_TECH,  "dash"),
        ("LR Tech+News", lr_all_prob,  CLR_LR_NEWS,  "dash"),
    ]
    fig_roc = go.Figure()
    for roc_name, prob, color, dash in _roc_models:
        fpr, tpr, _ = roc_curve(y_arr, prob)
        auc_val = roc_auc_score(y_arr, prob)
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{roc_name} (AUC={auc_val:.3f})",
            line=dict(color=color, width=2, dash=dash),
        ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random",
        line=dict(color="red", dash="dash", width=1),
    ))
    fig_roc.update_layout(
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=CHART_H, margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_roc, use_container_width=True)
    st.caption(
        "Solid lines = XGB models (tuned thresholds). "
        "Dashed lines = Logistic Regression (threshold = 0.50). "
        "LR Up% bias (80–96%) reflects the raw probability skew without threshold tuning."
    )

    st.divider()

    # ── Bootstrap CI ──────────────────────────────────────────────────────
    st.subheader("Bootstrap AUC Confidence Intervals (n = 1,000)")
    st.caption("95% CIs via non-parametric bootstrap. Diamonds = bootstrap mean; bars span [2.5%, 97.5%].")

    with st.spinner("Bootstrapping…"):
        mean_b,  lo_b,  hi_b  = bootstrap_auc(y_test.values, bl_prob)
        mean_e,  lo_e,  hi_e  = bootstrap_auc(y_test.values, en_prob)
        mean_lt, lo_lt, hi_lt = bootstrap_auc(y_test.values, lr_tech_prob)
        mean_la, lo_la, hi_la = bootstrap_auc(y_test.values, lr_all_prob)

    ci_rows = [
        ("XGB Baseline", mean_b,  lo_b,  hi_b,  CLR_BASELINE),
        ("XGB Enhanced", mean_e,  lo_e,  hi_e,  CLR_ENHANCED),
        ("LR Tech only", mean_lt, lo_lt, hi_lt, CLR_LR_TECH),
        ("LR Tech+News", mean_la, lo_la, hi_la, CLR_LR_NEWS),
    ]
    fig_ci = go.Figure()
    for model_name, mean_, lo_, hi_, color in ci_rows:
        fig_ci.add_trace(go.Scatter(
            x=[lo_, hi_], y=[model_name, model_name], mode="lines",
            line=dict(color=color, width=6), showlegend=False,
        ))
        fig_ci.add_trace(go.Scatter(
            x=[mean_], y=[model_name], mode="markers",
            marker=dict(color=color, size=14, symbol="diamond"), name=model_name,
        ))
    fig_ci.add_vline(x=0.5, line_dash="dash", line_color="red",
                     annotation_text="Random", annotation_position="top right")
    fig_ci.update_layout(
        height=320, xaxis_title="AUC", xaxis=dict(range=[0.3, 0.85]),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_ci, use_container_width=True)

    ci_df = pd.DataFrame({
        "Model":            ["XGB Baseline", "XGB Enhanced", "LR Tech only", "LR Tech+News"],
        "Mean AUC":         [mean_b,  mean_e,  mean_lt, mean_la],
        "CI Lower (2.5%)":  [lo_b,    lo_e,    lo_lt,   lo_la],
        "CI Upper (97.5%)": [hi_b,    hi_e,    hi_lt,   hi_la],
    }).set_index("Model")
    st.dataframe(ci_df.style.format("{:.3f}"), use_container_width=True)

    st.divider()

    # ── McNemar test ───────────────────────────────────────────────────────
    st.subheader("McNemar's Test — XGB Baseline vs XGB Enhanced")
    st.info(
        "Checks whether the two XGBoost models make **significantly different errors** "
        "on the same observations. p < 0.05 means the enhanced model's improvement is statistically meaningful."
    )

    correct_b = bl_pred == y_arr
    correct_e = en_pred == y_arr
    b_only    = int(np.sum(correct_b & ~correct_e))
    c_only    = int(np.sum(~correct_b & correct_e))

    if (b_only + c_only) == 0:
        st.warning("No discordant pairs — the models made identical predictions on every test day.")
    else:
        chi2_stat = (abs(b_only - c_only) - 1) ** 2 / (b_only + c_only)
        p_value   = chi2_dist.sf(chi2_stat, df=1)

        mn_c1, mn_c2, mn_c3, mn_c4 = st.columns(4)
        mn_c1.metric("Baseline-only correct (b)", b_only)
        mn_c2.metric("Enhanced-only correct (c)", c_only)
        mn_c3.metric("Chi² statistic", f"{chi2_stat:.3f}")
        mn_c4.metric("p-value", f"{p_value:.4f}")

        if p_value < 0.05:
            st.success(
                f"p = {p_value:.4f} < 0.05 — The models make **significantly different errors**. "
                "The enhanced model's improvement is statistically meaningful."
            )
        else:
            st.warning(
                f"p = {p_value:.4f} ≥ 0.05 — No significant difference detected. "
                "The improvement from news embeddings may be within noise."
            )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — Performance Overview
# ═════════════════════════════════════════════════════════════════════════════
with tab6:
    st.header("Performance Overview")
    st.caption(
        f"Test-set metrics ({VAL_END} onward, n = {len(y_test)} trading days). "
        "Baseline uses technical indicators only; Enhanced adds transformer news embeddings."
    )

    m_base = metrics(y_test.values, bl_prob, bl_pred)
    m_enh  = metrics(y_test.values, en_prob, en_pred)

    # ── Metric cards — grouped as Baseline / Enhanced rows ────────────────
    st.markdown("##### Baseline Model (XGB — Tech Only, threshold = 0.59)")
    b1, b2, b3 = st.columns(3)
    b1.metric("Accuracy",  f"{m_base['accuracy']:.3f}")
    b2.metric("ROC-AUC",   f"{m_base['auc']:.3f}")
    b3.metric("Log Loss",  f"{m_base['log_loss']:.3f}")

    st.markdown("##### Enhanced Model (XGB — Tech + News, threshold = 0.55)")
    e1, e2, e3 = st.columns(3)
    e1.metric(
        "Accuracy", f"{m_enh['accuracy']:.3f}",
        delta=f"{m_enh['accuracy'] - m_base['accuracy']:+.3f}",
    )
    e2.metric(
        "ROC-AUC", f"{m_enh['auc']:.3f}",
        delta=f"{m_enh['auc'] - m_base['auc']:+.3f}",
    )
    e3.metric(
        "Log Loss", f"{m_enh['log_loss']:.3f}",
        delta=f"{m_enh['log_loss'] - m_base['log_loss']:+.3f}",
        delta_color="inverse",
    )

    st.divider()

    # ── Walk-forward AUC ──────────────────────────────────────────────────
    st.subheader("Walk-Forward AUC Over Time")
    fig_wf = go.Figure()
    fig_wf.add_trace(go.Scatter(
        x=wf["month"], y=wf["baseline_auc"], name="Baseline",
        mode="lines+markers",
        line=dict(color=CLR_BASELINE, width=2), marker=dict(size=4),
    ))
    fig_wf.add_trace(go.Scatter(
        x=wf["month"], y=wf["enhanced_auc"], name="Enhanced",
        mode="lines+markers",
        line=dict(color=CLR_ENHANCED, width=2), marker=dict(size=4),
    ))
    fig_wf.add_hline(
        y=0.5, line_dash="dash", line_color="red",
        annotation_text="Random (0.5)", annotation_position="bottom right",
    )
    fig_wf.update_layout(
        xaxis_title="Month", yaxis_title="AUC",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=CHART_H, margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_wf, use_container_width=True)
    st.caption(
        "Expanding-window walk-forward validation: each point is trained on all data up to that month "
        "and evaluated on the following month. Shows how AUC varies across different market regimes."
    )

    st.divider()

    # ── Feature importance ─────────────────────────────────────────────────
    st.subheader("Feature Importance — Top 15 by Gain")
    st.caption(
        "Gain measures the average improvement in the loss function when a feature is used for a split. "
        "Higher = more useful to the model."
    )

    def top_importance(model, top_n: int = 15) -> pd.DataFrame:
        scores = model.get_booster().get_score(importance_type="gain")
        return (
            pd.DataFrame(scores.items(), columns=["feature", "gain"])
            .sort_values("gain", ascending=False)
            .head(top_n)
        )

    fi_col1, fi_col2 = st.columns(2)
    for col, model, title, color in [
        (fi_col1, baseline_model, "Baseline (Tech Only)", CLR_BASELINE),
        (fi_col2, enhanced_model, "Enhanced (Tech + News)", CLR_ENHANCED),
    ]:
        with col:
            st.markdown(f"**{title}**")
            imp     = top_importance(model)
            fig_fi  = px.bar(
                imp[::-1], x="gain", y="feature", orientation="h",
                color_discrete_sequence=[color],
            )
            fig_fi.update_layout(
                height=500, margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Gain", yaxis_title="",
            )
            st.plotly_chart(fig_fi, use_container_width=True)