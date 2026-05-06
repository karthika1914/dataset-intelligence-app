import streamlit as st
import pandas as pd
import numpy as np
from utils.cleaning import clean_data, detect_outliers
from utils.visualization import (
    plot_missing_values,
    plot_distributions,
    plot_correlation_heatmap,
    plot_outliers,
    plot_class_balance,
)
from utils.ml_engine import train_models, get_feature_importance
from utils.insights import generate_ai_insights

st.set_page_config(
    page_title="Auto Dataset Intelligence Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #0f172a; }
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-card h3 { color: #94a3b8; font-size: 0.8rem; margin: 0; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card p  { color: #f1f5f9; font-size: 1.8rem; font-weight: 700; margin: 4px 0 0; }
    .section-title {
        font-size: 1.1rem; font-weight: 600;
        color: #38bdf8; border-left: 3px solid #38bdf8;
        padding-left: 10px; margin: 20px 0 10px;
    }
    .badge {
        display: inline-block; padding: 2px 10px;
        border-radius: 99px; font-size: 0.75rem; font-weight: 600;
        background: #1e3a5f; color: #7dd3fc; border: 1px solid #2563eb;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=60)
    st.markdown("## 🧠 Dataset Intelligence Engine")
    st.markdown("**v2.0 — Major Release**")
    st.markdown("---")

    st.markdown("### ⚙️ Settings")
    target_col = None
    task_type  = None
    groq_key   = st.text_input("🔑 OpenRouter API Key (optional)", type="password",
                               help="For AI-powered insights. Free at openrouter.ai")
    st.markdown("---")
    st.markdown("### 📌 Navigation")
    tabs_help = [
        "📊 Overview", "🧹 Cleaning", "📉 Visualizations",
        "🔍 Outliers", "🤖 ML Engine", "💡 AI Insights"
    ]
    for t in tabs_help:
        st.markdown(f"- {t}")

# ── Main ─────────────────────────────────────────────────────────────────────
st.markdown("# 🧠 Auto Dataset Intelligence Engine")
st.markdown('<span class="badge">v2.0</span> &nbsp; <span class="badge">AI-Powered</span> &nbsp; <span class="badge">AutoML</span>', unsafe_allow_html=True)
st.markdown("Upload any CSV and get automated cleaning, visualizations, ML model benchmarking, and AI-generated insights.")
st.markdown("---")

uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])

if not uploaded_file:
    st.info("👆 Upload a CSV file to get started. Try any Kaggle dataset!")
    st.stop()

# ── Load Data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(uploaded_file)
clean_df = clean_data(df)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "🧹 Cleaning", "📉 Visualizations",
    "🔍 Outliers", "🤖 ML Engine", "💡 AI Insights"
])

# ════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Dataset Summary</div>', unsafe_allow_html=True)

    rows, cols = df.shape
    missing    = df.isnull().sum().sum()
    dup        = df.duplicated().sum()
    num_cols   = df.select_dtypes(include=np.number).columns.tolist()

    c1, c2, c3, c4 = st.columns(4)
    for col_ui, label, val in zip(
        [c1, c2, c3, c4],
        ["Rows", "Columns", "Missing Values", "Duplicates"],
        [rows, cols, missing, dup]
    ):
        col_ui.markdown(f"""
        <div class="metric-card">
            <h3>{label}</h3>
            <p>{val:,}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="section-title">Raw Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown('<div class="section-title">Column Info</div>', unsafe_allow_html=True)
    info_df = pd.DataFrame({
        "Column":      df.columns,
        "Type":        df.dtypes.astype(str).values,
        "Non-Null":    df.notnull().sum().values,
        "Null %":      (df.isnull().mean() * 100).round(2).astype(str).add("%").values,
        "Unique":      df.nunique().values,
    })
    st.dataframe(info_df, use_container_width=True)

# ════════════════════════════════════════════
# TAB 2 — CLEANING
# ════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Cleaning Report</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    col_a.metric("Rows Before", df.shape[0])
    col_b.metric("Rows After",  clean_df.shape[0], delta=f"{clean_df.shape[0]-df.shape[0]}")

    st.markdown("**What was done automatically:**")
    st.markdown("""
    - ✅ Numeric nulls → filled with **column mean**
    - ✅ Categorical nulls → filled with **mode (most frequent)**
    - ✅ Duplicate rows → **removed**
    - ✅ Whitespace in column names → **stripped**
    """)

    st.markdown('<div class="section-title">Cleaned Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(clean_df.head(10), use_container_width=True)

    csv_bytes = clean_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Cleaned CSV",
        data=csv_bytes,
        file_name="cleaned_dataset.csv",
        mime="text/csv",
    )

# ════════════════════════════════════════════
# TAB 3 — VISUALIZATIONS
# ════════════════════════════════════════════
with tab3:
    num_cols = clean_df.select_dtypes(include=np.number).columns.tolist()

    st.markdown('<div class="section-title">Missing Values</div>', unsafe_allow_html=True)
    fig_miss = plot_missing_values(df)  # use original to show before-cleaning state
    st.pyplot(fig_miss)

    st.markdown('<div class="section-title">Distribution Plot</div>', unsafe_allow_html=True)
    if num_cols:
        col_sel = st.selectbox("Select column", num_cols, key="dist_col")
        fig_dist = plot_distributions(clean_df, col_sel)
        st.pyplot(fig_dist)
    else:
        st.warning("No numeric columns found.")

    st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)
    if len(num_cols) >= 2:
        fig_corr = plot_correlation_heatmap(clean_df)
        st.pyplot(fig_corr)
    else:
        st.info("Need at least 2 numeric columns for correlation heatmap.")

# ════════════════════════════════════════════
# TAB 4 — OUTLIERS
# ════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Outlier Detection (IQR Method)</div>', unsafe_allow_html=True)
    num_cols = clean_df.select_dtypes(include=np.number).columns.tolist()

    if num_cols:
        out_col = st.selectbox("Select column to inspect", num_cols, key="out_col")
        outlier_df, n_outliers = detect_outliers(clean_df, out_col)
        st.metric("Outliers Detected", n_outliers)
        fig_out = plot_outliers(clean_df, out_col)
        st.pyplot(fig_out)

        if n_outliers > 0:
            st.markdown("**Outlier rows:**")
            st.dataframe(outlier_df, use_container_width=True)
    else:
        st.info("No numeric columns found.")

# ════════════════════════════════════════════
# TAB 5 — ML ENGINE
# ════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">AutoML — Model Benchmarking</div>', unsafe_allow_html=True)
    st.markdown("Select a target column and run classification or regression benchmarks across multiple models.")

    num_cols  = clean_df.select_dtypes(include=np.number).columns.tolist()
    all_cols  = clean_df.columns.tolist()

    # Show unique value counts to help user pick the right target
    col_info = {col: clean_df[col].nunique() for col in all_cols}
    col_labels = [f"{col}  ({col_info[col]} unique)" for col in all_cols]
    col_map = dict(zip(col_labels, all_cols))

    selected_label = st.selectbox("🎯 Select Target Column", col_labels, key="target")
    target_col = col_map[selected_label]
    task_type  = st.radio("Task Type", ["Classification", "Regression"], horizontal=True)

    # Warn early if column looks invalid for chosen task
    n_unique = clean_df[target_col].nunique()
    if task_type == "Classification" and n_unique < 2:
        st.error(f"❌ '{target_col}' has only {n_unique} unique value — not valid for Classification. Pick another column or switch to Regression.")
    elif task_type == "Classification" and n_unique > 20:
        st.warning(f"⚠️ '{target_col}' has {n_unique} unique values — this may be a continuous variable. Consider Regression instead.")
    elif task_type == "Regression" and n_unique < 5:
        st.warning(f"⚠️ '{target_col}' has only {n_unique} unique values — this looks categorical. Consider Classification instead.")

    if st.button("🚀 Run AutoML"):
        with st.spinner("Training models... this may take a moment"):
            try:
                results_df, best_model, fi_df = train_models(clean_df, target_col, task_type)
                st.success(f"✅ Best Model: **{best_model}**")
                st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)
                st.dataframe(results_df.style.highlight_max(axis=0, color="#14532d"), use_container_width=True)

                if fi_df is not None:
                    st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
                    fig_fi = get_feature_importance(fi_df)
                    st.pyplot(fig_fi)
            except Exception as e:
                st.error(f"AutoML error: {e}")

# ════════════════════════════════════════════
# TAB 6 — AI INSIGHTS
# ════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-title">AI-Generated Dataset Insights</div>', unsafe_allow_html=True)
    st.markdown("Uses **Mistral 7B (via OpenRouter)** to generate natural language analysis of your dataset.")

    if not groq_key:
        st.warning("⚠️ Enter your OpenRouter API key in the sidebar to enable AI Insights. Free at [openrouter.ai](https://openrouter.ai)")
    else:
        if st.button("💡 Generate AI Insights"):
            if not groq_key or groq_key.strip() == "":
                st.error("❌ Please enter your OpenRouter API key in the sidebar first.")
            else:
                with st.spinner("Asking Mistral 7B to analyze your dataset..."):
                    insights = generate_ai_insights(clean_df, groq_key.strip())
                    st.markdown(insights)

st.markdown("---")
st.markdown(
    "<center><small>Built by <b>Karthika Shree K</b> · Auto Dataset Intelligence Engine v2.0 · "
    "Streamlit + Groq + AutoML</small></center>",
    unsafe_allow_html=True
)