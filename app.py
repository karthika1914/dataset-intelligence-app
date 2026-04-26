import streamlit as st
import pandas as pd

from utils.cleaning import clean_data
from utils.visualization import plot_missing_values, plot_distribution

st.set_page_config(page_title="Dataset Intelligence Engine", layout="wide")

st.title("🧠 Dataset Intelligence Engine")

uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Original Data")
    st.write(df.head())

    # Cleaning
    st.subheader("🧹 Cleaned Data")
    clean_df = clean_data(df)
    st.write(clean_df.head())

    # Visualizations
    st.subheader("📉 Missing Values")
    st.pyplot(plot_missing_values(clean_df))

    column = st.selectbox("📌 Select column for distribution", clean_df.columns)

    st.subheader("📊 Distribution Plot")
    st.pyplot(plot_distribution(clean_df, column))