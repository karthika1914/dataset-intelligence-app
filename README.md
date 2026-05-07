# 🧠 Auto Dataset Intelligence Engine

> **AI-powered automated data analysis, visualization, and machine learning platform — built with Streamlit, Groq LLaMA 3, and scikit-learn.**

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Hugging%20Face%20Spaces-blue?logo=streamlit)](https://huggingface.co/spaces/karthi147006/dataset-intelligence)
[![Python](https://img.shields.io/badge/Python-3.10+-green?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---
DEMO : https://dataset-intelligence-app-wjcuxwfe2d63xg4nhbgy4v.streamlit.app/
## 🚀 Overview

The **Auto Dataset Intelligence Engine** is a no-code data science platform where users can upload any CSV file and instantly receive:

- 📊 Automated **EDA** (Exploratory Data Analysis)
- 🧹 Intelligent **data cleaning** (null imputation, deduplication)
- 📉 Rich **visualizations** (distributions, heatmaps, outlier plots)
- 🤖 **AutoML** benchmarking across 6 models (classification & regression)
- 💡 **AI-generated insights** powered by Groq LLaMA 3

---

## ✨ Features

| Feature | Description |
|---|---|
| **Smart Cleaning** | Auto-fills nulls (mean/mode), strips duplicates, cleans column names |
| **Distribution Plots** | Histogram for any numeric column |
| **Correlation Heatmap** | Seaborn heatmap for all numeric features |
| **Outlier Detection** | IQR-based detection with box + violin plots |
| **AutoML Engine** | Trains 6 models, ranks by Accuracy/F1 or RMSE/R² |
| **Feature Importance** | Random Forest feature importance chart |
| **AI Insights** | Groq LLaMA 3 generates natural language dataset analysis |
| **Download Cleaned CSV** | One-click download of the cleaned dataset |

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** scikit-learn (LR, DT, RF, GBM, KNN, SVM/SVR)
- **AI / LLM:** Groq API (LLaMA 3 8B)
- **Deployment:** Hugging Face Spaces

---

## 📸 Screenshots

> Upload any CSV → Get instant analysis, visualizations, and model results.

*(Add screenshots of the app tabs here)*

---

## 🏃 Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/auto-dataset-engine
cd auto-dataset-engine

pip install -r requirements.txt

# Optional: add Groq key for AI insights
echo "GROQ_API_KEY=your_key_here" > .env

streamlit run app.py
```

---

## 📁 Project Structure

```
auto-dataset-engine/
├── app.py                  # Main Streamlit app
├── requirements.txt
├── utils/
│   ├── __init__.py
│   ├── cleaning.py         # Data cleaning + outlier detection
│   ├── visualization.py    # All chart functions
│   ├── ml_engine.py        # AutoML benchmarking
│   └── insights.py         # Groq LLM insights
└── README.md
```

---

## 🎯 Use Cases

- **Students** exploring a new dataset before modeling
- **Data analysts** who want instant EDA without writing code
- **Researchers** benchmarking multiple ML models quickly

---

## 👩‍💻 Author

**Karthika Shree K**  
B.Tech Data Science & AI — Dr. MGR Educational and Research Institute, Chennai  
[GitHub](https://github.com/karthi147006) · [LinkedIn](https://linkedin.com/in/YOUR_PROFILE) · [Hugging Face](https://huggingface.co/karthi147006)

---

## 📄 License

MIT License — free to use, modify, and distribute.
