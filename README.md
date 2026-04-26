# 🧠 Dataset Intelligence Engine

A Python-based web application that automatically analyzes, cleans, and visualizes datasets using Streamlit.

---

## 🚀 Features

- 📤 Upload CSV datasets instantly
- 🧹 Automatic data cleaning:
  - Removes duplicate rows
  - Handles missing values (numeric + categorical)
- 📊 Interactive data visualization:
  - Missing value analysis
  - Column-wise distribution plots
- ⚡ Simple, clean Streamlit UI
- 📁 Modular project structure for scalability

---

## 🏗️ Project Structure


DATASET-INTELLIGENCE/
│
├── app.py # Main Streamlit application
├── requirements.txt # Python dependencies
├── Procfile # Deployment start command
├── runtime.txt # Python version
├── README.md # Project documentation
│
├── utils/
│ ├── cleaning.py # Data cleaning functions
│ └── visualization.py # Visualization functions


---
## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py

---





