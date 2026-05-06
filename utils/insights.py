import pandas as pd
import numpy as np
import requests


def generate_ai_insights(df: pd.DataFrame, api_key: str) -> str:

    api_key = api_key.strip()

    if not api_key:
        raise ValueError("API key is empty. Please enter your OpenRouter API key in the sidebar.")

    # Build compact dataset summary
    num_df = df.select_dtypes(include="number")
    corr_pairs = []
    if num_df.shape[1] >= 2:
        corr = num_df.corr().abs()
        upper = corr.where(
            np.triu(np.ones(corr.shape), k=1).astype(bool)
        )
        top = upper.stack().nlargest(5)
        corr_pairs = [f"{a} ↔ {b}: {v:.2f}" for (a, b), v in top.items()]

    summary = f"""
Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns
Columns: {list(df.columns)}
Numeric stats:
{num_df.describe().to_string()}

Top correlations: {corr_pairs if corr_pairs else 'N/A'}
Null counts (post-cleaning): {df.isnull().sum().to_dict()}
"""

    prompt = f"""You are a senior data scientist. Analyze this dataset summary and provide:

1. **Key Observations** – 3-4 bullet points about data patterns
2. **Potential Issues** – class imbalance, multicollinearity, outlier risk
3. **Feature Engineering Suggestions** – 2-3 actionable ideas
4. **Model Recommendation** – which ML algorithm would likely work best and why
5. **Business Insights** – 2-3 insights a non-technical stakeholder would care about

Dataset Summary:
{summary}

Be concise, specific, and insightful. Use markdown formatting."""

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://huggingface.co/spaces/karthi147006/dataset-intelligence",
                "X-Title": "Auto Dataset Intelligence Engine",
            },
            json={
                "model": "openrouter/free",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1200,
                "temperature": 0.4,
            },
            timeout=30,
        )
    except requests.exceptions.Timeout:
        raise Exception("Request timed out. Try again.")
    except requests.exceptions.ConnectionError:
        raise Exception("Could not connect to OpenRouter. Check your internet.")

    if response.status_code == 401:
        raise Exception("❌ Invalid API key. Get your key from openrouter.ai/keys (starts with sk-or-...)")
    elif response.status_code == 429:
        raise Exception("❌ Rate limited. Wait a minute and retry.")
    elif response.status_code != 200:
        raise Exception(f"OpenRouter API error {response.status_code}: {response.text}")

    data = response.json()
    if not data.get("choices"):
        raise Exception(f"Unexpected response: {data}")

    return data["choices"][0]["message"]["content"]