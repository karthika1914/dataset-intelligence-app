import pandas as pd
import numpy as np


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Auto-clean a DataFrame:
    - Strip whitespace from column names
    - Fill numeric nulls with column mean
    - Fill categorical nulls with column mode
    - Remove duplicate rows
    """
    df = df.copy()

    # Clean column names
    df.columns = df.columns.str.strip()

    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            df[col] = df[col].fillna(df[col].mean())
        else:
            if df[col].mode().empty:
                df[col] = df[col].fillna("Unknown")
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    df = df.drop_duplicates()
    return df


def detect_outliers(df: pd.DataFrame, column: str):
    """
    Detect outliers in a numeric column using the IQR method.
    Returns (outlier_rows_df, count).
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    mask = (df[column] < lower) | (df[column] > upper)
    return df[mask], int(mask.sum())