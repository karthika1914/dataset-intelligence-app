import pandas as pd

def clean_data(df):
    df = df.copy()

    for col in df.columns:

        # numeric columns → fill with mean
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].mean(), inplace=True)

        # non-numeric columns → fill with mode (most common value)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # remove duplicates
    df = df.drop_duplicates()

    return df