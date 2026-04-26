import matplotlib.pyplot as plt

def plot_missing_values(df):
    fig, ax = plt.subplots()
    df.isnull().sum().plot(kind="bar", ax=ax)
    ax.set_title("Missing Values per Column")
    return fig


def plot_distribution(df, column):
    fig, ax = plt.subplots()
    df[column].hist(ax=ax, bins=20)
    ax.set_title(f"Distribution of {column}")
    return fig