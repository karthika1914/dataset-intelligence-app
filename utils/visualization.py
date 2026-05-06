import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor":   "#1e293b",
    "axes.edgecolor":   "#334155",
    "axes.labelcolor":  "#94a3b8",
    "xtick.color":      "#94a3b8",
    "ytick.color":      "#94a3b8",
    "text.color":       "#f1f5f9",
    "grid.color":       "#334155",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
})


def plot_missing_values(df: pd.DataFrame):
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    fig, ax = plt.subplots(figsize=(10, 4))
    if missing.empty:
        ax.text(0.5, 0.5, "✅ No missing values!", ha="center", va="center",
                fontsize=14, color="#4ade80")
        ax.set_axis_off()
    else:
        bars = ax.bar(missing.index, missing.values, color="#38bdf8", edgecolor="#0ea5e9")
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, color="#f1f5f9")
        ax.set_title("Missing Values per Column", fontsize=13, pad=12, color="#f1f5f9")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_distributions(df: pd.DataFrame, column: str):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(df[column].dropna(), bins=30, color="#f472b6",
            edgecolor="#be185d", alpha=0.85)
    ax.set_title(f"Distribution of '{column}'", fontsize=13, pad=10, color="#f1f5f9")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    ax.yaxis.grid(True)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame):
    num_df = df.select_dtypes(include="number")
    corr   = num_df.corr()

    fig, ax = plt.subplots(figsize=(max(8, len(corr) * 0.9), max(6, len(corr) * 0.8)))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap=cmap,
        linewidths=0.5, linecolor="#334155",
        ax=ax, cbar_kws={"shrink": 0.8},
        annot_kws={"size": 8},
    )
    ax.set_title("Correlation Heatmap", fontsize=13, pad=12, color="#f1f5f9")
    plt.tight_layout()
    return fig


def plot_outliers(df: pd.DataFrame, column: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Box plot
    axes[0].boxplot(df[column].dropna(), patch_artist=True,
                    boxprops=dict(facecolor="#1e40af", color="#93c5fd"),
                    medianprops=dict(color="#fbbf24", linewidth=2),
                    whiskerprops=dict(color="#93c5fd"),
                    capprops=dict(color="#93c5fd"),
                    flierprops=dict(marker="o", color="#f87171", alpha=0.6))
    axes[0].set_title(f"Box Plot — {column}", color="#f1f5f9")
    axes[0].set_ylabel(column)

    # Violin plot
    axes[1].violinplot(df[column].dropna(), showmedians=True)
    axes[1].set_title(f"Violin Plot — {column}", color="#f1f5f9")
    axes[1].set_ylabel(column)

    plt.tight_layout()
    return fig


def plot_class_balance(df: pd.DataFrame, column: str):
    counts = df[column].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index.astype(str), counts.values, color="#a78bfa", edgecolor="#7c3aed")
    ax.set_title(f"Class Balance — {column}", fontsize=13, color="#f1f5f9")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    plt.tight_layout()
    return fig