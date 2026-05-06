import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor":   "#1e293b",
    "axes.edgecolor":   "#334155",
    "axes.labelcolor":  "#94a3b8",
    "xtick.color":      "#94a3b8",
    "ytick.color":      "#94a3b8",
    "text.color":       "#f1f5f9",
})

from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import LabelEncoder
from sklearn.impute           import SimpleImputer
from sklearn.metrics          import (
    accuracy_score, f1_score, mean_squared_error, r2_score
)
from sklearn.linear_model    import LogisticRegression, LinearRegression, Ridge
from sklearn.tree            import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble        import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors       import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm             import SVC, SVR


def _preprocess(df: pd.DataFrame, target: str, task: str):
    """Encode categoricals, impute any remaining NaNs, and split into X, y."""
    df = df.copy()
    le = LabelEncoder()

    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop(columns=[target])
    y = df[target]

    # Impute any remaining NaNs
    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Impute target if regression
    if y.isnull().any():
        y = y.fillna(y.mean())

    # Validate classification target has at least 2 classes
    if task == "Classification":
        n_classes = y.nunique()
        if n_classes < 2:
            raise ValueError(
                f"Target column '{target}' has only {n_classes} unique class. "
                "Classification needs at least 2 classes. Try a different target column or switch to Regression."
            )

    # Use stratify for classification to preserve class ratios in split
    try:
        stratify = y if task == "Classification" else None
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    except ValueError:
        # Fallback without stratify if any class has too few samples
        return train_test_split(X, y, test_size=0.2, random_state=42)


def train_models(df: pd.DataFrame, target: str, task: str):
    """
    Benchmark multiple models. Returns (results_df, best_model_name, feature_importance_df).
    """
    X_train, X_test, y_train, y_test = _preprocess(df, target, task)

    if task == "Classification":
        models = {
            "Logistic Regression":    LogisticRegression(max_iter=1000),
            "Decision Tree":          DecisionTreeClassifier(random_state=42),
            "Random Forest":          RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting":      GradientBoostingClassifier(random_state=42),
            "K-Nearest Neighbors":    KNeighborsClassifier(),
            "SVM":                    SVC(),
        }
        rows = []
        fi_df = None
        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            acc  = round(accuracy_score(y_test, pred), 4)
            f1   = round(f1_score(y_test, pred, average="weighted", zero_division=0), 4)
            rows.append({"Model": name, "Accuracy": acc, "F1 Score (Weighted)": f1})

        results_df  = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
        best_model  = results_df.iloc[0]["Model"]

        # Feature importance from Random Forest
        rf = models["Random Forest"]
        fi_df = pd.DataFrame({
            "Feature":   X_train.columns,
            "Importance": rf.feature_importances_,
        }).sort_values("Importance", ascending=False).head(15)

    else:  # Regression
        models = {
            "Linear Regression":  LinearRegression(),
            "Ridge Regression":   Ridge(),
            "Decision Tree":      DecisionTreeRegressor(random_state=42),
            "Random Forest":      RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting":  GradientBoostingRegressor(random_state=42),
            "K-Nearest Neighbors": KNeighborsRegressor(),
        }
        rows = []
        fi_df = None
        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            rmse = round(np.sqrt(mean_squared_error(y_test, pred)), 4)
            r2   = round(r2_score(y_test, pred), 4)
            rows.append({"Model": name, "RMSE": rmse, "R² Score": r2})

        results_df = pd.DataFrame(rows).sort_values("R² Score", ascending=False).reset_index(drop=True)
        best_model = results_df.iloc[0]["Model"]

        rf = models["Random Forest"]
        fi_df = pd.DataFrame({
            "Feature":    X_train.columns,
            "Importance": rf.feature_importances_,
        }).sort_values("Importance", ascending=False).head(15)

    return results_df, best_model, fi_df


def get_feature_importance(fi_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, max(4, len(fi_df) * 0.4)))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(fi_df)))[::-1]
    ax.barh(fi_df["Feature"][::-1], fi_df["Importance"][::-1], color=colors)
    ax.set_title("Feature Importance (Random Forest)", fontsize=13, color="#f1f5f9")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    return fig