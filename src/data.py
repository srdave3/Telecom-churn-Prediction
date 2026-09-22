"""
src/data.py
-----------
Loads the IBM Telco Customer Churn dataset, cleans it, and engineers features.
Dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")


# ── Load ──────────────────────────────────────────────────────────────────────

def load_raw() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {RAW_PATH}.\n"
            "Download it from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
            "and place it in the data/ folder."
        )
    df = pd.read_csv(RAW_PATH)
    return df


# ── Clean ─────────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TotalCharges has spaces instead of NaN for new customers
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["MonthlyCharges"], inplace=True)

    # Drop customerID — not a feature
    df.drop(columns=["customerID"], inplace=True)

    # Binary target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Convert SeniorCitizen from 0/1 int to Yes/No to match other binary cols
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Revenue per month of tenure — high value = customer pays well over time
    df["revenue_per_tenure"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"]
    )

    # 2. Number of services subscribed — more services = more sticky
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["num_services"] = df[service_cols].apply(
        lambda row: sum(v not in ["No", "No internet service", "No phone service"] for v in row),
        axis=1
    )

    # 3. Tenure bucket — early customers churn more
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12m", "13-24m", "25-48m", "49-72m"],
        include_lowest=True
    ).astype(str)

    # 4. High monthly spend flag
    df["high_spend"] = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)

    # 5. Has support services (security + backup + tech support)
    support_cols = ["OnlineSecurity", "OnlineBackup", "TechSupport"]
    df["has_support"] = df[support_cols].apply(
        lambda row: int(any(v == "Yes" for v in row)), axis=1
    )

    return df


# ── Encode ────────────────────────────────────────────────────────────────────

def encode(df: pd.DataFrame):
    """
    One-hot encode categoricals, return X (features) and y (target).
    Also returns feature names for SHAP plots.
    """
    target = "Churn"
    y = df[target]
    X = df.drop(columns=[target])

    # Identify categorical columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    X = X.astype(float)

    return X, y, X.columns.tolist()


# ── Full pipeline ─────────────────────────────────────────────────────────────

def build_dataset():
    df = load_raw()
    df = clean(df)
    df = engineer_features(df)
    X, y, feature_names = encode(df)
    return X, y, feature_names, df
