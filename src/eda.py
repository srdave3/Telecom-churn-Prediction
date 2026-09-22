"""
src/eda.py
----------
Exploratory Data Analysis — generates all plots used in the README
and notebook. Run standalone to produce PNG outputs.

Run: python src/eda.py
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from data import load_raw, clean, engineer_features

OUTPUT = Path("models")
OUTPUT.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")


def plot_churn_rate(df):
    rate = df["Churn"].value_counts(normalize=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Stayed", "Churned"], rate.values * 100, color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("Percentage (%)")
    ax.set_title(f"Overall Churn Rate: {rate[1]:.1%}")
    for i, v in enumerate(rate.values * 100):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT / "eda_churn_rate.png", dpi=150)
    plt.close()
    print("Saved: eda_churn_rate.png")


def plot_churn_by_contract(df):
    ct = df.groupby("Contract")["Churn"].mean().reset_index()
    ct["Churn"] = ct["Churn"] * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=ct, x="Contract", y="Churn", ax=ax, palette="muted")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_title("Churn Rate by Contract Type")
    plt.tight_layout()
    plt.savefig(OUTPUT / "eda_contract.png", dpi=150)
    plt.close()
    print("Saved: eda_contract.png")


def plot_tenure_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, label, color in zip(axes, [0, 1], ["#4C72B0", "#DD8452"]):
        subset = df[df["Churn"] == label]["tenure"]
        ax.hist(subset, bins=30, color=color, edgecolor="white")
        ax.set_title(f"Tenure Distribution — {'Churned' if label else 'Stayed'}")
        ax.set_xlabel("Tenure (months)")
        ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT / "eda_tenure.png", dpi=150)
    plt.close()
    print("Saved: eda_tenure.png")


def plot_monthly_charges(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, color, name in [(0, "#4C72B0", "Stayed"), (1, "#DD8452", "Churned")]:
        subset = df[df["Churn"] == label]["MonthlyCharges"]
        ax.hist(subset, bins=40, alpha=0.6, color=color, label=name, edgecolor="white")
    ax.set_xlabel("Monthly Charges ($)")
    ax.set_ylabel("Count")
    ax.set_title("Monthly Charges by Churn Status")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "eda_monthly_charges.png", dpi=150)
    plt.close()
    print("Saved: eda_monthly_charges.png")


def plot_num_services(df):
    ct = df.groupby("num_services")["Churn"].mean().reset_index()
    ct["Churn"] = ct["Churn"] * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.lineplot(data=ct, x="num_services", y="Churn", marker="o", ax=ax)
    ax.set_xlabel("Number of Services Subscribed")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_title("More Services = Lower Churn (Stickiness Effect)")
    plt.tight_layout()
    plt.savefig(OUTPUT / "eda_num_services.png", dpi=150)
    plt.close()
    print("Saved: eda_num_services.png")


def plot_correlation_heatmap(df):
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, ax=ax, linewidths=0.5)
    ax.set_title("Correlation Matrix — Numeric Features")
    plt.tight_layout()
    plt.savefig(OUTPUT / "eda_correlation.png", dpi=150)
    plt.close()
    print("Saved: eda_correlation.png")


def run_eda():
    print("Running EDA...")
    df = load_raw()
    df = clean(df)
    df = engineer_features(df)
    df["Churn"] = df["Churn"].astype(int)

    plot_churn_rate(df)
    plot_churn_by_contract(df)
    plot_tenure_distribution(df)
    plot_monthly_charges(df)
    plot_num_services(df)
    plot_correlation_heatmap(df)

    print("\nKey findings:")
    print(f"  Overall churn rate:         {df['Churn'].mean():.1%}")
    print(f"  Month-to-month churn rate:  {df[df['Contract']=='Month-to-month']['Churn'].mean():.1%}")
    print(f"  Two-year contract churn:    {df[df['Contract']=='Two year']['Churn'].mean():.1%}")
    print(f"  Avg tenure (churned):       {df[df['Churn']==1]['tenure'].mean():.1f} months")
    print(f"  Avg tenure (stayed):        {df[df['Churn']==0]['tenure'].mean():.1f} months")
    print(f"  Avg monthly charge (churned): ${df[df['Churn']==1]['MonthlyCharges'].mean():.2f}")
    print(f"  Avg monthly charge (stayed):  ${df[df['Churn']==0]['MonthlyCharges'].mean():.2f}")


if __name__ == "__main__":
    run_eda()
