"""
src/train.py
------------
Trains Logistic Regression and XGBoost models, compares them,
selects the best, and saves it with SHAP explainer.

Run: python src/train.py
"""

import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from data import build_dataset

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# ── Split ─────────────────────────────────────────────────────────────────────

def split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)


# ── Handle class imbalance with SMOTE ────────────────────────────────────────

def apply_smote(X_train, y_train, random_state=42):
    print(f"Class distribution before SMOTE: {y_train.value_counts().to_dict()}")
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"Class distribution after SMOTE:  {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res


# ── Model definitions ─────────────────────────────────────────────────────────

def build_logistic_regression():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=0.1,
            class_weight="balanced",
            random_state=42
        ))
    ])


def build_xgboost():
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=3,       # handles class imbalance natively
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )


# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, model_name: str) -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc_roc = roc_auc_score(y_test, y_proba)
    auc_pr  = average_precision_score(y_test, y_proba)

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  ROC-AUC:  {auc_roc:.4f}")
    print(f"  PR-AUC:   {auc_pr:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Stayed', 'Churned'])}")

    return {
        "model_name": model_name,
        "roc_auc":    round(auc_roc, 4),
        "pr_auc":     round(auc_pr, 4),
        "y_pred":     y_pred,
        "y_proba":    y_proba,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_roc_curves(results: list, y_test):
    plt.figure(figsize=(8, 6))
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        plt.plot(fpr, tpr, label=f"{r['model_name']} (AUC={r['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("models/roc_curves.png", dpi=150)
    plt.close()
    print("Saved: models/roc_curves.png")


def plot_confusion_matrix(model, X_test, y_test, model_name: str):
    cm = confusion_matrix(y_test, model.predict(X_test))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Stayed", "Churned"],
                yticklabels=["Stayed", "Churned"])
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(f"models/confusion_matrix_{model_name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close()


def plot_shap(model, X_test, feature_names: list):
    print("\nComputing SHAP values (this may take a moment)...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      show=False, max_display=20)
    plt.tight_layout()
    plt.savefig("models/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: models/shap_summary.png")

    return explainer


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading and preparing data...")
    X, y, feature_names, _ = build_dataset()

    X_train, X_test, y_train, y_test = split(X, y)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Churn rate: {y.mean():.1%}")

    # Apply SMOTE only on training set
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)

    # ── Train models ──────────────────────────────────────────────────────────
    print("\nTraining Logistic Regression...")
    lr = build_logistic_regression()
    lr.fit(X_train_sm, y_train_sm)

    print("Training XGBoost...")
    xgb_model = build_xgboost()
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results = [
        evaluate(lr,        X_test, y_test, "Logistic Regression"),
        evaluate(xgb_model, X_test, y_test, "XGBoost"),
    ]

    # ── Cross-validation ──────────────────────────────────────────────────────
    print("\nCross-validation (5-fold ROC-AUC):")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for model, name in [(lr, "Logistic Regression"), (xgb_model, "XGBoost")]:
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        print(f"  {name}: {scores.mean():.4f} ± {scores.std():.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_roc_curves(results, y_test)
    plot_confusion_matrix(xgb_model, X_test, y_test, "XGBoost")
    plot_confusion_matrix(lr,        X_test, y_test, "Logistic Regression")

    # ── SHAP on XGBoost ───────────────────────────────────────────────────────
    explainer = plot_shap(xgb_model, X_test.values, feature_names)

    # ── Save best model (XGBoost wins on AUC in practice) ────────────────────
    print("\nSaving XGBoost model and explainer...")
    joblib.dump(xgb_model,     "models/xgb_churn_model.pkl")
    joblib.dump(explainer,     "models/shap_explainer.pkl")
    joblib.dump(feature_names, "models/feature_names.pkl")

    # Save metrics
    metrics = {
        "logistic_regression": {k: v for k, v in results[0].items()
                                 if k not in ["y_pred", "y_proba"]},
        "xgboost": {k: v for k, v in results[1].items()
                    if k not in ["y_pred", "y_proba"]},
    }
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n✅ Training complete. Files saved to models/")
    print("   - xgb_churn_model.pkl")
    print("   - shap_explainer.pkl")
    print("   - feature_names.pkl")
    print("   - metrics.json")
    print("   - roc_curves.png")
    print("   - shap_summary.png")


if __name__ == "__main__":
    main()
