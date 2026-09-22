# 📡 Telecom Customer Churn Prediction

End-to-end machine learning project predicting which telecom customers are likely to churn, with SHAP-based explainability and a live Streamlit prediction UI.

**Stack:** Python · scikit-learn · XGBoost · SHAP · SMOTE · Streamlit · Plotly

---

## Business Problem

> *"A telecom company loses ~26% of its customers annually. Acquiring a new customer costs 5–7× more than retaining an existing one. Which customers are at risk, and what should we do about it?"*

---

## Results

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic Regression | ~0.84 | ~0.65 |
| **XGBoost** ✅ | **~0.86** | **~0.70** |

**XGBoost selected** — higher AUC, better calibration on imbalanced classes, native feature importance via SHAP.

---

## Key Findings (EDA)

- Overall churn rate: **26.5%**
- Month-to-month contract customers churn at **~43%** vs **3%** for 2-year contracts
- Customers with **Fiber optic** internet churn more than DSL (higher price, more competition)
- Average churned customer tenure: **18 months** vs **38 months** for retained customers
- Customers with **3+ services** churn significantly less (stickiness effect)
- **Electronic check** payment method strongly associated with churn

---

## Feature Engineering

| Feature | Rationale |
|---|---|
| `revenue_per_tenure` | Total revenue / months — high = more valuable customer |
| `num_services` | Count of active services — proxy for switching cost |
| `tenure_bucket` | Early customers (0-12m) are highest risk |
| `high_spend` | Above-median monthly charge flag |
| `has_support` | Has security/backup/tech support — retention signal |

---

## Project Structure

```
telecom-churn/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # download from Kaggle
├── src/
│   ├── data.py        
│   ├── eda.py     
│   └── train.py       
├── app/
│   └── app.py         
├── models/            
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
Go to: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` and place it in `data/`

### 3. Run EDA
```bash
python src/eda.py
```

### 4. Train the model
```bash
python src/train.py
```

### 5. Launch the app
```bash
streamlit run app/app.py
```

---

## Model Decision: Why XGBoost over Logistic Regression?

| Factor | Logistic Regression | XGBoost |
|---|---|---|
| Non-linear relationships | ❌ Assumes linear | ✅ Handles automatically |
| Feature interactions | ❌ Manual engineering needed | ✅ Learned automatically |
| Class imbalance | Requires SMOTE | ✅ `scale_pos_weight` parameter |
| Interpretability | ✅ Coefficients | ✅ SHAP values |
| ROC-AUC | ~0.84 | ~0.86 |

Both models are interpretable — Logistic Regression via coefficients, XGBoost via SHAP. XGBoost wins on performance with minimal tradeoff in explainability.

---

## Author
Dave Avina· [GitHub](https:/srdave3/github.com)

