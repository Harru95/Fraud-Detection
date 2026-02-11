````markdown
# Fraud Detection Machine Learning Project

## 📌 Project Overview
This project focuses on detecting fraudulent financial transactions using machine learning.  
The dataset contains **6.3+ million transaction records**, and the objective was to build predictive models and derive actionable fraud prevention insights.

Two models were implemented and compared:

- Logistic Regression (baseline model)
- Random Forest Classifier (final model)

The Random Forest model achieved strong performance with **ROC-AUC ≈ 0.996** and balanced fraud detection capability.

---

## 📊 Dataset Description
The dataset includes transaction-level financial data such as:

- Transaction type (`type`)
- Transaction amount (`amount`)
- Sender & receiver balances before/after transaction
- Fraud label (`isFraud`)

Due to dataset size, optimized storage using:

```python
df.to_parquet("fraud.parquet")
```

was used for faster loading.

---

## ⚙️ Workflow

### 1. Data Cleaning
- Checked missing values (`df.isnull().sum()`)
- No missing values found.
- Outliers retained since extreme values often indicate fraud.
- Removed identifier columns:

```python
df = df.drop(columns=['nameOrig','nameDest'], errors='ignore')
```

---

### 2. Feature Engineering
Created balance difference features:

```python
df['balanceOrig_diff'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['balanceDest_diff'] = df['newbalanceDest'] - df['oldbalanceDest']
```

These capture abnormal fund movements.

---

### 3. Encoding Categorical Variables
Transaction type encoded using:

```python
df = pd.get_dummies(df, columns=['type'], drop_first=True)
```

---

### 4. Handling Class Imbalance
Used stratified train-test split:

```python
train_test_split(X, y, stratify=y)
```

and balanced class weights in models.

---

### 5. Model Training

#### Logistic Regression (Baseline)
```python
LogisticRegression(max_iter=1000, class_weight='balanced')
```

#### Random Forest (Final Model)
```python
RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    n_jobs=-1
)
```

---

## 📈 Model Performance

### Logistic Regression
- Very high recall (~98%)
- Low precision due to imbalance
- ROC-AUC ≈ 0.995

### Random Forest (Final Model)
- Precision ≈ 97%
- Recall ≈ 81%
- ROC-AUC ≈ 0.996

This indicates excellent fraud discrimination.

---

## 🔎 Feature Importance Highlights
Key predictors:

- Balance difference in origin account
- Transaction amount
- Transfer and cash-out transaction types
- Account balance changes

These align with real-world fraud behavior.

---

## 🛡️ Business Insights & Recommendations

Suggested fraud prevention strategies:

- Real-time ML fraud monitoring
- Transaction threshold alerts
- Multi-factor authentication
- Behavioral analytics
- Continuous model retraining

---

## 📊 Measuring Effectiveness

Model impact can be evaluated through:

- Fraud detection rate
- False positive rate
- Financial loss reduction
- Continuous model performance tracking

---

## 🧠 Key Learnings

- Handling highly imbalanced datasets
- Feature engineering for financial data
- Model comparison and evaluation
- Translating ML outputs into business insights

---

## 🚀 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Google Colab

---

## 📎 Notes

Dataset not included due to large size.  
Refer to original financial fraud dataset sources for reproduction.

---

## 👨‍💻 Author

Harsh Soni  
Machine Learning 
````
