# Explainable AI for Loan Approval Prediction System

An end-to-end Machine Learning and Explainable AI (XAI) web application built using Python, Scikit-Learn, XGBoost, SHAP, and Streamlit to predict loan approvals and transparently explain the reasoning behind every decision based on applicant financial profiles and credit history.

---

## 🚀 Project Overview

This project automates the loan eligibility evaluation process using supervised classification algorithms while eliminating the "black-box" nature of traditional ML models. It includes a complete machine learning pipeline—from data preprocessing, feature engineering, and exploratory data analysis (EDA) to multi-model comparison, SHAP explainability, and interactive deployment via a user-friendly Streamlit web interface.

---

## 📊 Dataset Information

* **Source:** Historical loan application records (`loan_approval_dataset.csv`, 4,269 records).
* **Key Features Used:**
  * `no_of_dependents` : Number of dependents.
  * `education` : Education level (Graduate / Not Graduate).
  * `self_employed` : Self-employment status (Yes / No).
  * `income_annum` : Annual income of the applicant.
  * `loan_amount` : Requested loan amount.
  * `loan_term` : Repayment duration in years.
  * `cibil_score` : Credit score of the applicant.
  * `Assets` : Engineered total asset value combining residential, commercial, luxury, and bank assets.

---

## 🛠️ Algorithms & Methodology

1. **Data Preprocessing & Feature Engineering:** Handled whitespace cleanup, dropped non-predictive identifiers (`loan_id`), combined four asset columns into a single `Assets` feature, and encoded categorical targets (`1` for Approved, `0` for Rejected).
2. **Exploratory Data Analysis (EDA):** Visualized loan approval class distributions, CIBIL score impact via boxplots, and feature correlations using heatmaps.
3. **Feature Scaling:** Standardized numerical features using `StandardScaler` to ensure uniform feature weighting.
4. **Multi-Model Training & Comparison:** Trained and evaluated **Logistic Regression**, **Random Forest**, and **XGBoost** classifiers on the same test split.
5. **Explainable AI (XAI):**
   * **Global & Local SHAP Analysis:** Used `shap.TreeExplainer` with XGBoost to generate global feature importance summary plots and individual applicant waterfall plots.
   * **Real-Time Web App Explainability:** Integrated an interactive **Feature Impact Bar Chart** in Streamlit (`app.py`) showing which factors pushed a decision toward approval (green) or rejection (red).
6. **Serialization:** Saved the trained model (`model.pkl`) and scaler (`scaler.pkl`) using Python's `pickle` library for real-time inference.

---

## 📈 Experimental Results & Model Comparison

Three classification algorithms were evaluated using Accuracy, Precision, Recall, and F1-Score:

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 80.09% | 80.88% | 90.56% | 85.45% |
| **Random Forest Classifier** | 97.78% | 97.33% | 99.27% | 98.29% |
| **XGBoost Classifier** | **98.71%** | **98.74%** | **99.27%** | **99.00%** |

* **Key Findings:** **XGBoost** achieved the highest overall accuracy (**98.71%**) and F1-Score (**99.00%**), while `cibil_score` and engineered `Assets` emerged as the top drivers of loan approval decisions.
* **Interactive Validation:** Validated through real-time applicant inputs and live visual explanations on the Streamlit UI.

---

## ⚙️ Instructions for Execution

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/infiWAY942/LOAN_APPROVAL_PREDICTION.git](https://github.com/infiWAY942/LOAN_APPROVAL_PREDICTION.git)
   cd LOAN_APPROVAL_PREDICTION
pip install streamlit pandas numpy scikit-learn xgboost shap matplotlib seaborn
streamlit run app.py
