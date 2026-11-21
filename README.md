# Explainable AI for Loan Approval Prediction System

An end-to-end Machine Learning and Explainable AI (XAI) web application built using Python, Scikit-Learn, XGBoost, SHAP, and Streamlit to predict loan approvals and transparently explain the reasoning behind every decision based on applicant financial profiles and credit history[span_5](start_span)[span_5](end_span)[span_6](start_span)[span_6](end_span)[span_7](start_span)[span_7](end_span)[span_8](start_span)[span_8](end_span)[span_9](start_span)[span_9](end_span).

---

## 🚀 Project Overview

This project automates the loan eligibility evaluation process using supervised classification algorithms while eliminating the "black-box" nature of traditional ML models[span_10](start_span)[span_10](end_span). It includes a complete machine learning pipeline—from data preprocessing, feature engineering, and exploratory data analysis (EDA) to multi-model comparison, SHAP explainability, and interactive deployment via a user-friendly Streamlit web interface[span_11](start_span)[span_11](end_span)[span_12](start_span)[span_12](end_span).

---

## 📊 Dataset Information

* **Source:** Historical loan application records (`loan_approval_dataset.csv`, 4,269 records)[span_13](start_span)[span_13](end_span).
* **Key Features Used:**
  * `no_of_dependents` : Number of dependents[span_14](start_span)[span_14](end_span).
  * `education` : Education level (Graduate / Not Graduate)[span_15](start_span)[span_15](end_span).
  * `self_employed` : Self-employment status (Yes / No)[span_16](start_span)[span_16](end_span).
  * `income_annum` : Annual income of the applicant[span_17](start_span)[span_17](end_span).
  * `loan_amount` : Requested loan amount[span_18](start_span)[span_18](end_span)[span_19](start_span)[span_19](end_span).
  * `loan_term` : Repayment duration in years[span_20](start_span)[span_20](end_span).
  * `cibil_score` : Credit score of the applicant[span_21](start_span)[span_21](end_span).
  * `Assets` : Engineered total asset value combining residential, commercial, luxury, and bank assets[span_22](start_span)[span_22](end_span).

---

## 🛠️ Algorithms & Methodology

1. **Data Preprocessing & Feature Engineering:** Handled whitespace cleanup, dropped non-predictive identifiers (`loan_id`), combined four asset columns into a single `Assets` feature, and encoded categorical targets (`1` for Approved, `0` for Rejected)[span_23](start_span)[span_23](end_span).
2. **Exploratory Data Analysis (EDA):** Visualized loan approval class distributions, CIBIL score impact via boxplots, and feature correlations using heatmaps[span_24](start_span)[span_24](end_span).
3. **Feature Scaling:** Standardized numerical features using `StandardScaler` to ensure uniform feature weighting[span_25](start_span)[span_25](end_span).
4. **Multi-Model Training & Comparison:** Trained and evaluated **Logistic Regression**, **Random Forest**, and **XGBoost** classifiers on the same test split[span_26](start_span)[span_26](end_span).
5. **Explainable AI (XAI):**
   * **Global & Local SHAP Analysis:** Used `shap.TreeExplainer` with XGBoost to generate global feature importance summary plots and individual applicant waterfall plots[span_27](start_span)[span_27](end_span)[span_28](start_span)[span_28](end_span).
   * **Real-Time Web App Explainability:** Integrated an interactive **Feature Impact Bar Chart** in Streamlit (`app.py`) showing which factors pushed a decision toward approval (green) or rejection (red)[span_29](start_span)[span_29](end_span)[span_30](start_span)[span_30](end_span).
6. **Serialization:** Saved the trained model (`model.pkl`) and scaler (`scaler.pkl`) using Python's `pickle` library for real-time inference[span_31](start_span)[span_31](end_span).

---

## 📈 Experimental Results & Model Comparison

Three classification algorithms were evaluated using Accuracy, Precision, Recall, and F1-Score[span_32](start_span)[span_32](end_span):

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 80.09% | 80.88% | 90.56% | 85.45% |
| **Random Forest Classifier** | 97.78% | 97.33% | 99.27% | 98.29% |
| **XGBoost Classifier** | **98.71%** | **98.74%** | **99.27%** | **99.00%** |

* **Key Findings:** **XGBoost** achieved the highest overall accuracy (**98.71%**) and F1-Score (**99.00%**)[span_33](start_span)[span_33](end_span), while `cibil_score` and engineered `Assets` emerged as the top drivers of loan approval decisions[span_34](start_span)[span_34](end_span)[span_35](start_span)[span_35](end_span).
* **Interactive Validation:** Validated through real-time applicant inputs and live visual explanations on the Streamlit UI[span_36](start_span)[span_36](end_span)[span_37](start_span)[span_37](end_span)[span_38](start_span)[span_38](end_span).

---

## ⚙️ Instructions for Execution

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/infiWAY942/LOAN_APPROVAL_PREDICTION.git](https://github.com/infiWAY942/LOAN_APPROVAL_PREDICTION.git)
   cd LOAN_APPROVAL_PREDICTION
