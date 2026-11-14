# Loan Approval Prediction System

An end-to-end Machine Learning web application built using Python, Scikit-Learn, and Streamlit to predict loan approvals based on applicant financial profiles and credit history.

## 🚀 Project Overview
This project automates the loan eligibility evaluation process using classification algorithms. It includes a complete machine learning pipeline—from data preprocessing and scaling to model training and interactive deployment via a user-friendly web interface.

## 📊 Dataset Information
* **Source:** Historical loan application records (`loan_approval_dataset.csv`).
* **Key Features Used:**
  * `no_of_dependents`: Number of dependents.
  * `education`: Education level (Graduated / Not Graduated).
  * `self_employed`: Self-employment status (Yes / No).
  * `income_annum`: Annual income of the applicant.
  * `loan_amount`: Requested loan amount.
  * `loan_term`: Repayment duration in years.
  * `cibil_score`: Credit score of the applicant.
  * `Assets`: Total asset value.

## 🛠️ Algorithms & Methodology
1. **Data Preprocessing:** Handled target encoding (`Approved`/`Rejected` mapped to `1`/`0`), dropped redundant columns, and checked feature distributions.
2. **Feature Scaling:** Standardized numerical features using `StandardScaler` to ensure uniform weighting.
3. **Model Training:** Trained a **Logistic Regression** classifier to predict binary outcomes (`1` for Approved, `0` for Rejected).
4. **Serialization:** Saved the trained model (`model.pkl`) and scaler (`scaler.pkl`) using Python's `pickle` library for real-time inference.

## 📈 Experimental Results
* Achieved an accuracy score of **~91.2%** on the test dataset.
* Validated through real-time inputs on an interactive Streamlit UI.

## ⚙️ Instructions for Execution

1. Clone the repository:
   ```bash
   git clone [https://github.com/infiWAY942/LOAN_APPROVAL_PREDICTION.git](https://github.com/infiWAY942/LOAN_APPROVAL_PREDICTION.git)
   cd LOAN_APPROVAL_PREDICTION
