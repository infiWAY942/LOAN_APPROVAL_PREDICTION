import streamlit as st
import pandas as pd
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt

# Load saved model and scaler
model = pk.load(open('model.pkl', 'rb'))
scaler = pk.load(open('scaler.pkl', 'rb'))

st.title('Explainable AI: Loan Approval Prediction App')

# User Inputs
no_of_dep = st.slider('Choose No of Dependents', 0, 5)
grad = st.selectbox('Choose Education', ['Graduate', 'Not Graduate'])
self_emp = st.selectbox('Self Employment Status', ['No', 'Yes'])
Annual_Income = st.slider('Choose Annual Income', 0, 10000000)
Loan_Amount = st.slider('Choose Loan Amount', 0, 10000000)
Loan_Dur = st.slider('Choose Loan Duration (Years)', 1, 20)
Cibil = st.slider('Choose CIBIL Score', 0, 900)
Assets = st.slider('Choose Total Assets', 0, 10000000)

# Match exact encoding from notebook (Graduate=1, Not Graduate=0; No=0, Yes=1)
grad_s = 1 if grad == 'Graduate' else 0
emp_s = 1 if self_emp == 'Yes' else 0

feature_names = [
    'no_of_dependents', 'education', 'self_employed',
    'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'Assets'
]

if st.button("Predict & Explain"):
    pred_data = pd.DataFrame(
        [[no_of_dep, grad_s, emp_s, Annual_Income, Loan_Amount, Loan_Dur, Cibil, Assets]],
        columns=feature_names
    )
    pred_scaled = scaler.transform(pred_data)
    prediction = model.predict(pred_scaled)

    # Note: Verify in your notebook whether 0 or 1 represents 'Approved'
    if prediction[0] == 1:
        st.success('Loan is Approved! :)')
    else:
        st.error('Loan is Rejected!!')

    # Explainable AI: Feature Contribution Chart for this Applicant
    st.subheader("Why did the AI make this decision? (Feature Impact)")
    contributions = pred_scaled[0] * model.coef_[0]
    impact_df = pd.DataFrame({'Feature': feature_names, 'Impact': contributions}).sort_values(by='Impact')

    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ['green' if val > 0 else 'red' for val in impact_df['Impact']]
    ax.barh(impact_df['Feature'], impact_df['Impact'], color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Impact on Decision (Green = Helps Approval, Red = Pushes Rejection)")
    st.pyplot(fig)