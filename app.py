import streamlit as st
import pandas as pd
import pickle as pk
model= pk.load(open('model.pkl','rb'))
scaler= pk.load(open('scaler.pkl','rb'))
st.header('LOAN PREDICTION APP')
no_of_dep=st.slider('Choose No of dependents',0,5)
grad=st.selectbox('Choose Education',['GRADUATED ','NOT GRADUATED'])
self_emp=st.selectbox('SELF EMPLOYMENT STATUS',['YES','NO'])
Annual_Income=st.slider('Choose Annual Income',0,10000000)
Loan_Amount=st.slider('Choose Loan amt.',0,10000000)
Loan_Dur=st.slider('Choose loan Duration',1,20)
Cibil=st.slider('Choose cibil Score',300,1000)
Assets=st.slider('Choose Assets',0,10000000)
if grad=='GRADUATED':
    grad_s=0
else:
    grad_s=1
if self_emp=='No':
    emp_s=0
else:
    emp_s=1
if st.button("predict"):
 pred_data=pd.DataFrame([[no_of_dep,grad_s,emp_s,Annual_Income,Loan_Amount,Loan_Dur,Cibil,Assets]],columns=['no_of_dependents','education','self_employed','income_annum','loan_amount','loan_term','cibil_score','Assets'])
 pred_data=scaler.transform(pred_data)
 predict=model.predict(pred_data)
 if predict[0]==1:
    st.markdown('Loan is approved :)')
 else:
    st.markdown('Loan is Rejected !!')