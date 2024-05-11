import streamlit as st
import requests
import pandas as pd 
import joblib
import numpy as np 

st.header("Welcome")
Algorithm = st.selectbox("Choose Algorithm:", ["DecisionTree", "knn", "random_forest"])
transaction_type = st.selectbox('Type', ['Payment', 'Transfer', 'Debit', 'Cash out'])
amount = st.text_input('Amount', '')
old_balance = st.text_input('Old Balance Orig', '')
new_balance = st.text_input('New Balance Orig', '')
ml = joblib.load(open("stroke", "rb"))

type_mapping = {'Payment': 0, 'Transfer': 1, 'Debit': 2, 'Cash out': 3}

def predict(type, amount, oldbalanceOrg, newbalanceOrig):
    
    type_numeric = type_mapping[type]
    
    
    amount = float(amount)
    oldbalanceOrg = float(oldbalanceOrg)
    newbalanceOrig = float(newbalanceOrig)
    
    features = np.array([type_numeric, amount, oldbalanceOrg, newbalanceOrig]).reshape(1, -1)
    prediction = ml.predict(features)
    return prediction
data=pd.read_csv('Final_data.csv')
df=data.__dataframe__(data)
btn = st.button("Try Algorithm")
if btn:
    
    sample_prediction = predict(transaction_type, amount, old_balance, new_balance)
    st.write(sample_prediction)
