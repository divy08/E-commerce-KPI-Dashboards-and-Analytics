import numpy as np
import pandas as pd
import streamlit as st
import pickle

import pickle

# Open the file in binary read mode
with open(r'C:\Users\Lenovo\Desktop\CDAC DBDA\final\myenv\models\decisionTree (1).pkl', 'rb') as file:
    loaded_model = pickle.load(file)





# Title
st.sidebar.title('Navigation')
st.sidebar.radio('Go to', ['RFM Analysis', 'Churn Prediction', 'Customer Retention Suggestions'])

st.title('Churn Prediction')

# Input fields
freight_value = st.number_input('Enter freight value:', min_value=0.0, step=1.0)
price = st.number_input('Enter price:', min_value=0.0, step=1.0)
monetary_value = st.number_input('Enter monetary value:', min_value=0.0, step=1.0)
review_score = st.number_input('Enter review score: Range 1-5', min_value=1.0, max_value=5.0, step=1.0)
payment_installments = st.number_input('Enter payment installments:', min_value=0.0, step=1.0)
customer_state = st.number_input('Enter customer state: Centralwest:0, Northeastern:1, Northern:2, Southeastern:3, Southern:4', min_value=0, max_value=4, step=1)
frequency = st.number_input('Enter frequency:', min_value=0.0, step=1.0)

# Display input data
input_data = {
    'freight_value': [freight_value],
    'price': [price],
    'monetary_value': [monetary_value],
    'payment_installments': [payment_installments],
    'review_score': [review_score],
    
    'customer_state': [customer_state],
    'customer_tenure': [frequency]
}

input_df = pd.DataFrame(input_data)
# Check the feature names the model was trained with
print("Feature names the model was trained with:", loaded_model.feature_names_in_)
st.subheader('Input Data:')
st.write(input_df)

# Predict churn
prediction = loaded_model.predict(input_df)
prediction_proba = loaded_model.predict_proba(input_df)

# Display prediction
if prediction[0] == 1:
    st.subheader('Churn Prediction:')
    st.write('The customer is predicted to churn.')
else:
    st.subheader('Churn Prediction:')
    st.write('The customer is predicted to not churn.')
#print("Expected features:", loaded_model.feature_names_in_)
#print("Input features:", input_df.columns)
st.write(f"Prediction Probability: {prediction_proba[0][prediction[0]]}")
st.write(f"Prediction Probability: {prediction_proba[0][prediction[0]]}")

# Additional debugging information
print("Prediction:", prediction)
print("Prediction Probability:", prediction_proba)
