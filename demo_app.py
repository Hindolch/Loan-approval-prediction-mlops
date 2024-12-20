import pandas as pd
import streamlit as st
import joblib

st.title("Loan Approval Prediction")

# Create user inputs
dependents = st.number_input("Dependents", min_value=0, max_value=10, value=2)
applicant_income = st.number_input("Applicant Income", value=5674)
coapplicant_income = st.number_input("Coapplicant Income", value=4996)
loan_amount = st.number_input("Loan Amount", value=898)
loan_amount_term = st.number_input("Loan Amount Term", value=120)

# Credit history as a more user-friendly dropdown
credit_history = st.selectbox("Credit History", ["Good", "Bad"])

# Convert credit history to numeric
credit_history_numeric = 1 if credit_history == "Good" else 0

# Convert property area to numeric
property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])
property_area_numeric = 1 if property_area == "Rural" else (2 if property_area == "Semiurban" else 3)

gender_male = st.selectbox("Gender", ["Male", "Female"])
gender_male_numeric = 1 if gender_male == "Male" else 0

married = st.selectbox("Marital Status", ["Married", "Unmarried"])
married_yes = 1 if married == "Married" else 0

education = st.selectbox("Education", ["Not Graduate", "Graduate"])
education_not_graduate = 1 if education == "Not Graduate" else 0

self_employed = st.selectbox("Employment Status", ["Self Employed", "Not Self Employed"])
self_employed_yes = 1 if self_employed == "Self Employed" else 0
if st.button('predict'):
    input_data = [[
        dependents,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_amount_term,
        credit_history_numeric,
        property_area_numeric,  # Use the converted numeric value
        gender_male_numeric,
        married_yes,
        education_not_graduate,
        self_employed_yes
    ]]
    
    input_df = pd.DataFrame(input_data, columns=[
        dependents,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_amount_term,
        credit_history_numeric,
        property_area_numeric,  # Use the converted numeric value
        gender_male_numeric,
        married_yes,
        education_not_graduate,
        self_employed_yes
    ])
    model = joblib.load('Artifacts/model.pkl')

    pred = model.predict(input_df)
    if pred == [1]:
        st.success("Your loan approval is accepted!")
    else:
        st.error("Your loan approval is rejected :(")