import joblib
import pandas as pd
import pytest

input_df = pd.DataFrame([{
     'Dependents': 3,
    'ApplicantIncome': 5674,
    'CoapplicantIncome': 4996,
    'LoanAmount': 898,
    'Loan_Amount_Term': 360,
    'Credit_History': 1,
    'Property_Area': 2,
    'Gender_Male': 1,
    'Married_Yes': 1,
    'Education_Not Graduate': 1,
    'Self_Employed_Yes': 1
    
}])

model_path = 'Artifacts/model.pkl'
model = joblib.load(model_path)


def test_inference(input_df):
    result = model.predict(input_df)
    assert result == [1.]