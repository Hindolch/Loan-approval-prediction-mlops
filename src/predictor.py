

# import bentoml
# import numpy as np
# import pandas as pd
# import os
# import pytest
# import sys

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# input_df = pd.DataFrame([{
#     'Dependents': 3,
#     'ApplicantIncome': 5674,
#     'CoapplicantIncome': 4996,
#     'LoanAmount': 898,
#     'Loan_Amount_Term': 360,
#     'Credit_History': 1,
#     'Property_Area': 2,
#     'Gender_Male': 1,
#     'Married_Yes': 1,
#     'Education_Not Graduate': 1,
#     'Self_Employed_Yes': 1
# }])



# def predict_loan_inference(df):

# # Load the model reference
#     model_ref = bentoml.sklearn.get("loan_approval_model_prefect:latest")

#     # Create a Runner instance and initialize it locally
#     model_runner = model_ref.to_runner()
#     model_runner.init_local()  # Required for standalone usage outside a Bento service


#     # Make predictions
#     result = model_runner.predict.run(df)


#     #print("Prediction Result:", result)
#     if result == [1.]:
#         return "Your loan is approved"
#     else:
#         return "Not approved"



