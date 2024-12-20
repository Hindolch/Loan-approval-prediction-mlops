from tasks.data_ingestion_task import data_ingestion_task
from tasks.handle_missing_values_task import handle_missing_values_task
from tasks.feature_engineering_task import feature_engineer_task
from tasks.oversampling_task import oversample_task
from tasks.data_splitting_task import data_splitter_task
from tasks.model_building_task import model_building_task
from tasks.model_loader import load_model
from tasks.model_evaluator_task import model_evaluation_task
from prefect import flow
import pandas as pd
import logging

# @flow
# def loan_approval_pipeline():
#     raw_data = data_ingestion_task(
#         file_path = "/home/kenzi/loan approval system zenml/data/Dataset.zip"

#     )
#     filled_data = handle_missing_values_task(raw_data)

#     engineered_data = feature_engineer_task(
#         filled_data, strategy="log", features=['LoanAmount']
#     )

#     engineered_data2 = feature_engineer_task(
#         engineered_data, strategy="onehot_encoding", features=['Gender', 'Married', 'Education', 'Self_Employed', 'Loan_Status']
#     )

#     engineered_data3 = feature_engineer_task(
#         engineered_data2, strategy="label_encoding", features=['Dependents']
#     )
#     # return engineered_data3
#     engineered_data4 = feature_engineer_task(
#         engineered_data3, strategy="label_encoding", features=['Property_Area']
#     )
    
#     oversampled_data = oversample_task(engineered_data4, feature='Loan_Status_Y')
    
#     X_train, X_test, y_train, y_test = data_splitter_task(
#         oversampled_data,
#         target_column="Loan_Status_Y"
#     )
#     model = model_building_task(X_train=X_train, y_train=y_train)
#     loaded_model = load_model(model_path='Artifacts/model.pkl')
#     # evaluation_metrics = model_evaluation_task(model=loaded_model, X_test=X_test, y_test=y_test)
#     # print("-------------------------------------------------")
#     # print("THE DATA ISSSSS", engineered_data4)
#     print("The LOADEDDD MODELLLL IS", loaded_model)
#     # print('----------------------------------------')
#     # print("THE EVALUATION METRICS ARE", evaluation_metrics)
    

# if __name__ == "__main__":
#     loan_approval_pipeline()





@flow
def loan_approval_pipeline():
    try:
        raw_data = data_ingestion_task(file_path="/home/kenzi/loan approval system zenml/data/Dataset.zip")
        filled_data = handle_missing_values_task(raw_data)

        engineered_data = feature_engineer_task(
            filled_data, strategy="log", features=['LoanAmount']
        )
        engineered_data2 = feature_engineer_task(
            engineered_data, strategy="onehot_encoding", features=['Gender', 'Married', 'Education', 'Self_Employed', 'Loan_Status']
        )
        engineered_data3 = feature_engineer_task(
            engineered_data2, strategy="label_encoding", features=['Dependents', 'Property_Area']
        )

        target_column = 'Loan_Status_Y'
        
        # Debug print to verify the column exists
        print("Columns before oversampling:", engineered_data3.columns)
        print("Target column:", target_column)
        
        # Perform oversampling
        oversampled_data = oversample_task(
            engineered_data3, 
            strategy="smote_oversample", 
            feature=target_column
        )
        
        X_train, X_test, y_train, y_test = data_splitter_task(
            oversampled_data, target_column="Loan_Status_Y"
        )
        
        # Train and save the model
        model_path = model_building_task(X_train=X_train, y_train=y_train)
        
        # Load the latest model
        loaded_model = load_model()
        print("The LOADEDDD MODELLLL IS", loaded_model)
        # Verify model existence
        print("Model saved at:", model_path['model_path'])
        # engineered_data.to_csv('datax.csv')
        logging.info("Model loaded successfully for evaluation.")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise

    