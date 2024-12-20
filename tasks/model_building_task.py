from prefect import task
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier
import joblib
import mlflow
import os

@task
def model_building_task(X_train: pd.DataFrame, y_train: pd.Series, 
                        n_estimators=100, max_depth=20, random_state=42):
    """
    Trains a RandomForestClassifier, logs training metrics with MLflow, 
    and saves the model to a specified directory.

    Parameters:
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Training labels.
    - n_estimators (int): Number of estimators for RandomForest.
    - max_depth (int): Maximum depth of the trees.
    - random_state (int): Random state for reproducibility.

    Returns:
    - dict: Contains model path and the trained model object.
    """
    # Input validation
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")
    
    logging.info(f"Training RandomForestClassifier with {X_train.shape[0]} samples and {X_train.shape[1]} features.")
    logging.info(f"Hyperparameters: n_estimators={n_estimators}, max_depth={max_depth}, random_state={random_state}")
    
    # Define the model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    
    # Set MLflow experiment
    experiment_name = "Loan Approval Model Training"
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    with mlflow.start_run():
        try:
            # Enable MLflow autologging
            mlflow.sklearn.autolog()
            
            # Train the model
            model.fit(X_train, y_train)

            # Save the model
            os.makedirs("Artifacts", exist_ok=True)
            model_path = "Artifacts/model.pkl"
            joblib.dump(model, model_path)
            logging.info(f"Model saved at {model_path}")
            
            # Log feature importance as a JSON artifact
            feature_importances = {f"feature_{i}": imp for i, imp in enumerate(model.feature_importances_)}
            mlflow.log_dict(feature_importances, "feature_importances.json")
            
            # Return the model path and object
            return {"model_path": model_path, "model_object": model}
        
        except Exception as e:
            logging.error(f"Error during model training: {str(e)}")
            raise e
        finally:
            mlflow.end_run()