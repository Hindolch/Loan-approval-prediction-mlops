import joblib
from prefect import task
import glob
import os
import logging

@task
def load_model():
    """
    Load the most recently created model from the Artifacts directory.
    
    Returns:
    - The loaded model object
    
    Raises:
    - FileNotFoundError if no model is found
    """
    # Ensure the Artifacts directory exists
    artifacts_dir = 'Artifacts'
    if not os.path.exists(artifacts_dir):
        raise FileNotFoundError(f"Artifacts directory {artifacts_dir} does not exist")

    # Get all .pkl files in the Artifacts directory
    model_files = glob.glob(os.path.join(artifacts_dir, 'model*.pkl'))
    
    # Add debug logging
    logging.info(f"Found model files: {model_files}")
    
    if not model_files:
        raise FileNotFoundError("No model found in the Artifacts directory")
    
    # Get the most recently created file
    latest_model = max(model_files, key=os.path.getctime)
    
    logging.info(f"Loading latest model: {latest_model}")
    
    try:
        model = joblib.load(latest_model)
        logging.info("Model loaded successfully")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise