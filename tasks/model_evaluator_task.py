import pandas as pd
import logging
from src.model_evaluator import ModelEvaluator, RFModelEvaluationStrategy
from prefect import task
from sklearn.base import ClassifierMixin

@task
def model_evaluation_task(model: ClassifierMixin, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Task function for evaluating a trained model using a specific strategy.

    Parameters:
    model (ClassifierMixin): The trained model to evaluate.
    X_test (pd.DataFrame): The test dataset features.
    y_test (pd.Series): The true labels for the test dataset.

    Returns:
    dict: Evaluation metrics.
    """
    try:
        logging.info("Starting model evaluation task")
        
        # Initialize the evaluation context and strategy
        evaluation_strategy = RFModelEvaluationStrategy()
        evaluator = ModelEvaluator(strategy=evaluation_strategy)
        
        # Perform evaluation
        metrics = evaluator.evaluate(model=model, X_test=X_test, y_test=y_test)
        
        logging.info(f"Model evaluation completed. Metrics: {metrics}")
        return metrics
    except Exception as e:
        logging.error(f"Error during model evaluation: {str(e)}")
        raise