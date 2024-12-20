import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score

#Abstract base class for model evaluation strategy
class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(self, model:ClassifierMixin, X_test:pd.DataFrame, y_test:pd.Series)->dict:
        pass

#Concrete strategy for RandomForest model evaluation
class RFModelEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate_model(self, model:ClassifierMixin, X_test:pd.DataFrame, y_test:pd.Series)->dict:
        logging.info("Predicting using the trained model")
        y_pred = model.predict(X_test)

        logging.info("Calculating evaluation metrics")
        accu_score = accuracy_score(y_test, y_pred)
        preci_score = precision_score(y_test, y_pred)

        metrics = {"Accuracy score": accu_score, "Precision score": preci_score}
        logging.info(f"Model Evaluation metrics: {metrics}")
        return metrics
    
#Context class for model evaluation
class ModelEvaluator:
    def __init__(self, strategy: ModelEvaluationStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy:ModelEvaluationStrategy):
        self._strategy = strategy

    def evaluate(self, model:ClassifierMixin, X_test:pd.DataFrame, y_test:pd.Series)->dict:
        logging.info("Evaluating the model using the selected strategy")
        return self._strategy.evaluate_model(model,X_test,y_test)
