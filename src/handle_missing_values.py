import pandas as pd
import logging
from abc import ABC, abstractmethod

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#Abstract base class for handling missing values strategy
class MissingValuesHandlingStrategy(ABC):
    @abstractmethod
    def handle(self,df:pd.DataFrame)->pd.DataFrame:
        pass

#Concrete strategy for Dropping missing values
class DropMissingValuesStrategy(MissingValuesHandlingStrategy):
    def __init__(self, axis=0, thresh=None):
        self.axis = axis
        self.thresh = thresh

    def handle(self, df:pd.DataFrame)->pd.DataFrame:
        logging.info(f"Dropping missing values with axis={self.axis}")
        df_cleaned = df.dropna(axis=self.axis, thresh=self.thresh)
        logging.info("Missing values dropped")
        return df_cleaned
    
#Concrete strategy for filling missing values
class FillMissingValuesStrategy(MissingValuesHandlingStrategy):
    def __init__(self, method="mean", fill_value=None):
        self.method = method
        self.fill_value = fill_value

    def handle(self, df:pd.DataFrame)->pd.DataFrame:
        logging.info(f"Filling the missing values with the strategy={self.method}")

        df_cleaned = df.copy()
        if self.method == "mean":
            numeric_columns = df.select_dtypes(include=["number", "float"]).columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].mean()
            )
        elif self.method == "median":
            numeric_columns = df.select_dtypes(include=["number", "float"]).columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].median()
            )
        elif self.method == "mode":
            for column in df_cleaned.columns:
                df_cleaned[column].fillna(df[column].mode().iloc[0],inplace=True)
        elif self.method == "constant":
            df_cleaned = df_cleaned.fillna(self.fill_value)
        else:
            logging.warning(f"Unknown method '{self.method}'. No missing values filled")
        logging.info("Missing values filled")
        return df_cleaned
    

#Context class for handling missing values
class MissingValuesHandler:
    def __init__(self, strategy:MissingValuesHandlingStrategy):
        self._strategy=strategy

    def set_strategy(self, strategy:MissingValuesHandlingStrategy):
        logging.info("Switching missing value handling strategy")
        self._strategy = strategy
    
    def handle_missing_values(self, df:pd.DataFrame)->pd.DataFrame:
        logging.info("Executing missing value handling strategy")
        return self._strategy.handle(df)