import pandas as pd
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod

#Abstract base class for data splitting strategy
class DataSplittingStrategy(ABC):
    @abstractmethod
    def split_data(self, df:pd.DataFrame, target_column:str):
        pass

#Concrete strategy for simple Train-Test split
class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, df:pd.DataFrame, target_column:str):
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=self.test_size, random_state=self.random_state)
        return X_train,X_test, y_train, y_test
    

#Context class for data splitting
class DataSplitter:
    def __init__(self, strategy:DataSplittingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy:DataSplittingStrategy):
        self._strategy = strategy

    def split(self, df:pd.DataFrame, target_column:str):
        return self._strategy.split_data(df,target_column)