import logging
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from abc import abstractmethod, ABC

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#Abstract base class for random oversampling strategy
class OverSamplingStrategy(ABC):
    @abstractmethod
    def apply_oversampling(self, df:pd.DataFrame)->pd.DataFrame:
        pass

#Concrete strategy for oversampling
class SmoteTransformation(OverSamplingStrategy):
    def __init__(self, feature):
        self.feature = feature

    def apply_oversampling(self, df:pd.DataFrame)->pd.DataFrame:
        x = df.drop(columns=[self.feature])
        y = df[self.feature]
        smote = SMOTE(random_state=42)
        x_balanced, y_balanced = smote.fit_resample(x,y)
        df_oversampled = pd.concat([pd.DataFrame(x_balanced, columns=x_balanced.columns),
                                    pd.Series(y_balanced, name=self.feature)], axis=1)
        return df_oversampled
    
#Context class for oversampling
class OverSampler:
    def __init__(self, strategy:OverSamplingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy:OverSamplingStrategy):
        self._strategy =  strategy
    
    def execute_oversampling(self, df:pd.DataFrame)->pd.DataFrame:
        return self._strategy.apply_oversampling(df)
    