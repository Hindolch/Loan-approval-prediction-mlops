import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, LabelEncoder

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#Abstract base class for feature engineering strategy
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df:pd.DataFrame)->pd.DataFrame:
        pass

# Concrete strategy for log transformation
class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features

    def apply_transformation(self, df:pd.DataFrame)->pd.DataFrame:
        logging.info(f"Applying log transformation to features: {self.features}")
        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = np.log1p(df[feature])
        logging.info("Log transformation completed.")
        return df_transformed

    
# Concrete strategy for minmax scaling
class MinMaxTransformation(FeatureEngineeringStrategy):
    def __init__(self, features, feature_range=(0,1)):
        self.features = features
        self.scaler = MinMaxScaler(feature_range=feature_range)
    
    def apply_transformation(self, df:pd.DataFrame)->pd.DataFrame:
        logging.info(
            f"Applying Min-Max scaling to features: {self.features} with range {self.scaler.feature_range}"
        )
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Min-Max scaling completed")
        return df_transformed
    
# Concrete strategy for standard scaling
class StandardScalingTransformation(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features
        self.scaler = StandardScaler()

    def apply_transformation(self, df:pd.DataFrame)->pd.DataFrame:
        logging.info(f"Applying standard scaling to features: {self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Standard-scaling completed")
        return df_transformed

# Concrete strategy for One-Hot encoding
class OneHotEncodingTransformation(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features
        self.encoder = OneHotEncoder(sparse_output=False, drop="first")

    def apply_transformation(self, df:pd.DataFrame)->pd.DataFrame:
        df_transformed = df.copy()
        encoded_df = pd.DataFrame(
            self.encoder.fit_transform(df[self.features]),
            columns=self.encoder.get_feature_names_out(self.features),
        )
        df_transformed = df_transformed.drop(columns=self.features).reset_index(drop=True)
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
        return df_transformed


# Concrete strategy for label-encoding
# class LabelEncodingTransformation(FeatureEngineeringStrategy):
#     def __init__(self, features):
#         self.features = features
#         self.encoder = LabelEncoder()
    
#     def apply_transformation(self, df:pd.DataFrame)->pd.DataFrame:
#         df_transformed = df.copy()
#         for feature in self.features:
#             df_transformed[feature] = self.encoder.fit_transform(df[self.features])
#         return df_transformed
class LabelEncodingTransformation(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features
        self.encoders = {}  # Store a separate encoder for each feature
    
    def apply_transformation(self, df:pd.DataFrame)->pd.DataFrame:
        df_transformed = df.copy()
        for feature in self.features:
            # Create a new LabelEncoder for each feature
            encoder = LabelEncoder()
            # Fit and transform the specific column
            df_transformed[feature] = encoder.fit_transform(df[feature])
            # Store the encoder if needed for inverse transform later
            self.encoders[feature] = encoder
        return df_transformed
    
# Context class for feature engineering
class FeatureEngineer:
    def __init__(self, strategy:FeatureEngineeringStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy:FeatureEngineeringStrategy):
        self._strategy = strategy

    def apply_feature_engineering(self,df:pd.DataFrame)->pd.DataFrame:
        return self._strategy.apply_transformation(df)