from src.feature_engineering import (
    FeatureEngineer,
    LogTransformation,
    MinMaxTransformation,
    StandardScalingTransformation,
    OneHotEncodingTransformation,
    LabelEncodingTransformation
)
import pandas as pd
from prefect import task

@task
def feature_engineer_task(
    df:pd.DataFrame, strategy:str = "log", features:list=None
)->pd.DataFrame:
     # Ensuring features is a list, even if not provided
    if features is None:
        features = []
    
    if strategy == "log":
        engineer = FeatureEngineer(LogTransformation(features))
    elif strategy == "standard_scaling":
        engineer = FeatureEngineer(StandardScalingTransformation(features))
    elif strategy == "minmax_scaling":
        engineer = FeatureEngineer(MinMaxTransformation(features))
    elif strategy == "label_encoding":
        engineer = FeatureEngineer(LabelEncodingTransformation(features))
    elif strategy == "onehot_encoding":
        engineer = FeatureEngineer(OneHotEncodingTransformation(features))
    else:
        raise ValueError(f"Unsupported feature engineering strategy: {strategy}")

    transformed_df = engineer.apply_feature_engineering(df)
    print("THE FINAL DATA COLUMNS ARE", transformed_df.columns)
    return transformed_df

