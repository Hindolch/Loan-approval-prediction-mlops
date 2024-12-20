from prefect import task
from src.outlier_detection import (
    OutlierDetector,
    ZScoreOutlierDetection,
    IQROutlierDetection
)
import logging
import pandas as pd

@task
def outlier_detection_step(df:pd.DataFrame, column_name:str)->pd.DataFrame:
    """Detects and removes outliers using OutlierDetector"""
    logging.info(f"Starting outlier detection step with DataFrame of shape: {df.shape}")

    if df is None:
        logging.error("Recieved a NoneType DataFrame")
        raise ValueError("Input df must be a non-null pandas DataFrame")

    if not isinstance(df, pd.DataFrame):
        logging.error(f"Expected pandas DataFrame, got {type(df)} instead.")
        raise ValueError("Input df must be a pandas dataframe")

    if column_name not in df.columns:
        logging.error(f"Column '{column_name}' does not exist in the dataframe")
        raise ValueError(f"Column '{column_name}' does not exist in the dataframe")
        # Ensure only numeric columns are passed
    df_numeric = df.select_dtypes(include=[int, float])

    outlier_detector = OutlierDetector(ZScoreOutlierDetection(threshold=3))
    outliers = outlier_detector.detect_outliers(df_numeric)
    df_cleaned = outlier_detector.handle_outliers(df_numeric, method="cap")
    return df_cleaned