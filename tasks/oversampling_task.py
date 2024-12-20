from src.oversampling import OverSampler, SmoteTransformation
import pandas as pd
from prefect import task

@task
def oversample_task(df: pd.DataFrame, strategy: str = "smote_oversample", feature: str = None) -> pd.DataFrame:
    """
    Performs oversampling using OverSampler and a specified strategy
    
    Args:
        df (pd.DataFrame): Input dataframe
        strategy (str): Oversampling strategy to use (default: "smote_oversample")
        oversample_column (str): Column to use for oversampling
        
    Returns:
        pd.DataFrame: Resampled dataframe
    """
    if feature is None:
        raise ValueError("oversample_column must be specified")
        
    # Initialize the appropriate oversampling strategy
    if strategy == "smote_oversample":
        oversampler = OverSampler(SmoteTransformation(feature=feature))
    else:
        raise ValueError(f"Unsupported oversampling strategy: {strategy}")
    
    # Execute oversampling
    resampled_df = oversampler.execute_oversampling(df)
    
    return resampled_df
