from prefect import task
from src.handle_missing_values import (
    MissingValuesHandler,
    DropMissingValuesStrategy,
    FillMissingValuesStrategy
)
import pandas as pd

@task
def handle_missing_values_task(df:pd.DataFrame, method:str="mode")->pd.DataFrame:
    if method == "drop":
        handler = MissingValuesHandler(DropMissingValuesStrategy(axis=0))
    elif method in ["mean", "mode", "median", "constant"]:
        handler = MissingValuesHandler(FillMissingValuesStrategy(method=method))
    else:
        raise ValueError(f"Unsupported missing value handling method called")

    cleaned_df = handler.handle_missing_values(df)
    return cleaned_df