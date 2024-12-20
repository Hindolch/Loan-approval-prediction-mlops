import pandas as pd
from src.ingest_data import DataIngestorFactory
from prefect import task

@task
def data_ingestion_task(file_path: str)->pd.DataFrame:
    """Ingests data from a zip file using the appropriate DataIngestorFactory"""
    file_extension = ".zip"

    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)
    df = data_ingestor.ingest(file_path)
    return df