from pipelines.training_pipeline import loan_approval_pipeline
from mlflow import get_tracking_uri

def main():
    run = loan_approval_pipeline()
    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the experiment."
    )


if __name__ == "__main__":
    main()