import os
import pandas as pd


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)


def save_to_parquet(
    data,
    filename="nifty500_prices.parquet"
):
    parquet_dir = os.path.join(
        PROJECT_ROOT,
        "data",
        "parquet"
    )

    os.makedirs(parquet_dir, exist_ok=True)

    file_path = os.path.join(
        parquet_dir,
        filename
    )

    data.to_parquet(
        file_path,
        index=False
    )

    return file_path


def load_from_parquet(
    filename="nifty500_prices.parquet"
):
    file_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "parquet",
        filename
    )

    return pd.read_parquet(file_path)