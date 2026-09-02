import os
import duckdb


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)


def create_connection():

    database_dir = os.path.join(
        PROJECT_ROOT,
        "database"
    )

    os.makedirs(database_dir, exist_ok=True)

    database_file = os.path.join(
        database_dir,
        "portfolio.duckdb"
    )

    return duckdb.connect(database_file)


def load_parquet_to_db(
    parquet_file=None
):

    if parquet_file is None:
        parquet_file = os.path.join(
            PROJECT_ROOT,
            "data",
            "parquet",
            "nifty500_prices.parquet"
        )

    con = create_connection()

    con.execute("""
        CREATE OR REPLACE TABLE prices AS
        SELECT *
        FROM read_parquet(?)
    """, [parquet_file])

    return con