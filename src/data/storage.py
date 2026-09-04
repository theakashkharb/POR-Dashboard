from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_DIR = Path("data") / "market"
MARKET_DATA_FILE = DATA_DIR / "nifty500.parquet"


def ensure_storage_directory() -> None:
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def save_market_data(
    data: pd.DataFrame,
    path: Path = MARKET_DATA_FILE,
) -> None:

    if data.empty:
        raise ValueError("Cannot save empty market data.")

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = data.copy()

    data["Date"] = pd.to_datetime(
        data["Date"]
    ).dt.tz_localize(None)

    data = data.sort_values(
        ["Ticker", "Date"]
    )

    data.to_parquet(
        path,
        engine="pyarrow",
        index=False,
        compression="snappy",
    )


def load_market_data(
    path: Path = MARKET_DATA_FILE,
) -> pd.DataFrame:

    if not path.exists():
        return pd.DataFrame()

    data = pd.read_parquet(
        path,
        engine="pyarrow",
    )

    if not data.empty:
        data["Date"] = pd.to_datetime(
            data["Date"]
        ).dt.tz_localize(None)

    return data


def market_data_exists(
    path: Path = MARKET_DATA_FILE,
) -> bool:

    return path.exists()