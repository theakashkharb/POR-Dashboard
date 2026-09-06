from __future__ import annotations

from pathlib import Path

import pandas as pd


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Existing data files
UNIVERSE_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nifty500_universe.csv"
)

MARKET_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "market"
    / "nifty500.parquet"
)

NIFTY50_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "market"
    / "nifty50.parquet"
)


def load_universe() -> pd.DataFrame:
    """
    Load the NIFTY 500 universe metadata.
    """
    if not UNIVERSE_FILE.exists():
        raise FileNotFoundError(
            f"Universe file not found: {UNIVERSE_FILE}"
        )

    return pd.read_csv(UNIVERSE_FILE)


def load_market_data() -> pd.DataFrame:
    """
    Load the NIFTY 500 historical market data.
    """
    if not MARKET_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Market data file not found: {MARKET_DATA_FILE}"
        )

    return pd.read_parquet(MARKET_DATA_FILE)


def load_nifty50_data() -> pd.DataFrame:
    """
    Load the NIFTY 50 historical market data.
    """
    if not NIFTY50_DATA_FILE.exists():
        raise FileNotFoundError(
            f"NIFTY 50 data file not found: {NIFTY50_DATA_FILE}"
        )

    return pd.read_parquet(NIFTY50_DATA_FILE)


def load_market_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both the NIFTY 500 universe and market data.
    """
    universe = load_universe()
    market_data = load_market_data()

    return universe, market_data