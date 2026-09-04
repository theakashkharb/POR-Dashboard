import numpy as np
import pandas as pd


# ============================================================
# SIMPLE RETURNS
# ============================================================

def calculate_simple_returns(data):

    data = data.copy()

    required_columns = ["Ticker", "Close"]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    data["Close"] = pd.to_numeric(
        data["Close"],
        errors="coerce"
    )

    if data["Close"].isna().any():
        raise ValueError(
            "Close contains non-numeric or missing values."
        )

    data = data.sort_values(
        ["Ticker", "Date"]
    )

    data["Return"] = (
        data
        .groupby("Ticker")["Close"]
        .pct_change()
    )

    return data


# ============================================================
# LOG RETURNS
# ============================================================

def calculate_log_returns(data):

    data = data.copy()

    required_columns = ["Ticker", "Close"]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    data["Close"] = pd.to_numeric(
        data["Close"],
        errors="coerce"
    )

    if data["Close"].isna().any():
        raise ValueError(
            "Close contains non-numeric or missing values."
        )

    if (data["Close"] <= 0).any():
        raise ValueError(
            "Close prices must be positive."
        )

    data = data.sort_values(
        ["Ticker", "Date"]
    )

    data["Log_Return"] = (
        data
        .groupby("Ticker")["Close"]
        .transform(
            lambda x: np.log(x / x.shift(1))
        )
    )

    return data


# ============================================================
# RETURN MATRIX
# ============================================================

def create_return_matrix(data):

    data = data.copy()

    required_columns = [
        "Date",
        "Ticker",
        "Close"
    ]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    data["Close"] = pd.to_numeric(
        data["Close"],
        errors="coerce"
    )

    if data["Close"].isna().any():
        raise ValueError(
            "Close contains missing/non-numeric values."
        )

    data = data.sort_values(
        ["Ticker", "Date"]
    )

    data["Return"] = (
        data
        .groupby("Ticker")["Close"]
        .pct_change()
    )

    matrix = (
        data
        .pivot(
            index="Date",
            columns="Ticker",
            values="Return"
        )
        .sort_index()
    )

    matrix = matrix.dropna(how="all")
    matrix = matrix.dropna(how="any")

    return matrix


# ============================================================
# LOG RETURN MATRIX
# ============================================================

def create_log_return_matrix(data):

    data = data.copy()

    required_columns = [
        "Date",
        "Ticker",
        "Close"
    ]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    data["Close"] = pd.to_numeric(
        data["Close"],
        errors="coerce"
    )

    if data["Close"].isna().any():
        raise ValueError(
            "Close contains missing/non-numeric values."
        )

    if (data["Close"] <= 0).any():
        raise ValueError(
            "Close prices must be positive."
        )

    data = data.sort_values(
        ["Ticker", "Date"]
    )

    data["Log_Return"] = (
        data
        .groupby("Ticker")["Close"]
        .transform(
            lambda x: np.log(x / x.shift(1))
        )
    )

    matrix = (
        data
        .pivot(
            index="Date",
            columns="Ticker",
            values="Log_Return"
        )
        .sort_index()
    )

    matrix = matrix.dropna(how="all")
    matrix = matrix.dropna(how="any")

    return matrix


# ============================================================
# CUMULATIVE RETURNS
# ============================================================

def calculate_cumulative_returns(data):

    data = data.copy()

    if "Return" not in data.columns:
        raise ValueError(
            "Return column is required."
        )

    data["Return"] = pd.to_numeric(
        data["Return"],
        errors="coerce"
    )

    data["Return"] = (
        data.groupby("Ticker")["Return"]
        .transform(
            lambda x: x.fillna(0)
        )
    )

    data["Cumulative_Return"] = (
        data.groupby("Ticker")["Return"]
        .transform(
            lambda x: (1 + x).cumprod() - 1
        )
    )

    return data


# ============================================================
# ROLLING RETURNS
# ============================================================

def calculate_rolling_returns(
    data,
    window=90,
):
    """
    Calculate rolling cumulative simple return
    over the selected number of observations.
    """

    data = data.copy()

    required_columns = [
        "Date",
        "Ticker",
        "Close",
    ]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    data["Close"] = pd.to_numeric(
        data["Close"],
        errors="coerce"
    )

    data = data.sort_values(
        ["Ticker", "Date"]
    )

    data["Rolling_Return"] = (
        data
        .groupby("Ticker")["Close"]
        .transform(
            lambda x: (
                x / x.shift(window) - 1
            )
        )
    )

    return data[
        [
            "Date",
            "Ticker",
            "Rolling_Return",
        ]
    ].dropna()


# ============================================================
# HISTORICAL EXPECTED RETURN
# ============================================================

def calculate_historical_expected_return(
    returns,
    annualization=252
):

    returns = returns.copy()

    if not isinstance(
        returns,
        pd.DataFrame
    ):
        raise TypeError(
            "returns must be a pandas DataFrame."
        )

    numeric_returns = returns.apply(
        pd.to_numeric,
        errors="coerce"
    )

    if numeric_returns.isna().any().any():
        raise ValueError(
            "returns contain NaN or non-numeric values."
        )

    return (
        numeric_returns.mean()
        * annualization
    )


# ============================================================
# GEOMETRIC EXPECTED RETURN
# ============================================================

def calculate_geometric_expected_return(
    returns,
    annualization=252
):

    returns = returns.copy()

    if not isinstance(
        returns,
        pd.DataFrame
    ):
        raise TypeError(
            "returns must be a pandas DataFrame."
        )

    numeric_returns = returns.apply(
        pd.to_numeric,
        errors="coerce"
    )

    if numeric_returns.isna().any().any():
        raise ValueError(
            "returns contain NaN or non-numeric values."
        )

    if (numeric_returns <= -1).any().any():
        raise ValueError(
            "Returns must be greater than -100%."
        )

    periods = len(
        numeric_returns
    )

    if periods == 0:
        raise ValueError(
            "returns cannot be empty."
        )

    return (
        (1 + numeric_returns).prod()
        ** (annualization / periods)
        - 1
    )


# ============================================================
# RETURN STATISTICS
# ============================================================

def calculate_return_statistics(data):

    data = data.copy()

    required_columns = [
        "Ticker",
        "Return"
    ]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    data["Return"] = pd.to_numeric(
        data["Return"],
        errors="coerce"
    )

    data = data.dropna(
        subset=[
            "Ticker",
            "Return"
        ]
    )

    if data.empty:
        raise ValueError(
            "No valid return observations available."
        )

    statistics = (
        data
        .groupby("Ticker")["Return"]
        .agg(
            mean="mean",
            std="std",
            min="min",
            max="max",
            median="median"
        )
    )

    return statistics