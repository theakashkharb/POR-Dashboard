import pandas as pd


def clean_stock(data):
    required_columns = [
        'Ticker',
        'Date',
        'Open',
        'High',
        'Low',
        'Close',
        'Volume'
    ]

    # Check required columns
    if not all(column in data.columns for column in required_columns):
        raise ValueError("Missing required columns")

    data = data.copy()

    # Sort chronologically
    data = data.sort_values('Date').reset_index(drop=True)

    # Check missing percentage for each column
    missing_pct = data[required_columns].isna().mean()

    # Drop stock if any required column has more than 5% missing data
    if missing_pct.max() > 0.05:
        return pd.DataFrame()

    # Forward-fill remaining missing values
    data = data.ffill()

    return data