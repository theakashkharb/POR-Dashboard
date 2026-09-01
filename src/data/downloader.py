import yfinance as yf
import pandas as pd


def download_stock(ticker, period='5y'):
    data = yf.download(
        ticker,
        period=period,
        auto_adjust=False,
        multi_level_index=False,
        progress=False
    )

    if data.empty:
        return pd.DataFrame()

    data = data.reset_index()

    data = data[
        ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    ]

    data['Ticker'] = ticker

    data = data[
        ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    ]

    return data