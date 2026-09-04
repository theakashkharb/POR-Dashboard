from __future__ import annotations

import pandas as pd
import yfinance as yf


REQUIRED_COLUMNS = [
    "Ticker",
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]


def _normalise_downloaded_data(
    data: pd.DataFrame,
    tickers: list[str],
) -> pd.DataFrame:

    if data.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    frames = []

    # yfinance returns a MultiIndex when downloading multiple tickers.
    if isinstance(data.columns, pd.MultiIndex):

        level_0 = set(data.columns.get_level_values(0))
        level_1 = set(data.columns.get_level_values(1))

        price_columns = {"Open", "High", "Low", "Close", "Volume"}

        # group_by="ticker" -> Ticker / Price
        if price_columns.intersection(level_1):
            for ticker in tickers:

                if ticker not in level_0:
                    continue

                try:
                    ticker_data = data[ticker].copy()
                except KeyError:
                    continue

                if ticker_data.empty:
                    continue

                ticker_data = ticker_data.reset_index()

                available = [
                    column
                    for column in ["Date", "Open", "High", "Low", "Close", "Volume"]
                    if column in ticker_data.columns
                ]

                if "Date" not in available:
                    continue

                ticker_data = ticker_data[available]
                ticker_data["Ticker"] = ticker

                ticker_data = ticker_data[
                    ["Ticker", "Date", "Open", "High", "Low", "Close", "Volume"]
                ]

                frames.append(ticker_data)

        # Defensive support for Price / Ticker layout.
        else:
            for ticker in tickers:

                if ticker not in level_1:
                    continue

                try:
                    ticker_data = data.xs(ticker, axis=1, level=1).copy()
                except KeyError:
                    continue

                ticker_data = ticker_data.reset_index()

                available = [
                    column
                    for column in ["Date", "Open", "High", "Low", "Close", "Volume"]
                    if column in ticker_data.columns
                ]

                if "Date" not in available:
                    continue

                ticker_data = ticker_data[available]
                ticker_data["Ticker"] = ticker

                ticker_data = ticker_data[
                    ["Ticker", "Date", "Open", "High", "Low", "Close", "Volume"]
                ]

                frames.append(ticker_data)

    else:
        # Single ticker.
        ticker_data = data.copy().reset_index()

        available = [
            column
            for column in ["Date", "Open", "High", "Low", "Close", "Volume"]
            if column in ticker_data.columns
        ]

        if "Date" in available:
            ticker_data = ticker_data[available]
            ticker_data["Ticker"] = tickers[0]

            ticker_data = ticker_data[
                ["Ticker", "Date", "Open", "High", "Low", "Close", "Volume"]
            ]

            frames.append(ticker_data)

    if not frames:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    result = pd.concat(frames, ignore_index=True)

    result["Date"] = pd.to_datetime(result["Date"]).dt.tz_localize(None)

    for column in ["Open", "High", "Low", "Close", "Volume"]:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    result = result.drop_duplicates(
        subset=["Ticker", "Date"],
        keep="last",
    )

    result = result.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)

    return result


def download_stocks(
    tickers: list[str],
    start=None,
    end=None,
    period: str = "max",
) -> pd.DataFrame:

    tickers = list(dict.fromkeys(
        str(ticker).strip()
        for ticker in tickers
        if ticker
    ))

    if not tickers:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    try:

        kwargs = {
            "tickers": tickers,
            "auto_adjust": False,
            "progress": False,
            "threads": True,
            "group_by": "ticker",
        }

        if start is not None or end is not None:

            if start is not None:
                kwargs["start"] = pd.to_datetime(start)

            if end is not None:
                # yfinance end date is exclusive.
                kwargs["end"] = pd.to_datetime(end) + pd.Timedelta(days=1)

        else:
            kwargs["period"] = period

        data = yf.download(**kwargs)

        return _normalise_downloaded_data(
            data,
            tickers,
        )

    except Exception as e:

        print(f"BATCH DOWNLOAD FAILED: {e}")

        return pd.DataFrame(columns=REQUIRED_COLUMNS)


def download_stock(
    ticker: str,
    period: str = "20y",
) -> pd.DataFrame:

    return download_stocks(
        [ticker],
        period=period,
    )