from __future__ import annotations

import pandas as pd

from src.analytics.stocks.data import _stock_price_series

MA_SHORT_DAYS = 50
MA_LONG_DAYS = 200


def calculate_stock_technical_metrics(
    market_data: pd.DataFrame,
    ticker: str,
    start_date: str,
    end_date: str,
) -> dict[str, float | str]:
    prices = _stock_price_series(
        market_data,
        ticker,
        start_date,
        end_date,
    )

    returns = prices.pct_change().dropna()

    current_price = float(
        prices.iloc[-1]
    )

    recent_prices = prices.tail(252)

    high_52w = float(
        recent_prices.max()
    )

    low_52w = float(
        recent_prices.min()
    )

    distance_from_high = (
        current_price / high_52w - 1.0
        if high_52w > 0
        else float("nan")
    )

    distance_from_low = (
        current_price / low_52w - 1.0
        if low_52w > 0
        else float("nan")
    )

    ma_50 = (
        prices.rolling(MA_SHORT_DAYS).mean().iloc[-1]
    )

    ma_200 = (
        prices.rolling(MA_LONG_DAYS).mean().iloc[-1]
    )

    price_vs_50dma = (
        current_price / ma_50 - 1.0
        if pd.notna(ma_50) and ma_50 > 0
        else float("nan")
    )

    price_vs_200dma = (
        current_price / ma_200 - 1.0
        if pd.notna(ma_200) and ma_200 > 0
        else float("nan")
    )

    if pd.isna(ma_50) or pd.isna(ma_200):
        trend = "Insufficient Data"
    elif current_price > ma_50 and ma_50 > ma_200:
        trend = "Bullish"
    elif current_price < ma_50 and ma_50 < ma_200:
        trend = "Bearish"
    else:
        trend = "Mixed"

    return {
        "Current Price": current_price,
        "52W High": high_52w,
        "52W Low": low_52w,
        "Distance from 52W High":
            distance_from_high,
        "Distance from 52W Low":
            distance_from_low,
        "50 DMA": float(ma_50)
        if pd.notna(ma_50)
        else float("nan"),
        "200 DMA": float(ma_200)
        if pd.notna(ma_200)
        else float("nan"),
        "Price vs 50 DMA":
            price_vs_50dma,
        "Price vs 200 DMA":
            price_vs_200dma,
        "MA Trend":
            trend,
        "Autocorrelation Lag 1":
            float(returns.autocorr(lag=1))
            if len(returns) > 1
            else float("nan"),
        "Autocorrelation Lag 5":
            float(returns.autocorr(lag=5))
            if len(returns) > 5
            else float("nan"),
    }