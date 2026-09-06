from __future__ import annotations

import pandas as pd

ROLLING_6M_DAYS = 126


def calculate_rolling_beta(
    stock_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = ROLLING_6M_DAYS,
) -> pd.Series:
    combined = pd.concat(
        [
            stock_returns.rename("Stock"),
            benchmark_returns.rename("Benchmark"),
        ],
        axis=1,
    ).dropna()

    if combined.empty:
        return pd.Series(
            dtype=float,
            name="Rolling Beta",
        )

    covariance = (
        combined["Stock"]
        .rolling(window)
        .cov(combined["Benchmark"])
    )

    benchmark_variance = (
        combined["Benchmark"]
        .rolling(window)
        .var()
    )

    beta = covariance / benchmark_variance
    beta.name = "Rolling Beta"

    return beta.dropna()


def calculate_benchmark_correlation(
    stock_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    combined = pd.concat(
        [
            stock_returns.rename("Stock"),
            benchmark_returns.rename("Benchmark"),
        ],
        axis=1,
    ).dropna()

    if len(combined) < 2:
        return float("nan")

    return float(
        combined["Stock"].corr(
            combined["Benchmark"]
        )
    )


def calculate_up_down_capture(
    stock_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> dict[str, float]:
    combined = pd.concat(
        [
            stock_returns.rename("Stock"),
            benchmark_returns.rename("Benchmark"),
        ],
        axis=1,
    ).dropna()

    if combined.empty:
        return {
            "Up Capture": float("nan"),
            "Down Capture": float("nan"),
        }

    up = combined[
        combined["Benchmark"] > 0
    ]

    down = combined[
        combined["Benchmark"] < 0
    ]

    if up.empty or up["Benchmark"].mean() == 0:
        up_capture = float("nan")
    else:
        up_capture = float(
            up["Stock"].mean()
            / up["Benchmark"].mean()
        )

    if down.empty or down["Benchmark"].mean() == 0:
        down_capture = float("nan")
    else:
        down_capture = float(
            down["Stock"].mean()
            / down["Benchmark"].mean()
        )

    return {
        "Up Capture": up_capture,
        "Down Capture": down_capture,
    }


def calculate_stock_relative_metrics(
    stock_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> dict[str, float]:
    capture = calculate_up_down_capture(
        stock_returns,
        benchmark_returns,
    )

    return {
        "Benchmark Correlation":
            calculate_benchmark_correlation(
                stock_returns,
                benchmark_returns,
            ),
        "Up Capture":
            capture["Up Capture"],
        "Down Capture":
            capture["Down Capture"],
    }