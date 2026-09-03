import pandas as pd


def calculate_momentum(data, lookback=90):
    data = data.sort_values(["Ticker", "Date"]).copy()

    momentum = (
        data.groupby("Ticker")["Close"]
        .transform(
            lambda x: x / x.shift(lookback) - 1
        )
    )

    data["Momentum"] = momentum

    return (
        data.groupby("Ticker")["Momentum"]
        .last()
        .dropna()
    )


def calculate_volatility(data, window=90):
    data = data.sort_values(["Ticker", "Date"]).copy()

    data["Return"] = (
        data.groupby("Ticker")["Close"]
        .pct_change()
    )

    volatility = (
        data.groupby("Ticker")["Return"]
        .rolling(window)
        .std()
        .reset_index()
    )

    volatility = (
        volatility
        .groupby("Ticker")["Return"]
        .last()
        .dropna()
    )

    return volatility


def calculate_mean_return(data, window=90):
    data = data.sort_values(["Ticker", "Date"]).copy()

    data["Return"] = (
        data.groupby("Ticker")["Close"]
        .pct_change()
    )

    mean_return = (
        data.groupby("Ticker")["Return"]
        .rolling(window)
        .mean()
        .reset_index()
    )

    mean_return = (
        mean_return
        .groupby("Ticker")["Return"]
        .last()
        .dropna()
    )

    return mean_return


def momentum_selection(
    data,
    n_stocks=25,
    lookback=90
):
    momentum = calculate_momentum(
        data,
        lookback
    )

    return (
        momentum
        .sort_values(ascending=False)
        .head(n_stocks)
        .index
        .tolist()
    )


def low_volatility_selection(
    data,
    n_stocks=25,
    window=90
):
    volatility = calculate_volatility(
        data,
        window
    )

    return (
        volatility
        .sort_values(ascending=True)
        .head(n_stocks)
        .index
        .tolist()
    )


def multi_factor_selection(
    data,
    n_stocks=25,
    momentum_weight=0.5,
    low_vol_weight=0.5,
    lookback=90
):
    momentum = calculate_momentum(
        data,
        lookback
    )

    volatility = calculate_volatility(
        data,
        lookback
    )

    metrics = pd.concat(
        [
            momentum.rename("momentum"),
            volatility.rename("volatility")
        ],
        axis=1
    ).dropna()

    metrics["momentum_score"] = (
        metrics["momentum"].rank(pct=True)
    )

    metrics["low_vol_score"] = (
        1 - metrics["volatility"].rank(pct=True)
    )

    metrics["score"] = (
        momentum_weight * metrics["momentum_score"]
        + low_vol_weight * metrics["low_vol_score"]
    )

    return (
        metrics["score"]
        .sort_values(ascending=False)
        .head(n_stocks)
        .index
        .tolist()
    )


def custom_quant_selection(
    data,
    n_stocks=25,
    momentum_weight=0.4,
    low_vol_weight=0.3,
    return_weight=0.3,
    lookback=90
):
    momentum = calculate_momentum(
        data,
        lookback
    )

    volatility = calculate_volatility(
        data,
        lookback
    )

    mean_return = calculate_mean_return(
        data,
        lookback
    )

    metrics = pd.concat(
        [
            momentum.rename("momentum"),
            volatility.rename("volatility"),
            mean_return.rename("mean_return")
        ],
        axis=1
    ).dropna()

    metrics["momentum_score"] = (
        metrics["momentum"].rank(pct=True)
    )

    metrics["low_vol_score"] = (
        1 - metrics["volatility"].rank(pct=True)
    )

    metrics["return_score"] = (
        metrics["mean_return"].rank(pct=True)
    )

    metrics["score"] = (
        momentum_weight * metrics["momentum_score"]
        + low_vol_weight * metrics["low_vol_score"]
        + return_weight * metrics["return_score"]
    )

    return (
        metrics["score"]
        .sort_values(ascending=False)
        .head(n_stocks)
        .index
        .tolist()
    )


def baseline_selection(
    data,
    n_stocks=25
):
    return (
        data["Ticker"]
        .drop_duplicates()
        .sort_values()
        .head(n_stocks)
        .tolist()
    )


def select_stocks(
    data,
    method="baseline",
    n_stocks=25,
    lookback=90
):
    if method == "baseline":

        return baseline_selection(
            data,
            n_stocks=n_stocks
        )

    elif method == "momentum":

        return momentum_selection(
            data,
            n_stocks=n_stocks,
            lookback=lookback
        )

    elif method == "low_volatility":

        return low_volatility_selection(
            data,
            n_stocks=n_stocks,
            window=lookback
        )

    elif method == "multi_factor":

        return multi_factor_selection(
            data,
            n_stocks=n_stocks,
            lookback=lookback
        )

    elif method == "custom_quant":

        return custom_quant_selection(
            data,
            n_stocks=n_stocks,
            lookback=lookback
        )

    else:

        raise ValueError(
            f"Unknown selection method: {method}"
        )