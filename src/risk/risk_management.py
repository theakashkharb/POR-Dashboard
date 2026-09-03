"""
Portfolio Risk Management Engine
---------------------------------

18 distinct risk-management mechanisms for POR-Dashboard.

Categories
----------
1. Exposure & Position Control
2. Volatility Management
3. Drawdown / Capital Protection
4. Risk Budgeting
5. Tail Risk
6. Market / Factor Risk

The functions are designed to be independently testable and later exposed
through the Streamlit dashboard.

Important design principle:
Some risk controls directly modify portfolio exposure, while others return
risk diagnostics/constraint violations. A risk metric such as VaR or
correlation does not automatically imply one unique reallocation algorithm.
"""

import numpy as np
import pandas as pd


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def _validate_weights(weights):
    """Validate and return weights as a float Series."""
    if isinstance(weights, pd.Series):
        w = weights.astype(float).copy()
    elif isinstance(weights, dict):
        w = pd.Series(weights, dtype=float)
    else:
        w = pd.Series(weights, dtype=float)

    if w.empty:
        raise ValueError("weights cannot be empty")

    if not np.all(np.isfinite(w.values)):
        raise ValueError("weights must contain only finite values")

    return w


def _validate_returns(returns):
    """Validate a return Series/DataFrame."""
    if isinstance(returns, pd.Series):
        r = pd.to_numeric(returns, errors="coerce").astype(float)
    elif isinstance(returns, pd.DataFrame):
        r = returns.apply(pd.to_numeric, errors="coerce").astype(float)
    else:
        raise TypeError("returns must be a pandas Series or DataFrame")

    if r.empty:
        raise ValueError("returns cannot be empty")

    if not np.all(np.isfinite(r.to_numpy())):
        raise ValueError("returns must contain only finite values")

    return r


def _validate_covariance(covariance, assets=None):
    """Validate covariance matrix and optionally align it to assets."""
    if isinstance(covariance, pd.DataFrame):
        cov = covariance.astype(float).copy()
    else:
        cov = pd.DataFrame(np.asarray(covariance, dtype=float))

    if cov.empty:
        raise ValueError("covariance cannot be empty")

    if cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance matrix must be square")

    if not np.all(np.isfinite(cov.values)):
        raise ValueError("covariance must contain only finite values")

    if assets is not None:
        assets = list(assets)

        if isinstance(covariance, pd.DataFrame):
            missing = [
                asset
                for asset in assets
                if asset not in cov.index or asset not in cov.columns
            ]

            if missing:
                raise ValueError(
                    f"covariance missing assets: {missing}"
                )

            cov = cov.loc[assets, assets]

    if not np.allclose(
        cov.values,
        cov.values.T,
        atol=1e-10,
    ):
        raise ValueError("covariance matrix must be symmetric")

    return cov


def _validate_positive_parameter(value, name):
    """Validate a strictly positive scalar."""
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive")


def _normalize_long_only(weights):
    """Normalize non-negative weights to sum to one."""
    w = _validate_weights(weights)

    if (w < 0).any():
        raise ValueError(
            "long-only risk management requires non-negative weights"
        )

    total = float(w.sum())

    if total <= 0:
        raise ValueError(
            "weights must have positive total"
        )

    return w / total


def _portfolio_volatility(
    weights,
    covariance,
    annualization=252,
):
    """Calculate annualized portfolio volatility."""
    w = _validate_weights(weights)
    cov = _validate_covariance(covariance, w.index)

    variance = float(
        w.values @ cov.values @ w.values
    )

    if variance < -1e-12:
        raise ValueError(
            "portfolio variance cannot be negative"
        )

    return float(
        np.sqrt(max(variance, 0.0) * annualization)
    )


def _scale_exposure(
    weights,
    scale,
    cash_name="CASH",
):
    """
    Scale risky exposure and place unused exposure into cash.

    Example:
        scale = 1.0 -> 100% risky assets
        scale = 0.5 -> 50% risky assets + 50% cash
        scale = 0.25 -> 25% risky assets + 75% cash
    """
    w = _validate_weights(weights)

    scale = float(scale)

    if not np.isfinite(scale) or scale < 0:
        raise ValueError(
            "scale must be finite and non-negative"
        )

    if cash_name in w.index:
        raise ValueError(
            f"cash_name '{cash_name}' conflicts with asset name"
        )

    risky = w * scale

    if scale <= 1.0:
        result = risky.copy()
        result[cash_name] = 1.0 - scale
        return result

    return risky


# ============================================================================
# 1. MAXIMUM POSITION WEIGHT
# ============================================================================

def maximum_position_weight(
    weights,
    max_weight=0.20,
    iterations=100,
):
    """
    Cap individual asset weights and redistribute excess proportionally.

    Parameters
    ----------
    weights : Series/dict/array
    max_weight : float
        Maximum allowed weight for one asset.

    Returns
    -------
    Series
        Adjusted long-only weights summing to 1.
    """
    _validate_positive_parameter(
        max_weight,
        "max_weight",
    )

    if max_weight > 1:
        raise ValueError(
            "max_weight cannot exceed 1"
        )

    if iterations < 1:
        raise ValueError(
            "iterations must be >= 1"
        )

    result = _normalize_long_only(weights)

    if len(result) * max_weight < 1 - 1e-12:
        raise ValueError(
            "maximum position weight is infeasible for "
            "the number of assets"
        )

    for _ in range(int(iterations)):
        excess = float(
            (result - max_weight)
            .clip(lower=0)
            .sum()
        )

        if excess <= 1e-12:
            break

        result = result.clip(
            upper=max_weight
        )

        eligible = (
            result < max_weight - 1e-12
        )

        eligible_total = float(
            result.loc[eligible].sum()
        )

        if eligible_total <= 1e-12:
            raise ValueError(
                "unable to redistribute excess weight"
            )

        result.loc[eligible] += (
            excess
            * result.loc[eligible]
            / eligible_total
        )

    result = result.clip(
        lower=0,
        upper=max_weight,
    )

    return result / result.sum()


# ============================================================================
# 2. MAXIMUM SECTOR / INDUSTRY EXPOSURE
# ============================================================================

def maximum_group_exposure(
    weights,
    group_data,
    max_group_weight=0.35,
    iterations=100,
):
    """
    Control exposure to groups such as sectors or industries.

    Parameters
    ----------
    weights : Series
    group_data : Series/dict
        Asset -> group mapping.
    max_group_weight : float
        Maximum exposure to any group.

    Returns
    -------
    Series
        Adjusted weights.
    """
    _validate_positive_parameter(
        max_group_weight,
        "max_group_weight",
    )

    if max_group_weight > 1:
        raise ValueError(
            "max_group_weight cannot exceed 1"
        )

    groups = pd.Series(group_data)

    w = _normalize_long_only(weights)

    groups = groups.reindex(w.index)

    if groups.isna().any():
        missing = list(
            groups.index[groups.isna()]
        )

        raise ValueError(
            f"group_data missing assets: {missing}"
        )

    unique_groups = groups.unique()

    if len(unique_groups) * max_group_weight < 1 - 1e-12:
        raise ValueError(
            "group exposure constraint is infeasible"
        )

    result = w.copy()

    for _ in range(int(iterations)):
        exposure = result.groupby(groups).sum()

        violating = (
            exposure > max_group_weight + 1e-12
        )

        if not violating.any():
            break

        excess = (
            exposure[violating]
            - max_group_weight
        )

        violating_assets = groups.isin(
            exposure.index[violating]
        )

        # Cap violating groups.
        for group in exposure.index[violating]:
            mask = groups == group

            group_total = float(
                result.loc[mask].sum()
            )

            if group_total > max_group_weight:
                result.loc[mask] *= (
                    max_group_weight
                    / group_total
                )

        total_excess = float(excess.sum())

        eligible = ~violating_assets

        eligible_total = float(
            result.loc[eligible].sum()
        )

        if eligible_total <= 1e-12:
            raise ValueError(
                "no eligible assets available for redistribution"
            )

        result.loc[eligible] += (
            total_excess
            * result.loc[eligible]
            / eligible_total
        )

        result = result.clip(lower=0)
        result /= result.sum()

    return result / result.sum()


def maximum_industry_exposure(
    weights,
    industry_data,
    max_industry_weight=0.35,
):
    """Maximum industry exposure."""
    return maximum_group_exposure(
        weights,
        industry_data,
        max_group_weight=max_industry_weight,
    )


def maximum_sector_exposure(
    weights,
    sector_data,
    max_sector_weight=0.35,
):
    """Maximum sector exposure."""
    return maximum_group_exposure(
        weights,
        sector_data,
        max_group_weight=max_sector_weight,
    )


# ============================================================================
# 3. GROSS / NET EXPOSURE CONTROL
# ============================================================================

def control_gross_net_exposure(
    weights,
    max_gross_exposure=1.0,
    min_net_exposure=None,
    max_net_exposure=1.0,
    cash_name="CASH",
):
    """
    Control gross and net exposure.

    Gross exposure = sum(abs(weights))
    Net exposure   = sum(weights)

    Long-only portfolios have gross = net.
    """
    _validate_positive_parameter(
        max_gross_exposure,
        "max_gross_exposure",
    )

    if not np.isfinite(max_net_exposure):
        raise ValueError(
            "max_net_exposure must be finite"
        )

    if (
        min_net_exposure is not None
        and not np.isfinite(min_net_exposure)
    ):
        raise ValueError(
            "min_net_exposure must be finite"
        )

    if (
        min_net_exposure is not None
        and min_net_exposure > max_net_exposure
    ):
        raise ValueError(
            "min_net_exposure cannot exceed max_net_exposure"
        )

    w = _validate_weights(weights)

    gross = float(
        np.abs(w).sum()
    )

    net = float(
        w.sum()
    )

    if gross <= 1e-16:
        raise ValueError(
            "portfolio has zero gross exposure"
        )

    scale = min(
        1.0,
        max_gross_exposure / gross,
    )

    if net > max_net_exposure and net > 0:
        scale = min(
            scale,
            max_net_exposure / net,
        )

    adjusted = w * scale

    if (
        min_net_exposure is not None
        and adjusted.sum() < min_net_exposure - 1e-12
    ):
        raise ValueError(
            "minimum net exposure is incompatible "
            "with the requested limits"
        )

    if (
        scale < 1.0
        and (w >= 0).all()
    ):
        adjusted[cash_name] = (
            1.0 - scale
        )

    return {
        "weights": adjusted,
        "gross_exposure": float(
            np.abs(
                adjusted.drop(
                    labels=[cash_name],
                    errors="ignore",
                )
            ).sum()
        ),
        "net_exposure": float(
            adjusted.drop(
                labels=[cash_name],
                errors="ignore",
            ).sum()
        ),
        "scaling_factor": float(scale),
    }


# ============================================================================
# 4. LEVERAGE CONTROL
# ============================================================================

def control_leverage(
    weights,
    max_leverage=1.0,
    cash_name="CASH",
):
    """
    Control portfolio gross leverage.
    """
    _validate_positive_parameter(
        max_leverage,
        "max_leverage",
    )

    w = _validate_weights(weights)

    gross = float(
        np.abs(w).sum()
    )

    if gross <= max_leverage + 1e-12:
        return {
            "weights": w.copy(),
            "leverage": gross,
            "scaling_factor": 1.0,
        }

    scale = (
        max_leverage / gross
    )

    adjusted = w * scale

    if (w >= 0).all():
        adjusted[cash_name] = (
            1.0 - scale
        )

    return {
        "weights": adjusted,
        "leverage": float(
            np.abs(
                adjusted.drop(
                    labels=[cash_name],
                    errors="ignore",
                )
            ).sum()
        ),
        "scaling_factor": float(scale),
    }


# ============================================================================
# 5. VOLATILITY TARGETING
# ============================================================================

def volatility_targeting(
    weights,
    covariance,
    target_volatility=0.10,
    annualization=252,
    max_leverage=1.0,
    cash_name="CASH",
):
    """
    Scale portfolio exposure toward a target volatility.
    """
    _validate_positive_parameter(
        target_volatility,
        "target_volatility",
    )

    _validate_positive_parameter(
        max_leverage,
        "max_leverage",
    )

    current_vol = _portfolio_volatility(
        weights,
        covariance,
        annualization,
    )

    if current_vol <= 1e-16:
        raise ValueError(
            "current portfolio volatility is effectively zero"
        )

    raw_scale = (
        target_volatility
        / current_vol
    )

    applied_scale = min(
        raw_scale,
        max_leverage,
    )

    adjusted = _scale_exposure(
        weights,
        applied_scale,
        cash_name=cash_name,
    )

    return {
        "weights": adjusted,
        "current_volatility": float(current_vol),
        "target_volatility": float(target_volatility),
        "raw_scaling_factor": float(raw_scale),
        "applied_scaling_factor": float(applied_scale),
        "max_leverage": float(max_leverage),
    }


# ============================================================================
# 6. VOLATILITY LIMIT / CAP
# ============================================================================

def volatility_limit(
    weights,
    covariance,
    max_volatility=0.15,
    annualization=252,
    cash_name="CASH",
):
    """
    Reduce exposure only if portfolio volatility exceeds the cap.
    """
    _validate_positive_parameter(
        max_volatility,
        "max_volatility",
    )

    current_vol = _portfolio_volatility(
        weights,
        covariance,
        annualization,
    )

    if current_vol <= max_volatility:
        scale = 1.0
    else:
        scale = (
            max_volatility
            / current_vol
        )

    adjusted = _scale_exposure(
        weights,
        scale,
        cash_name=cash_name,
    )

    return {
        "weights": adjusted,
        "current_volatility": float(current_vol),
        "max_volatility": float(max_volatility),
        "scaling_factor": float(scale),
        "triggered": bool(scale < 1.0),
    }


# ============================================================================
# 7. DYNAMIC VOLATILITY SCALING
# ============================================================================

def dynamic_volatility_scaling(
    weights,
    volatility,
    target_volatility=0.10,
    min_scale=0.25,
    max_scale=1.0,
    cash_name="CASH",
):
    """
    Dynamically scale exposure according to current volatility.
    """
    _validate_positive_parameter(
        volatility,
        "volatility",
    )

    _validate_positive_parameter(
        target_volatility,
        "target_volatility",
    )

    if not (
        0 < min_scale <= max_scale
    ):
        raise ValueError(
            "require 0 < min_scale <= max_scale"
        )

    raw_scale = (
        target_volatility
        / volatility
    )

    scale = float(
        np.clip(
            raw_scale,
            min_scale,
            max_scale,
        )
    )

    adjusted = _scale_exposure(
        weights,
        scale,
        cash_name=cash_name,
    )

    return {
        "weights": adjusted,
        "current_volatility": float(volatility),
        "target_volatility": float(target_volatility),
        "raw_scaling_factor": float(raw_scale),
        "applied_scaling_factor": scale,
        "min_scale": float(min_scale),
        "max_scale": float(max_scale),
    }


# ============================================================================
# 8. EWMA RISK SCALING
# ============================================================================

def ewma_volatility(
    returns,
    span=60,
    annualization=252,
):
    """
    Calculate annualized EWMA volatility.
    """
    _validate_positive_parameter(
        span,
        "span",
    )

    r = _validate_returns(returns)

    if isinstance(r, pd.DataFrame):
        result = (
            r.ewm(
                span=int(span),
                adjust=False,
            )
            .std()
            .iloc[-1]
            * np.sqrt(annualization)
        )

        return result

    return float(
        r.ewm(
            span=int(span),
            adjust=False,
        )
        .std()
        .iloc[-1]
        * np.sqrt(annualization)
    )


def ewma_risk_scaling(
    weights,
    returns,
    target_volatility=0.10,
    span=60,
    annualization=252,
    min_scale=0.25,
    max_scale=1.0,
    cash_name="CASH",
):
    """
    Estimate recent volatility using EWMA and dynamically scale exposure.
    """
    r = _validate_returns(returns)

    if isinstance(r, pd.DataFrame):
        w = _validate_weights(
            weights
        ).reindex(r.columns)

        if w.isna().any():
            raise ValueError(
                "weights must contain every return-series asset"
            )

        portfolio_returns = (
            r.mul(w, axis=1)
            .sum(axis=1)
        )
    else:
        portfolio_returns = r

    vol = ewma_volatility(
        portfolio_returns,
        span=span,
        annualization=annualization,
    )

    result = dynamic_volatility_scaling(
        weights,
        vol,
        target_volatility=target_volatility,
        min_scale=min_scale,
        max_scale=max_scale,
        cash_name=cash_name,
    )

    result["ewma_span"] = int(span)

    return result


# ============================================================================
# 9. MAXIMUM DRAWDOWN CONTROL
# ============================================================================

def calculate_drawdown(nav):
    """Calculate drawdown from the historical high-water mark."""
    n = pd.Series(
        nav,
        dtype=float,
    )

    if n.empty:
        raise ValueError(
            "nav cannot be empty"
        )

    if not np.all(
        np.isfinite(n.values)
    ):
        raise ValueError(
            "nav must contain only finite values"
        )

    if (n <= 0).any():
        raise ValueError(
            "nav must be strictly positive"
        )

    high_water_mark = n.cummax()

    return (
        n / high_water_mark
        - 1.0
    )


def maximum_drawdown_control(
    weights,
    nav,
    max_drawdown=0.15,
    de_risk_scale=0.50,
    cash_name="CASH",
):
    """
    Reduce exposure once current drawdown reaches the limit.
    """
    _validate_positive_parameter(
        max_drawdown,
        "max_drawdown",
    )

    if not (
        0 <= de_risk_scale <= 1
    ):
        raise ValueError(
            "de_risk_scale must be between 0 and 1"
        )

    drawdown = calculate_drawdown(
        nav
    )

    current_drawdown = float(
        drawdown.iloc[-1]
    )

    triggered = (
        abs(current_drawdown)
        >= max_drawdown
    )

    scale = (
        de_risk_scale
        if triggered
        else 1.0
    )

    adjusted = _scale_exposure(
        weights,
        scale,
        cash_name=cash_name,
    )

    return {
        "weights": adjusted,
        "current_drawdown": current_drawdown,
        "max_drawdown": float(max_drawdown),
        "scaling_factor": float(scale),
        "triggered": bool(triggered),
    }


# ============================================================================
# 10. HIGH-WATER-MARK DE-RISKING
# ============================================================================

def high_water_mark_derisking(
    weights,
    nav,
    drawdown_start=0.05,
    drawdown_full=0.20,
    minimum_scale=0.25,
    cash_name="CASH",
):
    """
    Gradually reduce exposure as drawdown increases.

    Drawdown <= drawdown_start:
        100% exposure

    Drawdown >= drawdown_full:
        minimum_scale exposure

    Between the two:
        linear scaling
    """
    _validate_positive_parameter(
        drawdown_start,
        "drawdown_start",
    )

    _validate_positive_parameter(
        drawdown_full,
        "drawdown_full",
    )

    if drawdown_full <= drawdown_start:
        raise ValueError(
            "drawdown_full must exceed drawdown_start"
        )

    if not (
        0 <= minimum_scale <= 1
    ):
        raise ValueError(
            "minimum_scale must be between 0 and 1"
        )

    current_dd = abs(
        float(
            calculate_drawdown(nav).iloc[-1]
        )
    )

    if current_dd <= drawdown_start:
        scale = 1.0

    elif current_dd >= drawdown_full:
        scale = minimum_scale

    else:
        progress = (
            current_dd - drawdown_start
        ) / (
            drawdown_full
            - drawdown_start
        )

        scale = (
            1.0
            - progress
            * (1.0 - minimum_scale)
        )

    adjusted = _scale_exposure(
        weights,
        scale,
        cash_name=cash_name,
    )

    return {
        "weights": adjusted,
        "current_drawdown": -current_dd,
        "scaling_factor": float(scale),
        "drawdown_start": float(drawdown_start),
        "drawdown_full": float(drawdown_full),
        "minimum_scale": float(minimum_scale),
    }


# ============================================================================
# 11. TREND / MOVING-AVERAGE DE-RISKING
# ============================================================================

def trend_derisking(
    weights,
    price_series,
    moving_average_window=200,
    risk_on_scale=1.0,
    risk_off_scale=0.50,
    cash_name="CASH",
):
    """
    Reduce exposure when price falls below its moving average.
    """
    _validate_positive_parameter(
        moving_average_window,
        "moving_average_window",
    )

    if not (
        0 <= risk_off_scale <= risk_on_scale
    ):
        raise ValueError(
            "require 0 <= risk_off_scale <= risk_on_scale"
        )

    prices = pd.Series(
        price_series,
        dtype=float,
    )

    if len(prices) < moving_average_window:
        raise ValueError(
            "not enough observations for moving average"
        )

    if not np.all(
        np.isfinite(prices.values)
    ):
        raise ValueError(
            "price_series must contain finite values"
        )

    if (prices <= 0).any():
        raise ValueError(
            "price_series must contain positive prices"
        )

    moving_average = float(
        prices
        .rolling(
            int(moving_average_window)
        )
        .mean()
        .iloc[-1]
    )

    current_price = float(
        prices.iloc[-1]
    )

    risk_on = (
        current_price >= moving_average
    )

    scale = (
        risk_on_scale
        if risk_on
        else risk_off_scale
    )

    adjusted = _scale_exposure(
        weights,
        scale,
        cash_name=cash_name,
    )

    return {
        "weights": adjusted,
        "current_price": current_price,
        "moving_average": moving_average,
        "risk_on": bool(risk_on),
        "scaling_factor": float(scale),
    }


# ============================================================================
# 12. RISK CONTRIBUTION CONSTRAINT
# ============================================================================

def calculate_risk_contributions(
    weights,
    covariance,
):
    """
    Calculate marginal risk, component risk and percentage contribution.
    """
    w = _validate_weights(weights)

    cov = _validate_covariance(
        covariance,
        w.index,
    )

    variance = float(
        w.values
        @ cov.values
        @ w.values
    )

    if variance <= 1e-16:
        raise ValueError(
            "portfolio variance is effectively zero"
        )

    volatility = np.sqrt(
        variance
    )

    marginal = (
        cov.values @ w.values
        / volatility
    )

    component = (
        w.values * marginal
    )

    total_component = float(
        component.sum()
    )

    if abs(total_component) <= 1e-16:
        raise ValueError(
            "component risk sums to zero"
        )

    contribution = (
        component
        / total_component
    )

    return pd.DataFrame(
        {
            "weight": w.values,
            "marginal_risk": marginal,
            "component_risk": component,
            "risk_contribution": contribution,
        },
        index=w.index,
    )


def risk_contribution_constraint(
    weights,
    covariance,
    max_risk_contribution=0.25,
    iterations=100,
):
    """
    Heuristic risk-contribution cap.

    Assets whose risk contribution exceeds the cap are gradually reduced and
    the freed weight is redistributed to the remaining assets.

    This is intentionally a risk-control heuristic, not a replacement for
    the portfolio optimization engine.
    """
    _validate_positive_parameter(
        max_risk_contribution,
        "max_risk_contribution",
    )

    if max_risk_contribution > 1:
        raise ValueError(
            "max_risk_contribution cannot exceed 1"
        )

    if iterations < 1:
        raise ValueError(
            "iterations must be >= 1"
        )

    result = _normalize_long_only(
        weights
    )

    cov = _validate_covariance(
        covariance,
        result.index,
    )

    for _ in range(int(iterations)):
        table = calculate_risk_contributions(
            result,
            cov,
        )

        violating = (
            table["risk_contribution"]
            > max_risk_contribution + 1e-10
        )

        if not violating.any():
            break

        violating_assets = table.index[
            violating
        ]

        reduction = (
            result.loc[violating_assets]
            * 0.10
        )

        total_reduction = float(
            reduction.sum()
        )

        if total_reduction <= 1e-14:
            break

        result.loc[violating_assets] -= (
            reduction
        )

        eligible = (
            ~result.index.isin(
                violating_assets
            )
        )

        eligible_total = float(
            result.loc[eligible].sum()
        )

        if eligible_total <= 1e-14:
            raise ValueError(
                "risk contribution constraint is infeasible"
            )

        result.loc[eligible] += (
            total_reduction
            * result.loc[eligible]
            / eligible_total
        )

        result = result.clip(
            lower=0
        )

        result /= result.sum()

    return result


# ============================================================================
# 13. RISK BUDGETING
# ============================================================================

def risk_budgeting(
    covariance,
    risk_budgets,
    iterations=1000,
    tolerance=1e-8,
):
    """
    Construct a portfolio whose risk contributions approximately match
    predefined risk budgets.

    Example:
        Technology  -> 25% risk budget
        Financials  -> 25%
        Energy      -> 20%
        Others      -> 30%

    risk_budgets are specified per asset.
    """
    cov = _validate_covariance(
        covariance
    )

    budgets = _validate_weights(
        risk_budgets
    ).reindex(cov.index)

    if budgets.isna().any():
        raise ValueError(
            "risk_budgets must contain every covariance asset"
        )

    if (budgets <= 0).any():
        raise ValueError(
            "risk budgets must be strictly positive"
        )

    budgets = (
        budgets
        / budgets.sum()
    )

    weights = pd.Series(
        1.0 / len(cov),
        index=cov.index,
    )

    for _ in range(int(iterations)):
        table = calculate_risk_contributions(
            weights,
            cov,
        )

        current_rc = (
            table["risk_contribution"]
        )

        error = float(
            np.max(
                np.abs(
                    current_rc.values
                    - budgets.values
                )
            )
        )

        if error < tolerance:
            break

        ratio = (
            budgets.values
            / np.maximum(
                current_rc.values,
                1e-16,
            )
        )

        weights *= np.sqrt(
            ratio
        )

        weights = (
            weights
            / weights.sum()
        )

    return weights


# ============================================================================
# 14. MAXIMUM RISK CONCENTRATION
# ============================================================================

def risk_concentration(
    weights,
    covariance,
    max_concentration=0.25,
):
    """
    Calculate concentration of risk contributions.

    Herfindahl-style measure:

        H = sum(RC_i^2)

    Lower H means more diversified risk contribution.
    """
    _validate_positive_parameter(
        max_concentration,
        "max_concentration",
    )

    w = _normalize_long_only(
        weights
    )

    cov = _validate_covariance(
        covariance,
        w.index,
    )

    table = calculate_risk_contributions(
        w,
        cov,
    )

    concentration = float(
        (
            table["risk_contribution"]
            ** 2
        ).sum()
    )

    return {
        "risk_contribution_table": table,
        "risk_concentration": concentration,
        "limit": float(max_concentration),
        "within_limit": bool(
            concentration
            <= max_concentration
        ),
    }


# ============================================================================
# 15. VaR CONSTRAINT
# ============================================================================

def historical_var(
    returns,
    confidence=0.95,
):
    """
    Historical Value-at-Risk.

    Returned as a positive loss number.

    Example:
        VaR = 0.025
        means approximately 2.5% loss at the selected confidence level.
    """
    if not (
        0 < confidence < 1
    ):
        raise ValueError(
            "confidence must be between 0 and 1"
        )

    r = _validate_returns(
        returns
    )

    if isinstance(r, pd.DataFrame):
        r = r.sum(axis=1)

    return float(
        -np.quantile(
            r.values,
            1.0 - confidence,
        )
    )


def parametric_var(
    returns,
    confidence=0.95,
):
    """
    Parametric normal-distribution VaR.
    """
    if not (
        0 < confidence < 1
    ):
        raise ValueError(
            "confidence must be between 0 and 1"
        )

    r = _validate_returns(
        returns
    )

    if isinstance(r, pd.DataFrame):
        r = r.sum(axis=1)

    mu = float(
        r.mean()
    )

    sigma = float(
        r.std(ddof=1)
    )

    if sigma <= 0:
        return max(
            0.0,
            -mu,
        )

    from statistics import NormalDist

    z = NormalDist().inv_cdf(
        confidence
    )

    return float(
        -(mu - z * sigma)
    )


def monte_carlo_var(
    returns,
    confidence=0.95,
    simulations=10000,
    random_state=42,
):
    """
    Empirical Monte Carlo VaR using bootstrap resampling.
    """
    if not (
        0 < confidence < 1
    ):
        raise ValueError(
            "confidence must be between 0 and 1"
        )

    if simulations < 100:
        raise ValueError(
            "simulations must be >= 100"
        )

    r = _validate_returns(
        returns
    )

    if isinstance(r, pd.DataFrame):
        r = r.sum(axis=1)

    rng = np.random.default_rng(
        random_state
    )

    simulated = rng.choice(
        r.values,
        size=int(simulations),
        replace=True,
    )

    return float(
        -np.quantile(
            simulated,
            1.0 - confidence,
        )
    )


def var_constraint(
    returns,
    max_var,
    method="historical",
    confidence=0.95,
    simulations=10000,
    random_state=42,
):
    """
    Check whether VaR is within the specified risk limit.
    """
    _validate_positive_parameter(
        max_var,
        "max_var",
    )

    method = method.lower()

    if method == "historical":
        var = historical_var(
            returns,
            confidence,
        )

    elif method == "parametric":
        var = parametric_var(
            returns,
            confidence,
        )

    elif method in {
        "monte_carlo",
        "monte-carlo",
    }:
        var = monte_carlo_var(
            returns,
            confidence,
            simulations,
            random_state,
        )

    else:
        raise ValueError(
            "method must be "
            "'historical', 'parametric', "
            "or 'monte_carlo'"
        )

    return {
        "var": float(var),
        "max_var": float(max_var),
        "within_limit": bool(
            var <= max_var
        ),
        "method": method,
        "confidence": float(confidence),
    }


# ============================================================================
# 16. CVaR / EXPECTED SHORTFALL CONSTRAINT
# ============================================================================

def historical_cvar(
    returns,
    confidence=0.95,
):
    """
    Historical Expected Shortfall / CVaR.
    """
    if not (
        0 < confidence < 1
    ):
        raise ValueError(
            "confidence must be between 0 and 1"
        )

    r = _validate_returns(
        returns
    )

    if isinstance(r, pd.DataFrame):
        r = r.sum(axis=1)

    cutoff = np.quantile(
        r.values,
        1.0 - confidence,
    )

    tail = r[
        r <= cutoff
    ]

    return float(
        -tail.mean()
    )


def parametric_cvar(
    returns,
    confidence=0.95,
):
    """
    Normal-distribution Expected Shortfall.
    """
    if not (
        0 < confidence < 1
    ):
        raise ValueError(
            "confidence must be between 0 and 1"
        )

    r = _validate_returns(
        returns
    )

    if isinstance(r, pd.DataFrame):
        r = r.sum(axis=1)

    mu = float(
        r.mean()
    )

    sigma = float(
        r.std(ddof=1)
    )

    if sigma <= 0:
        return max(
            0.0,
            -mu,
        )

    from math import (
        exp,
        pi,
        sqrt,
    )

    from statistics import (
        NormalDist,
    )

    z = NormalDist().inv_cdf(
        confidence
    )

    pdf = (
        exp(-0.5 * z * z)
        / sqrt(2.0 * pi)
    )

    expected_shortfall_return = (
        mu
        - sigma
        * pdf
        / (1.0 - confidence)
    )

    return float(
        -expected_shortfall_return
    )


def monte_carlo_cvar(
    returns,
    confidence=0.95,
    simulations=10000,
    random_state=42,
):
    """
    Monte Carlo CVaR using empirical bootstrap resampling.
    """
    if not (
        0 < confidence < 1
    ):
        raise ValueError(
            "confidence must be between 0 and 1"
        )

    if simulations < 100:
        raise ValueError(
            "simulations must be >= 100"
        )

    r = _validate_returns(
        returns
    )

    if isinstance(r, pd.DataFrame):
        r = r.sum(axis=1)

    rng = np.random.default_rng(
        random_state
    )

    simulated = rng.choice(
        r.values,
        size=int(simulations),
        replace=True,
    )

    cutoff = np.quantile(
        simulated,
        1.0 - confidence,
    )

    tail = simulated[
        simulated <= cutoff
    ]

    return float(
        -tail.mean()
    )


def cvar_constraint(
    returns,
    max_cvar,
    method="historical",
    confidence=0.95,
    simulations=10000,
    random_state=42,
):
    """
    Check whether CVaR / Expected Shortfall is within the risk limit.
    """
    _validate_positive_parameter(
        max_cvar,
        "max_cvar",
    )

    method = method.lower()

    if method == "historical":
        cvar = historical_cvar(
            returns,
            confidence,
        )

    elif method == "parametric":
        cvar = parametric_cvar(
            returns,
            confidence,
        )

    elif method in {
        "monte_carlo",
        "monte-carlo",
    }:
        cvar = monte_carlo_cvar(
            returns,
            confidence,
            simulations,
            random_state,
        )

    else:
        raise ValueError(
            "method must be "
            "'historical', 'parametric', "
            "or 'monte_carlo'"
        )

    return {
        "cvar": float(cvar),
        "max_cvar": float(max_cvar),
        "within_limit": bool(
            cvar <= max_cvar
        ),
        "method": method,
        "confidence": float(confidence),
    }


# ============================================================================
# 17. BETA MANAGEMENT
# ============================================================================

def portfolio_beta(
    weights,
    asset_betas,
):
    """Calculate portfolio beta."""
    w = _validate_weights(
        weights
    )

    beta = pd.Series(
        asset_betas,
        dtype=float,
    ).reindex(w.index)

    if beta.isna().any():
        raise ValueError(
            "asset_betas must contain "
            "every portfolio asset"
        )

    if not np.all(
        np.isfinite(beta.values)
    ):
        raise ValueError(
            "asset_betas must contain "
            "only finite values"
        )

    return float(
        np.dot(
            w.values,
            beta.values,
        )
    )


def beta_management(
    weights,
    asset_betas,
    target_beta=None,
    min_beta=None,
    max_beta=None,
    cash_beta=0.0,
    cash_name="CASH",
):
    """
    Manage portfolio market beta through exposure scaling.

    Modes:
        target_beta -> scale toward target
        max_beta    -> reduce beta if above limit
        min_beta    -> increase exposure if below minimum
    """
    if target_beta is not None:
        if not np.isfinite(
            target_beta
        ):
            raise ValueError(
                "target_beta must be finite"
            )

    if min_beta is not None:
        if not np.isfinite(
            min_beta
        ):
            raise ValueError(
                "min_beta must be finite"
            )

    if max_beta is not None:
        if not np.isfinite(
            max_beta
        ):
            raise ValueError(
                "max_beta must be finite"
            )

    if (
        min_beta is not None
        and max_beta is not None
        and min_beta > max_beta
    ):
        raise ValueError(
            "min_beta cannot exceed max_beta"
        )

    w = _validate_weights(
        weights
    )

    beta = pd.Series(
        asset_betas,
        dtype=float,
    ).reindex(w.index)

    if beta.isna().any():
        raise ValueError(
            "asset_betas must contain every asset"
        )

    current_beta = portfolio_beta(
        w,
        beta,
    )

    scale = 1.0

    if target_beta is not None:
        risky_beta = (
            current_beta
            - cash_beta
        )

        if abs(risky_beta) <= 1e-16:
            raise ValueError(
                "cannot scale a zero-beta portfolio "
                "toward a different beta"
            )

        scale = (
            target_beta
            - cash_beta
        ) / risky_beta

        scale = max(
            0.0,
            float(scale),
        )

    else:
        if (
            max_beta is not None
            and current_beta > max_beta
        ):
            denominator = (
                current_beta
                - cash_beta
            )

            if abs(denominator) <= 1e-16:
                raise ValueError(
                    "cannot reduce beta through scaling"
                )

            scale = min(
                scale,
                (
                    max_beta
                    - cash_beta
                ) / denominator,
            )

        if (
            min_beta is not None
            and current_beta < min_beta
        ):
            denominator = (
                current_beta
                - cash_beta
            )

            if abs(denominator) <= 1e-16:
                raise ValueError(
                    "cannot increase beta through scaling"
                )

            required_scale = (
                min_beta
                - cash_beta
            ) / denominator

            scale = max(
                scale,
                required_scale,
            )

    scale = max(
        0.0,
        float(scale),
    )

    adjusted = _scale_exposure(
        w,
        scale,
        cash_name=cash_name,
    )

    risky_adjusted = adjusted.drop(
        labels=[cash_name],
        errors="ignore",
    )

    adjusted_beta = (
        portfolio_beta(
            risky_adjusted,
            beta,
        )
        + adjusted.get(
            cash_name,
            0.0,
        )
        * cash_beta
    )

    return {
        "weights": adjusted,
        "current_beta": float(current_beta),
        "target_beta": target_beta,
        "min_beta": min_beta,
        "max_beta": max_beta,
        "adjusted_beta": float(adjusted_beta),
        "scaling_factor": scale,
    }


# ============================================================================
# 18. CORRELATION / DIVERSIFICATION CONTROL
# ============================================================================

def average_pairwise_correlation(
    correlation,
):
    """Calculate average off-diagonal correlation."""
    corr = pd.DataFrame(
        correlation,
        dtype=float,
    )

    if (
        corr.shape[0]
        != corr.shape[1]
    ):
        raise ValueError(
            "correlation matrix must be square"
        )

    if corr.shape[0] < 2:
        return 0.0

    mask = ~np.eye(
        corr.shape[0],
        dtype=bool,
    )

    values = corr.values[
        mask
    ]

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "correlation must contain "
            "only finite values"
        )

    return float(
        values.mean()
    )


def diversification_ratio(
    weights,
    covariance,
):
    """
    Diversification ratio:

        weighted average asset volatility
        ---------------------------------
             portfolio volatility
    """
    w = _validate_weights(
        weights
    )

    cov = _validate_covariance(
        covariance,
        w.index,
    )

    asset_vol = np.sqrt(
        np.maximum(
            np.diag(cov.values),
            0.0,
        )
    )

    portfolio_vol = float(
        np.sqrt(
            max(
                w.values
                @ cov.values
                @ w.values,
                0.0,
            )
        )
    )

    if portfolio_vol <= 1e-16:
        raise ValueError(
            "portfolio volatility is effectively zero"
        )

    return float(
        np.dot(
            w.values,
            asset_vol,
        )
        / portfolio_vol
    )


def effective_number_of_assets(
    weights,
):
    """
    Effective number of assets:

        1 / sum(weights^2)
    """
    w = _normalize_long_only(
        weights
    )

    concentration = float(
        (w ** 2).sum()
    )

    return float(
        1.0 / concentration
    )


def correlation_diversification_control(
    weights,
    covariance,
    min_diversification_ratio=1.20,
    max_average_correlation=0.70,
):
    """
    Evaluate portfolio correlation and diversification risk.

    Returns diagnostics rather than forcing a reallocation.
    """
    _validate_positive_parameter(
        min_diversification_ratio,
        "min_diversification_ratio",
    )

    if not (
        -1
        <= max_average_correlation
        <= 1
    ):
        raise ValueError(
            "max_average_correlation must "
            "be between -1 and 1"
        )

    w = _validate_weights(
        weights
    )

    cov = _validate_covariance(
        covariance,
        w.index,
    )

    std = np.sqrt(
        np.maximum(
            np.diag(cov.values),
            0.0,
        )
    )

    denominator = np.outer(
        std,
        std,
    )

    correlation = np.divide(
        cov.values,
        denominator,
        out=np.zeros_like(
            cov.values
        ),
        where=denominator > 1e-16,
    )

    np.fill_diagonal(
        correlation,
        1.0,
    )

    correlation_df = pd.DataFrame(
        correlation,
        index=w.index,
        columns=w.index,
    )

    avg_corr = (
        average_pairwise_correlation(
            correlation_df
        )
    )

    div_ratio = (
        diversification_ratio(
            w,
            cov,
        )
    )

    effective_assets = (
        effective_number_of_assets(
            w
        )
    )

    diversification_ok = (
        div_ratio
        >= min_diversification_ratio
    )

    correlation_ok = (
        avg_corr
        <= max_average_correlation
    )

    return {
        "correlation": correlation_df,
        "average_pairwise_correlation": float(
            avg_corr
        ),
        "diversification_ratio": float(
            div_ratio
        ),
        "effective_number_of_assets": float(
            effective_assets
        ),
        "min_diversification_ratio": float(
            min_diversification_ratio
        ),
        "max_average_correlation": float(
            max_average_correlation
        ),
        "diversification_ok": bool(
            diversification_ok
        ),
        "correlation_ok": bool(
            correlation_ok
        ),
        "within_limits": bool(
            diversification_ok
            and correlation_ok
        ),
    }


# ============================================================================
# UNIFIED DASHBOARD INTERFACE
# ============================================================================

RISK_MANAGEMENT_ENGINES = {
    "None": None,

    "Maximum Position Weight":
        maximum_position_weight,

    "Maximum Industry Exposure":
        maximum_industry_exposure,

    "Maximum Sector Exposure":
        maximum_sector_exposure,

    "Gross / Net Exposure Control":
        control_gross_net_exposure,

    "Leverage Control":
        control_leverage,

    "Volatility Targeting":
        volatility_targeting,

    "Volatility Limit":
        volatility_limit,

    "Dynamic Volatility Scaling":
        dynamic_volatility_scaling,

    "EWMA Risk Scaling":
        ewma_risk_scaling,

    "Maximum Drawdown Control":
        maximum_drawdown_control,

    "High-Water-Mark De-Risking":
        high_water_mark_derisking,

    "Trend / Moving-Average De-Risking":
        trend_derisking,

    "Risk Contribution Constraint":
        risk_contribution_constraint,

    "Risk Budgeting":
        risk_budgeting,

    "Maximum Risk Concentration":
        risk_concentration,

    "VaR Constraint":
        var_constraint,

    "CVaR / Expected Shortfall Constraint":
        cvar_constraint,

    "Beta Management":
        beta_management,

    "Correlation / Diversification Control":
        correlation_diversification_control,
}


def list_risk_management_engines():
    """Return available risk-management methods."""
    return list(
        RISK_MANAGEMENT_ENGINES.keys()
    )


def get_risk_management_engine(
    name,
):
    """Return a risk-management function by dashboard name."""
    if name not in RISK_MANAGEMENT_ENGINES:
        raise ValueError(
            f"Unknown risk-management engine: {name}. "
            f"Available engines: "
            f"{list_risk_management_engines()}"
        )

    return RISK_MANAGEMENT_ENGINES[name]


__all__ = [
    # Exposure
    "maximum_position_weight",
    "maximum_group_exposure",
    "maximum_industry_exposure",
    "maximum_sector_exposure",
    "control_gross_net_exposure",
    "control_leverage",

    # Volatility
    "volatility_targeting",
    "volatility_limit",
    "dynamic_volatility_scaling",
    "ewma_volatility",
    "ewma_risk_scaling",

    # Drawdown
    "calculate_drawdown",
    "maximum_drawdown_control",
    "high_water_mark_derisking",
    "trend_derisking",

    # Risk budgeting
    "calculate_risk_contributions",
    "risk_contribution_constraint",
    "risk_budgeting",
    "risk_concentration",

    # Tail risk
    "historical_var",
    "parametric_var",
    "monte_carlo_var",
    "var_constraint",
    "historical_cvar",
    "parametric_cvar",
    "monte_carlo_cvar",
    "cvar_constraint",

    # Beta / diversification
    "portfolio_beta",
    "beta_management",
    "average_pairwise_correlation",
    "diversification_ratio",
    "effective_number_of_assets",
    "correlation_diversification_control",

    # Dashboard
    "RISK_MANAGEMENT_ENGINES",
    "list_risk_management_engines",
    "get_risk_management_engine",
]