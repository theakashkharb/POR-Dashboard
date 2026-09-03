"""
POR-Dashboard
Pipeline Integration Layer
==========================

Connects:

Data
  ↓
Returns
  ↓
Portfolio Construction
  ↓
Risk Management
  ↓
Stress Testing
  ↓
Backtest
  ↓
Performance + Risk Analytics

The dashboard should interact with this module instead of directly
calling individual engines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.returns.returns import (
    create_return_matrix,
    calculate_historical_expected_return,
    calculate_geometric_expected_return,
)

from src.risk.risk import (
    calculate_historical_volatility,
    calculate_ewma_volatility,
    calculate_covariance_matrix,
    calculate_correlation_matrix,
)

from src.optimization.optimization import (
    list_optimization_methods,
    get_optimizer,
)

from src.backtest.backtest import (
    run_backtest,
)

from src.performance.performance import (
    calculate_total_return,
    calculate_cagr,
    calculate_annualized_volatility,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
)

from src.risk.portfolio_risk import (
    portfolio_volatility,
    portfolio_beta,
    risk_contribution_table,
    weight_concentration,
    risk_concentration,
    diversification_ratio,
)

from src.risk.risk_management import (
    list_risk_management_engines,
    get_risk_management_engine,
)

from src.stress.stress_testing import (
    list_stress_test_scenarios,
    run_stress_test,
)


# ============================================================
# VALIDATION
# ============================================================

def _validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Validate price data."""

    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")

    if prices.empty:
        raise ValueError("prices cannot be empty")

    prices = prices.apply(pd.to_numeric, errors="coerce")

    if prices.isna().any().any():
        raise ValueError("prices contain NaN or non-numeric values")

    if not np.isfinite(prices.to_numpy()).all():
        raise ValueError("prices contain infinite values")

    if (prices <= 0).any().any():
        raise ValueError("prices must be positive")

    return prices.astype(float)


def _validate_weights(weights: pd.Series | dict) -> pd.Series:
    """Validate portfolio weights."""

    if isinstance(weights, dict):
        weights = pd.Series(weights, dtype=float)

    if not isinstance(weights, pd.Series):
        raise TypeError("weights must be a pandas Series or dict")

    weights = pd.to_numeric(weights, errors="coerce")

    if weights.empty:
        raise ValueError("weights cannot be empty")

    if weights.isna().any():
        raise ValueError("weights contain NaN values")

    if not np.isfinite(weights.to_numpy()).all():
        raise ValueError("weights contain infinite values")

    if (weights < 0).any():
        raise ValueError("negative weights are not supported")

    if weights.sum() <= 0:
        raise ValueError("weights must have a positive sum")

    return weights.astype(float)


def _normalize_weights(weights: pd.Series) -> pd.Series:
    """Normalize weights so that they sum to one."""

    total = float(weights.sum())

    if total <= 0:
        raise ValueError("cannot normalize zero-sum weights")

    return weights / total


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_price_data(
    prices: pd.DataFrame,
    selected_assets: list[str] | None = None,
) -> pd.DataFrame:
    """
    Prepare price data for the pipeline.

    Parameters
    ----------
    prices : DataFrame
        Wide-format price data.

    selected_assets : list[str], optional
        Assets selected by the user.

    Returns
    -------
    DataFrame
    """

    prices = _validate_prices(prices)

    if selected_assets is not None:

        missing = [
            asset
            for asset in selected_assets
            if asset not in prices.columns
        ]

        if missing:
            raise ValueError(
                f"Selected assets not found in prices: {missing}"
            )

        prices = prices.loc[:, selected_assets]

    if prices.shape[1] == 0:
        raise ValueError("No assets available")

    prices = prices.sort_index()

    return prices


def prepare_returns(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create daily simple returns from prices.
    """

    prices = _validate_prices(prices)

    returns = prices.pct_change().dropna()

    if returns.empty:
        raise ValueError(
            "Not enough price observations to calculate returns"
        )

    return returns


# ============================================================
# PORTFOLIO CONSTRUCTION
# ============================================================

def construct_portfolio(
    prices: pd.DataFrame,
    method: str,
    *,
    industry_data: pd.Series | dict | None = None,
    current_weights: pd.Series | dict | None = None,
    max_turnover: float | None = None,
    expected_return_method: str = "historical",
    optimizer_kwargs: dict | None = None,
) -> dict:
    """
    Construct a portfolio using the selected optimization method.

    Returns a structured result for the dashboard.
    """

    prices = _validate_prices(prices)

    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    returns = prepare_returns(prices)

    # --------------------------------------------------------
    # Expected return
    # --------------------------------------------------------

    if expected_return_method == "historical":

        expected_returns = (
            calculate_historical_expected_return(
                returns
            )
        )

    elif expected_return_method == "geometric":

        expected_returns = (
            calculate_geometric_expected_return(
                returns
            )
        )

    else:
        raise ValueError(
            "expected_return_method must be "
            "'historical' or 'geometric'"
        )

    # --------------------------------------------------------
    # Covariance
    # --------------------------------------------------------

    covariance = calculate_covariance_matrix(
        returns
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = get_optimizer(method)

    weights = optimizer(
        expected_returns=expected_returns,
        covariance=covariance,
        industry_data=industry_data,
        current_weights=current_weights,
        max_turnover=max_turnover,
        **optimizer_kwargs,
    )

    weights = _validate_weights(weights)
    weights = weights.reindex(prices.columns, fill_value=0.0)

    weights = _normalize_weights(weights)

    return {
        "method": method,
        "weights": weights,
        "expected_returns": expected_returns,
        "covariance": covariance,
        "returns": returns,
    }


# ============================================================
# RISK ANALYSIS
# ============================================================

def calculate_portfolio_risk(
    returns: pd.DataFrame,
    weights: pd.Series | dict,
    covariance: pd.DataFrame | None = None,
    benchmark_returns: pd.Series | None = None,
    asset_betas: pd.Series | None = None,
) -> dict:
    """
    Calculate portfolio-level risk metrics.
    """

    weights = _validate_weights(weights)

    returns = returns.copy()

    common_assets = weights.index.intersection(
        returns.columns
    )

    if len(common_assets) == 0:
        raise ValueError(
            "No common assets between weights and returns"
        )

    returns = returns.loc[:, common_assets]
    weights = weights.loc[common_assets]

    weights = _normalize_weights(weights)

    if covariance is None:
        covariance = calculate_covariance_matrix(
            returns
        )

    covariance = covariance.loc[
        common_assets,
        common_assets,
    ]

    # --------------------------------------------------------
    # Portfolio volatility
    # --------------------------------------------------------

    volatility = portfolio_volatility(
        weights,
        covariance,
    )

    # --------------------------------------------------------
    # Risk contributions
    # --------------------------------------------------------

    risk_table = risk_contribution_table(
        weights,
        covariance,
    )

    # --------------------------------------------------------
    # Concentration
    # --------------------------------------------------------

    concentration = weight_concentration(
        weights
    )

    risk_concentration_value = risk_concentration(
        weights,
        covariance,
    )

    # Some versions of the engine return a scalar while others
    # return a structured result.
    if isinstance(risk_concentration_value, dict):

        risk_concentration_result = (
            risk_concentration_value
        )

    else:

        risk_concentration_result = (
            float(risk_concentration_value)
        )

    # --------------------------------------------------------
    # Diversification
    # --------------------------------------------------------

    diversification = diversification_ratio(
        weights,
        covariance,
    )

    # --------------------------------------------------------
    # Beta
    # --------------------------------------------------------

    beta = None

    if asset_betas is not None:

        asset_betas = pd.Series(
            asset_betas,
            dtype=float,
        )

        common_beta_assets = weights.index.intersection(
            asset_betas.index
        )

        if len(common_beta_assets) > 0:

            beta = portfolio_beta(
                weights.loc[common_beta_assets],
                asset_betas.loc[common_beta_assets],
            )

    elif benchmark_returns is not None:

        benchmark_returns = pd.Series(
            benchmark_returns,
            dtype=float,
        )

        common_dates = returns.index.intersection(
            benchmark_returns.index
        )

        if len(common_dates) > 1:

            portfolio_returns = (
                returns.loc[common_dates]
                .mul(weights, axis=1)
                .sum(axis=1)
            )

            benchmark = benchmark_returns.loc[
                common_dates
            ]

            benchmark_variance = benchmark.var()

            if benchmark_variance > 0:

                beta = float(
                    portfolio_returns.cov(
                        benchmark
                    )
                    / benchmark_variance
                )

    return {
        "portfolio_volatility": float(volatility),
        "risk_contributions": risk_table,
        "weight_concentration": float(concentration),
        "risk_concentration": risk_concentration_result,
        "diversification_ratio": float(
            diversification
        ),
        "portfolio_beta": beta,
    }


# ============================================================
# RISK MANAGEMENT
# ============================================================

def apply_risk_management(
    weights: pd.Series | dict,
    method: str,
    *,
    returns: pd.DataFrame | None = None,
    covariance: pd.DataFrame | None = None,
    industry_data: pd.Series | dict | None = None,
    sector_data: pd.Series | dict | None = None,
    asset_betas: pd.Series | None = None,
    nav: pd.Series | None = None,
    price_series: pd.DataFrame | None = None,
    risk_management_kwargs: dict | None = None,
) -> dict:
    """
    Apply the selected risk-management engine.

    The function keeps risk-management logic separate from
    portfolio construction.
    """

    weights = _validate_weights(weights)

    if risk_management_kwargs is None:
        risk_management_kwargs = {}

    # --------------------------------------------------------
    # NONE
    # --------------------------------------------------------

    if method == "None":

        return {
            "method": "None",
            "original_weights": weights.copy(),
            "adjusted_weights": weights.copy(),
            "result": weights.copy(),
        }

    # --------------------------------------------------------
    # Get selected engine
    # --------------------------------------------------------

    engine = get_risk_management_engine(
        method
    )

    if engine is None:

        return {
            "method": method,
            "original_weights": weights.copy(),
            "adjusted_weights": weights.copy(),
            "result": weights.copy(),
        }

    # --------------------------------------------------------
    # Build arguments according to engine
    # --------------------------------------------------------

    kwargs = dict(
        risk_management_kwargs
    )

    # Exposure controls
    if method == "Maximum Position Weight":

        result = engine(
            weights,
            **kwargs,
        )

    elif method == "Maximum Industry Exposure":

        if industry_data is None:
            raise ValueError(
                "industry_data is required"
            )

        result = engine(
            weights,
            industry_data,
            **kwargs,
        )

    elif method == "Maximum Sector Exposure":

        if sector_data is None:
            raise ValueError(
                "sector_data is required"
            )

        result = engine(
            weights,
            sector_data,
            **kwargs,
        )

    elif method == "Gross / Net Exposure Control":

        result = engine(
            weights,
            **kwargs,
        )

    elif method == "Leverage Control":

        result = engine(
            weights,
            **kwargs,
        )

    # --------------------------------------------------------
    # Volatility controls
    # --------------------------------------------------------

    elif method in (
        "Volatility Targeting",
        "Volatility Limit",
    ):

        if covariance is None:
            raise ValueError(
                "covariance is required"
            )

        result = engine(
            weights,
            covariance,
            **kwargs,
        )

    elif method == "Dynamic Volatility Scaling":

        if returns is None:
            raise ValueError(
                "returns are required"
            )

        volatility = (
            returns.std()
            * np.sqrt(
                kwargs.pop(
                    "annualization",
                    252,
                )
            )
        )

        result = engine(
            weights,
            volatility,
            **kwargs,
        )

    elif method == "EWMA Risk Scaling":

        if returns is None:
            raise ValueError(
                "returns are required"
            )

        result = engine(
            weights,
            returns,
            **kwargs,
        )

    # --------------------------------------------------------
    # Drawdown controls
    # --------------------------------------------------------

    elif method == "Maximum Drawdown Control":

        if nav is None:
            raise ValueError(
                "nav is required"
            )

        result = engine(
            weights,
            nav,
            **kwargs,
        )

    elif method == "High-Water-Mark De-Risking":

        if nav is None:
            raise ValueError(
                "nav is required"
            )

        result = engine(
            weights,
            nav,
            **kwargs,
        )

    elif method == "Trend / Moving-Average De-Risking":

        if price_series is None:
            raise ValueError(
                "price_series is required"
            )

        result = engine(
            weights,
            price_series,
            **kwargs,
        )

    # --------------------------------------------------------
    # Risk contribution
    # --------------------------------------------------------

    elif method == "Risk Contribution Constraint":

        if covariance is None:
            raise ValueError(
                "covariance is required"
            )

        result = engine(
            weights,
            covariance,
            **kwargs,
        )

    elif method == "Risk Budgeting":

        if covariance is None:
            raise ValueError(
                "covariance is required"
            )

        risk_budgets = kwargs.pop(
            "risk_budgets",
            np.ones(len(weights))
            / len(weights),
        )

        result = engine(
            covariance,
            risk_budgets,
            **kwargs,
        )

    elif method == "Maximum Risk Concentration":

        if covariance is None:
            raise ValueError(
                "covariance is required"
            )

        result = engine(
            weights,
            covariance,
            **kwargs,
        )

    # --------------------------------------------------------
    # Tail risk
    # --------------------------------------------------------

    elif method == "VaR Constraint":

        if returns is None:
            raise ValueError(
                "returns are required"
            )

        portfolio_returns = (
            returns
            .loc[:, weights.index]
            .mul(weights, axis=1)
            .sum(axis=1)
        )

        result = engine(
            portfolio_returns,
            **kwargs,
        )

    elif method == "CVaR / Expected Shortfall Constraint":

        if returns is None:
            raise ValueError(
                "returns are required"
            )

        portfolio_returns = (
            returns
            .loc[:, weights.index]
            .mul(weights, axis=1)
            .sum(axis=1)
        )

        result = engine(
            portfolio_returns,
            **kwargs,
        )

    # --------------------------------------------------------
    # Beta
    # --------------------------------------------------------

    elif method == "Beta Management":

        if asset_betas is None:
            raise ValueError(
                "asset_betas are required"
            )

        result = engine(
            weights,
            asset_betas,
            **kwargs,
        )

    # --------------------------------------------------------
    # Diversification
    # --------------------------------------------------------

    elif method == "Correlation / Diversification Control":

        if covariance is None:
            raise ValueError(
                "covariance is required"
            )

        result = engine(
            weights,
            covariance,
            **kwargs,
        )

    else:

        raise ValueError(
            f"Risk-management integration not implemented "
            f"for method: {method}"
        )

    # --------------------------------------------------------
    # Extract adjusted weights
    # --------------------------------------------------------

    if isinstance(result, pd.Series):

        adjusted_weights = result.copy()

    elif isinstance(result, dict):

        if "weights" in result:

            adjusted_weights = pd.Series(
                result["weights"],
                dtype=float,
            )

        elif "adjusted_weights" in result:

            adjusted_weights = pd.Series(
                result["adjusted_weights"],
                dtype=float,
            )

        else:

            # Some engines are analytical rather than
            # weight-transforming engines.
            adjusted_weights = weights.copy()

    else:

        adjusted_weights = weights.copy()

    adjusted_weights = adjusted_weights.reindex(
        weights.index,
        fill_value=0.0,
    )

    # Normalize only when the engine returned a
    # portfolio-weight vector.
    if np.isfinite(adjusted_weights.to_numpy()).all():
        if adjusted_weights.sum() > 0:
            adjusted_weights = (
                adjusted_weights
                / adjusted_weights.sum()
            )

    return {
        "method": method,
        "original_weights": weights.copy(),
        "adjusted_weights": adjusted_weights,
        "result": result,
    }


# ============================================================
# STRESS TESTING
# ============================================================

def apply_stress_test(
    weights: pd.Series | dict,
    scenario: str,
    *,
    returns: pd.DataFrame | None = None,
    covariance: pd.DataFrame | None = None,
    sector_data: pd.Series | dict | None = None,
    scenario_parameters: dict | None = None,
    initial_nav: float = 1.0,
) -> dict:
    """
    Run the selected stress scenario.
    """

    weights = _validate_weights(weights)

    return run_stress_test(
        weights=weights,
        scenario=scenario,
        returns=returns,
        covariance=covariance,
        sector_data=sector_data,
        scenario_parameters=scenario_parameters,
        initial_nav=initial_nav,
    )


# ============================================================
# PERFORMANCE ANALYTICS
# ============================================================

def calculate_performance_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
) -> dict:
    """
    Calculate dashboard performance metrics.
    """

    portfolio_returns = pd.Series(
        portfolio_returns,
        dtype=float,
    )

    if portfolio_returns.empty:
        raise ValueError(
            "portfolio_returns cannot be empty"
        )

    metrics = {
        "total_return": float(
            calculate_total_return(
                portfolio_returns
            )
        ),
        "cagr": float(
            calculate_cagr(
                portfolio_returns
            )
        ),
        "annualized_volatility": float(
            calculate_annualized_volatility(
                portfolio_returns
            )
        ),
        "sharpe_ratio": float(
            calculate_sharpe_ratio(
                portfolio_returns
            )
        ),
        "maximum_drawdown": float(
            calculate_max_drawdown(
                portfolio_returns
            )
        ),
    }

    if benchmark_returns is not None:

        benchmark_returns = pd.Series(
            benchmark_returns,
            dtype=float,
        )

        common = portfolio_returns.index.intersection(
            benchmark_returns.index
        )

        if len(common) > 0:

            benchmark_total_return = (
                calculate_total_return(
                    benchmark_returns.loc[common]
                )
            )

            metrics["benchmark_return"] = float(
                benchmark_total_return
            )

            metrics["excess_return"] = float(
                metrics["total_return"]
                - benchmark_total_return
            )

    return metrics


# ============================================================
# NAV
# ============================================================

def calculate_portfolio_nav(
    portfolio_returns: pd.Series,
    initial_capital: float = 1.0,
) -> pd.Series:
    """
    Calculate portfolio NAV.
    """

    portfolio_returns = pd.Series(
        portfolio_returns,
        dtype=float,
    )

    if portfolio_returns.empty:
        raise ValueError(
            "portfolio_returns cannot be empty"
        )

    if initial_capital <= 0:
        raise ValueError(
            "initial_capital must be positive"
        )

    if (portfolio_returns <= -1.0).any():
        raise ValueError(
            "returns cannot be <= -100%"
        )

    return (
        initial_capital
        * (1.0 + portfolio_returns).cumprod()
    )


# ============================================================
# COMPLETE ANALYSIS
# ============================================================

def run_analysis(
    prices: pd.DataFrame,
    selected_assets: list[str],
    portfolio_method: str,
    risk_method: str,
    stress_scenario: str,
    *,
    benchmark_prices: pd.Series | pd.DataFrame | None = None,
    industry_data: pd.Series | dict | None = None,
    sector_data: pd.Series | dict | None = None,
    asset_betas: pd.Series | None = None,
    optimizer_kwargs: dict | None = None,
    risk_management_kwargs: dict | None = None,
    stress_parameters: dict | None = None,
    train_window: int = 252,
    rebalance_frequency: str = "M",
    max_turnover: float | None = 0.25,
    initial_capital: float = 1.0,
) -> dict:
    """
    MAIN POR-DASHBOARD ANALYSIS FUNCTION.

    User workflow:

        Select stocks
            ↓
        Portfolio construction
            ↓
        Risk management
            ↓
        Stress test
            ↓
        Backtest
            ↓
        Performance
            ↓
        Risk analytics

    Returns
    -------
    dict
        Complete dashboard-ready result.
    """

    # ========================================================
    # 1. PREPARE DATA
    # ========================================================

    prices = prepare_price_data(
        prices,
        selected_assets,
    )

    returns = prepare_returns(prices)

    # ========================================================
    # 2. PORTFOLIO CONSTRUCTION
    # ========================================================

    construction = construct_portfolio(
        prices,
        portfolio_method,
        industry_data=industry_data,
        max_turnover=max_turnover,
        optimizer_kwargs=optimizer_kwargs,
    )

    original_weights = construction["weights"]

    covariance = construction["covariance"]

    # ========================================================
    # 3. RISK MANAGEMENT
    # ========================================================

    risk_management = apply_risk_management(
        original_weights,
        risk_method,
        returns=returns,
        covariance=covariance,
        industry_data=industry_data,
        sector_data=sector_data,
        asset_betas=asset_betas,
        price_series=prices,
        risk_management_kwargs=risk_management_kwargs,
    )

    final_weights = risk_management[
        "adjusted_weights"
    ]

    # ========================================================
    # 4. PORTFOLIO RETURNS
    # ========================================================

    portfolio_returns = (
        returns
        .loc[:, final_weights.index]
        .mul(final_weights, axis=1)
        .sum(axis=1)
    )

    portfolio_nav = calculate_portfolio_nav(
        portfolio_returns,
        initial_capital,
    )

    # ========================================================
    # 5. BENCHMARK
    # ========================================================

    benchmark_returns = None

    if benchmark_prices is not None:

        if isinstance(
            benchmark_prices,
            pd.DataFrame,
        ):

            if benchmark_prices.shape[1] != 1:
                raise ValueError(
                    "benchmark_prices DataFrame "
                    "must contain exactly one column"
                )

            benchmark_prices = (
                benchmark_prices.iloc[:, 0]
            )

        benchmark_prices = pd.Series(
            benchmark_prices,
            dtype=float,
        )

        benchmark_returns = (
            benchmark_prices
            .pct_change()
            .dropna()
        )

    # ========================================================
    # 6. PERFORMANCE
    # ========================================================

    performance = calculate_performance_metrics(
        portfolio_returns,
        benchmark_returns,
    )

    # ========================================================
    # 7. PORTFOLIO RISK
    # ========================================================

    portfolio_risk = calculate_portfolio_risk(
        returns,
        final_weights,
        covariance=covariance,
        benchmark_returns=benchmark_returns,
        asset_betas=asset_betas,
    )

    # ========================================================
    # 8. STRESS TEST
    # ========================================================

    stress = apply_stress_test(
        final_weights,
        stress_scenario,
        returns=returns,
        covariance=covariance,
        sector_data=sector_data,
        scenario_parameters=stress_parameters,
        initial_nav=initial_capital,
    )

    # ========================================================
    # 9. BACKTEST
    # ========================================================

    backtest = run_backtest(
        prices=prices,
        selected_assets=list(prices.columns),
        optimizer=get_optimizer(
            portfolio_method
        ),
        benchmark_prices=benchmark_prices,
        train_window=train_window,
        rebalance_frequency=rebalance_frequency,
        max_turnover=max_turnover,
        initial_capital=initial_capital,
        industry_data=industry_data,
        **(
            optimizer_kwargs
            if optimizer_kwargs is not None
            else {}
        ),
    )

    # ========================================================
    # 10. FINAL RESULT
    # ========================================================

    return {
        # Data
        "prices": prices,
        "returns": returns,

        # Portfolio construction
        "portfolio_method": portfolio_method,
        "original_weights": original_weights,

        # Risk management
        "risk_method": risk_method,
        "risk_management": risk_management,
        "final_weights": final_weights,

        # Portfolio returns
        "portfolio_returns": portfolio_returns,
        "portfolio_nav": portfolio_nav,

        # Risk
        "portfolio_risk": portfolio_risk,

        # Stress
        "stress_scenario": stress_scenario,
        "stress_test": stress,

        # Performance
        "performance": performance,

        # Backtest
        "backtest": backtest,
    }


# ============================================================
# DASHBOARD OPTIONS
# ============================================================

def get_dashboard_options() -> dict:
    """
    Return all selectable dashboard options.
    """

    return {
        "portfolio_methods":
            list_optimization_methods(),

        "risk_management_methods":
            list_risk_management_engines(),

        "stress_test_scenarios":
            list_stress_test_scenarios(),
    }