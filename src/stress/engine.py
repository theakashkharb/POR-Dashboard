from __future__ import annotations

import pandas as pd

from src.stress.market import (
    market_stress_summary,
)

from src.stress.factor import (
    factor_stress_summary,
)

from src.stress.historical import (
    historical_stress_analysis,
)

from src.stress.liquidity import (
    liquidity_stress_summary,
)


# ============================================================
# ENGINE REGISTRY
# ============================================================

SUPPORTED_STRESS_ENGINES = [
    "market",
    "factor",
    "historical",
    "liquidity",
]


def list_stress_engines():
    """
    Return all supported stress analysis engines.
    """
    return SUPPORTED_STRESS_ENGINES.copy()


def validate_stress_engine(engine):
    """
    Validate a stress engine name.
    """
    if engine not in SUPPORTED_STRESS_ENGINES:
        raise ValueError(
            f"Unsupported stress engine: {engine}. "
            f"Supported engines: "
            f"{SUPPORTED_STRESS_ENGINES}"
        )

    return engine


# ============================================================
# INDIVIDUAL ENGINE BUILDERS
# ============================================================

def run_market_stress(
    returns,
    weights,
):
    """
    Run market shock stress analysis.
    """
    return market_stress_summary(
        returns,
        weights,
    )


def run_factor_stress(
    weights,
    asset_returns,
    factor_returns,
    factor_shocks,
):
    """
    Run factor stress analysis.
    """
    return factor_stress_summary(
        weights,
        asset_returns,
        factor_returns,
        factor_shocks,
    )


def run_historical_stress(
    returns,
    weights,
    scenarios,
):
    """
    Run historical stress analysis.
    """
    return historical_stress_analysis(
        returns,
        weights,
        scenarios,
    )


def run_liquidity_stress(
    weights,
    returns,
    liquidity,
    impact_coefficient=0.10,
):
    """
    Run liquidity stress analysis.
    """
    return liquidity_stress_summary(
        weights,
        returns,
        liquidity,
        impact_coefficient,
    )


# ============================================================
# COMPLETE STRESS ANALYSIS
# ============================================================

def run_all_stress_tests(
    returns,
    weights,
    factor_returns=None,
    factor_shocks=None,
    historical_scenarios=None,
    liquidity=None,
    impact_coefficient=0.10,
):
    """
    Run every available stress analysis.

    Optional inputs determine whether factor,
    historical, and liquidity stress tests are executed.

    Market stress always runs.
    """
    results = {}

    # --------------------------------------------------------
    # MARKET
    # --------------------------------------------------------

    results["market"] = run_market_stress(
        returns,
        weights,
    )

    # --------------------------------------------------------
    # FACTOR
    # --------------------------------------------------------

    if (
        factor_returns is not None
        and factor_shocks is not None
    ):
        results["factor"] = run_factor_stress(
            weights,
            returns,
            factor_returns,
            factor_shocks,
        )

    # --------------------------------------------------------
    # HISTORICAL
    # --------------------------------------------------------

    if historical_scenarios is not None:
        results["historical"] = (
            run_historical_stress(
                returns,
                weights,
                historical_scenarios,
            )
        )

    # --------------------------------------------------------
    # LIQUIDITY
    # --------------------------------------------------------

    if liquidity is not None:
        results["liquidity"] = (
            run_liquidity_stress(
                weights,
                returns,
                liquidity,
                impact_coefficient,
            )
        )

    return results


# ============================================================
# STRESS SUMMARY
# ============================================================

def stress_engine_summary(
    returns,
    weights,
    factor_returns=None,
    factor_shocks=None,
    historical_scenarios=None,
    liquidity=None,
    impact_coefficient=0.10,
):
    """
    Run complete stress analysis and return a
    compact summary of available stress engines.
    """
    results = run_all_stress_tests(
        returns=returns,
        weights=weights,
        factor_returns=factor_returns,
        factor_shocks=factor_shocks,
        historical_scenarios=historical_scenarios,
        liquidity=liquidity,
        impact_coefficient=impact_coefficient,
    )

    summary = []

    if "market" in results:
        market_worst = (
            results["market"]["worst_case"]
        )

        summary.append(
            {
                "engine": "market",
                "worst_case_return":
                    float(
                        market_worst[
                            "portfolio_return"
                        ]
                    ),
                "worst_case_loss":
                    float(
                        market_worst[
                            "portfolio_loss"
                        ]
                    ),
            }
        )

    if "factor" in results:
        summary.append(
            {
                "engine": "factor",
                "worst_case_return":
                    float(
                        results["factor"][
                            "portfolio_return"
                        ]
                    ),
                "worst_case_loss":
                    float(
                        results["factor"][
                            "portfolio_loss"
                        ]
                    ),
            }
        )

    if "historical" in results:
        historical_worst = (
            results["historical"][
                "worst_case"
            ]
        )

        summary.append(
            {
                "engine": "historical",
                "worst_case_return":
                    float(
                        historical_worst[
                            "cumulative_return"
                        ]
                    ),
                "worst_case_loss":
                    float(
                        -historical_worst[
                            "cumulative_return"
                        ]
                    ),
            }
        )

    if "liquidity" in results:
        liquidity_worst = (
            results["liquidity"][
                "worst_case"
            ]
        )

        summary.append(
            {
                "engine": "liquidity",
                "worst_case_return":
                    float(
                        liquidity_worst[
                            "portfolio_return"
                        ]
                    ),
                "worst_case_loss":
                    float(
                        liquidity_worst[
                            "portfolio_loss"
                        ]
                    ),
            }
        )

    return {
        "results": results,
        "summary": pd.DataFrame(
            summary
        ),
        "engines_run": list(
            results.keys()
        ),
    }