"""
POR-Dashboard
Portfolio Optimization Engine

Dependencies:
    numpy
    pandas

Optimization Methods:
    1. Equal Weight
    2. Minimum Variance
    3. Maximum Sharpe
    4. Risk Parity
    5. Hierarchical Risk Parity (HRP)
    6. Black-Litterman

Portfolio Constraints:
    - Long only
    - Fully invested
    - Maximum industry exposure
    - Maximum turnover
"""

import numpy as np
import pandas as pd


# ============================================================
# VALIDATION
# ============================================================

def _validate_inputs(expected_returns, covariance):
    """
    Validate expected returns and covariance matrix.
    """

    if not isinstance(expected_returns, pd.Series):
        expected_returns = pd.Series(expected_returns)

    if not isinstance(covariance, pd.DataFrame):
        covariance = pd.DataFrame(
            covariance,
            index=expected_returns.index,
            columns=expected_returns.index
        )

    assets = expected_returns.index

    covariance = covariance.loc[assets, assets]

    if len(assets) == 0:
        raise ValueError("No assets supplied.")

    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError(
            "Covariance matrix must be square."
        )

    if not np.allclose(
        covariance.values,
        covariance.values.T,
        atol=1e-10
    ):
        raise ValueError(
            "Covariance matrix must be symmetric."
        )

    if expected_returns.isna().any():
        raise ValueError(
            "Expected returns contain NaN values."
        )

    if covariance.isna().any().any():
        raise ValueError(
            "Covariance matrix contains NaN values."
        )

    return (
        expected_returns.astype(float),
        covariance.astype(float)
    )


def _validate_weights(weights, assets):
    """
    Validate portfolio weights.

    Conditions:
        - No NaN
        - No infinite values
        - No negative weights
        - Weight sum = 1
    """

    weights = pd.Series(
        weights,
        index=assets,
        dtype=float
    )

    if not np.all(
        np.isfinite(weights.values)
    ):
        raise ValueError(
            "Weights contain NaN or infinite values."
        )

    if np.any(
        weights.values < -1e-10
    ):
        raise ValueError(
            "Negative weights are not allowed."
        )

    total = weights.sum()

    if total <= 0:
        raise ValueError(
            "Weight sum must be positive."
        )

    weights = weights / total

    return weights


# ============================================================
# BASIC PORTFOLIO METRICS
# ============================================================

def portfolio_return(
    weights,
    expected_returns
):
    """
    Expected portfolio return.

    Rp = w'μ
    """

    weights = np.asarray(
        weights,
        dtype=float
    )

    expected_returns = np.asarray(
        expected_returns,
        dtype=float
    )

    return float(
        weights @ expected_returns
    )


def portfolio_variance(
    weights,
    covariance
):
    """
    Portfolio variance.

    Var(Rp) = w'Σw
    """

    weights = np.asarray(
        weights,
        dtype=float
    )

    covariance = np.asarray(
        covariance,
        dtype=float
    )

    return float(
        weights @ covariance @ weights
    )


def portfolio_volatility(
    weights,
    covariance
):
    """
    Portfolio volatility.
    """

    variance = portfolio_variance(
        weights,
        covariance
    )

    return float(
        np.sqrt(
            max(variance, 0.0)
        )
    )


def portfolio_sharpe(
    weights,
    expected_returns,
    covariance,
    risk_free_rate=0.0
):
    """
    Portfolio Sharpe ratio.
    """

    ret = portfolio_return(
        weights,
        expected_returns
    )

    volatility = portfolio_volatility(
        weights,
        covariance
    )

    if volatility <= 1e-12:
        return 0.0

    return float(
        (ret - risk_free_rate)
        / volatility
    )


# ============================================================
# WEIGHT NORMALIZATION
# ============================================================

def _normalize_weights(weights):
    """
    Convert weights to long-only fully invested weights.

    Conditions:
        weights >= 0
        sum(weights) = 1
    """

    weights = np.asarray(
        weights,
        dtype=float
    )

    weights = np.maximum(
        weights,
        0.0
    )

    total = weights.sum()

    if total <= 0:
        weights = np.ones(
            len(weights)
        )

        total = weights.sum()

    weights = (
        weights / total
    )

    return weights


# ============================================================
# INDUSTRY CONSTRAINT
# ============================================================

def check_industry_exposure(
    weights,
    industry_data
):
    """
    Calculate portfolio exposure by industry.

    Returns
    -------
    pd.Series
        Industry portfolio weights.
    """

    weights = pd.Series(
        weights,
        dtype=float
    )

    industry_data = pd.Series(
        industry_data
    ).loc[
        weights.index
    ]

    exposure = (
        weights
        .groupby(industry_data)
        .sum()
    )

    return exposure


def _apply_industry_constraint(
    weights,
    industry_data,
    max_industry_weight=0.35
):
    """
    Attempt to enforce maximum industry exposure.

    Excess industry weight is redistributed
    to industries below the limit.
    """

    weights = pd.Series(
        weights,
        dtype=float
    ).copy()

    industry_data = pd.Series(
        industry_data
    ).loc[
        weights.index
    ]

    if max_industry_weight <= 0:
        raise ValueError(
            "Maximum industry weight must be positive."
        )

    if max_industry_weight >= 1:
        return weights / weights.sum()

    for _ in range(1000):

        exposure = (
            weights
            .groupby(industry_data)
            .sum()
        )

        violating = (
            exposure
            > max_industry_weight + 1e-12
        )

        if not violating.any():
            break

        excess_total = 0.0

        for industry in exposure[
            violating
        ].index:

            excess = (
                exposure[industry]
                - max_industry_weight
            )

            mask = (
                industry_data
                == industry
            )

            industry_weight = (
                weights[mask].sum()
            )

            if industry_weight > 0:

                reduction = (
                    weights[mask]
                    / industry_weight
                )

                weights.loc[mask] -= (
                    reduction * excess
                )

                excess_total += excess

        weights = weights.clip(
            lower=0.0
        )

        if excess_total <= 1e-12:
            break

        exposure = (
            weights
            .groupby(industry_data)
            .sum()
        )

        available = (
            exposure
            < max_industry_weight - 1e-12
        )

        if not available.any():
            raise ValueError(
                "Industry constraint is infeasible."
            )

        available_industries = (
            exposure[available]
        )

        remaining_capacity = (
            max_industry_weight
            - available_industries
        )

        capacity_sum = (
            remaining_capacity.sum()
        )

        if capacity_sum <= 0:
            raise ValueError(
                "Industry constraint is infeasible."
            )

        for industry in (
            available_industries.index
        ):

            capacity_share = (
                remaining_capacity[industry]
                / capacity_sum
            )

            mask = (
                industry_data
                == industry
            )

            industry_weight = (
                weights[mask].sum()
            )

            if industry_weight > 0:

                weights.loc[mask] += (
                    excess_total
                    * capacity_share
                    * weights[mask]
                    / industry_weight
                )

            else:

                indices = (
                    weights.index[mask]
                )

                if len(indices) > 0:

                    weights.loc[indices] += (
                        excess_total
                        * capacity_share
                        / len(indices)
                    )

        weights = (
            weights
            / weights.sum()
        )

    return weights


# ============================================================
# TURNOVER
# ============================================================

def calculate_turnover(
    current_weights,
    new_weights
):
    """
    Calculate portfolio turnover.

    Turnover =
        sum(abs(new_weights - current_weights))
    """

    current_weights = pd.Series(
        current_weights,
        dtype=float
    )

    new_weights = pd.Series(
        new_weights,
        index=current_weights.index,
        dtype=float
    )

    return float(
        np.abs(
            new_weights.values
            - current_weights.values
        ).sum()
    )


def _apply_turnover_constraint(
    new_weights,
    current_weights,
    max_turnover
):
    """
    Move the new portfolio toward the current
    portfolio if turnover exceeds the limit.
    """

    if current_weights is None:
        return new_weights

    if max_turnover is None:
        return new_weights

    current_weights = pd.Series(
        current_weights,
        index=new_weights.index,
        dtype=float
    )

    turnover = calculate_turnover(
        current_weights,
        new_weights
    )

    if turnover <= max_turnover:
        return new_weights

    if turnover <= 1e-12:
        return new_weights

    scale = (
        max_turnover
        / turnover
    )

    adjusted = (
        current_weights
        + scale
        * (
            new_weights
            - current_weights
        )
    )

    adjusted = adjusted.clip(
        lower=0.0
    )

    adjusted = (
        adjusted
        / adjusted.sum()
    )

    return adjusted


# ============================================================
# FINAL CONSTRAINT PROCESSOR
# ============================================================

def _apply_constraints(
    weights,
    assets,
    industry_data=None,
    max_industry_weight=0.35,
    current_weights=None,
    max_turnover=None
):
    """
    Apply portfolio-level constraints.

    Constraints:
        1. Long only
        2. Fully invested
        3. Industry limit
        4. Turnover limit
    """

    weights = pd.Series(
        weights,
        index=assets,
        dtype=float
    )

    # --------------------------------------------------------
    # Long only + fully invested
    # --------------------------------------------------------

    weights = _normalize_weights(
        weights.values
    )

    weights = pd.Series(
        weights,
        index=assets
    )

    # --------------------------------------------------------
    # Industry constraint
    # --------------------------------------------------------

    if industry_data is not None:

        weights = _apply_industry_constraint(
            weights,
            industry_data,
            max_industry_weight
        )

    # --------------------------------------------------------
    # Turnover constraint
    # --------------------------------------------------------

    if (
        current_weights is not None
        and max_turnover is not None
    ):

        weights = _apply_turnover_constraint(
            weights,
            current_weights,
            max_turnover
        )

    # --------------------------------------------------------
    # Final normalization
    # --------------------------------------------------------

    weights = _normalize_weights(
        weights.values
    )

    weights = pd.Series(
        weights,
        index=assets,
        name="weight"
    )

    return weights


# ============================================================
# 1. EQUAL WEIGHT
# ============================================================

def equal_weight(
    expected_returns,
    covariance=None,
    industry_data=None,
    max_industry_weight=0.35,
    current_weights=None,
    max_turnover=None
):
    """
    Equal-weight portfolio.
    """

    expected_returns = pd.Series(
        expected_returns
    )

    n_assets = len(
        expected_returns
    )

    if n_assets == 0:
        raise ValueError(
            "No assets supplied."
        )

    weights = np.ones(
        n_assets
    ) / n_assets

    weights = pd.Series(
        weights,
        index=expected_returns.index
    )

    weights = _apply_constraints(
        weights,
        expected_returns.index,
        industry_data,
        max_industry_weight,
        current_weights,
        max_turnover
    )

    return weights


# ============================================================
# NUMERICAL GRADIENT
# ============================================================

def _numerical_gradient(
    objective,
    weights,
    epsilon=1e-6
):
    """
    Calculate numerical gradient.
    """

    weights = np.asarray(
        weights,
        dtype=float
    )

    gradient = np.zeros_like(
        weights
    )

    base_value = objective(
        weights
    )

    for i in range(
        len(weights)
    ):

        test_weights = (
            weights.copy()
        )

        test_weights[i] += epsilon

        gradient[i] = (
            objective(test_weights)
            - base_value
        ) / epsilon

    return gradient


# ============================================================
# PROJECTED GRADIENT DESCENT
# ============================================================

def _gradient_descent(
    objective,
    initial_weights,
    learning_rate=0.01,
    max_iterations=5000,
    tolerance=1e-10
):
    """
    Generic projected gradient descent.

    Long-only constraint is enforced after
    every optimization step.
    """

    weights = _normalize_weights(
        initial_weights
    )

    previous_objective = (
        objective(weights)
    )

    for _ in range(
        max_iterations
    ):

        gradient = _numerical_gradient(
            objective,
            weights
        )

        if not np.all(
            np.isfinite(gradient)
        ):
            break

        step = learning_rate

        improved = False

        for _ in range(20):

            candidate = (
                weights
                - step * gradient
            )

            candidate = _normalize_weights(
                candidate
            )

            candidate_objective = (
                objective(candidate)
            )

            if (
                candidate_objective
                <= previous_objective
            ):

                improved = True
                break

            step *= 0.5

        if not improved:
            break

        change = np.max(
            np.abs(
                candidate
                - weights
            )
        )

        weights = candidate

        if (
            abs(
                previous_objective
                - candidate_objective
            )
            < tolerance
        ):
            break

        if change < tolerance:
            break

        previous_objective = (
            candidate_objective
        )

    return weights


# ============================================================
# 2. MINIMUM VARIANCE
# ============================================================

def minimum_variance(
    expected_returns=None,
    covariance=None,
    industry_data=None,
    max_industry_weight=0.35,
    current_weights=None,
    max_turnover=None
):
    """
    Minimum variance portfolio.

    Objective:

        minimize w'Σw
    """

    covariance = pd.DataFrame(
        covariance
    )

    assets = covariance.index

    sigma = covariance.values

    n_assets = len(
        assets
    )

    initial_weights = (
        np.ones(n_assets)
        / n_assets
    )

    def objective(weights):

        return (
            weights
            @ sigma
            @ weights
        )

    weights = _gradient_descent(
        objective=objective,
        initial_weights=initial_weights,
        learning_rate=0.01,
        max_iterations=5000
    )

    weights = pd.Series(
        weights,
        index=assets
    )

    weights = _apply_constraints(
        weights,
        assets,
        industry_data,
        max_industry_weight,
        current_weights,
        max_turnover
    )

    return weights


# ============================================================
# 3. MAXIMUM SHARPE
# ============================================================

def maximum_sharpe(
    expected_returns,
    covariance,
    risk_free_rate=0.0,
    industry_data=None,
    max_industry_weight=0.35,
    current_weights=None,
    max_turnover=None
):
    """
    Maximum Sharpe ratio portfolio.
    """

    expected_returns, covariance = (
        _validate_inputs(
            expected_returns,
            covariance
        )
    )

    assets = expected_returns.index

    mu = expected_returns.values

    sigma = covariance.values

    n_assets = len(
        assets
    )

    initial_weights = (
        np.ones(n_assets)
        / n_assets
    )

    def objective(weights):

        portfolio_ret = (
            weights @ mu
        )

        variance = (
            weights
            @ sigma
            @ weights
        )

        volatility = np.sqrt(
            max(
                variance,
                1e-16
            )
        )

        sharpe = (
            portfolio_ret
            - risk_free_rate
        ) / volatility

        return -sharpe

    weights = _gradient_descent(
        objective=objective,
        initial_weights=initial_weights,
        learning_rate=0.01,
        max_iterations=5000
    )

    weights = pd.Series(
        weights,
        index=assets
    )

    weights = _apply_constraints(
        weights,
        assets,
        industry_data,
        max_industry_weight,
        current_weights,
        max_turnover
    )

    return weights


# ============================================================
# RISK CONTRIBUTION
# ============================================================

def _risk_contributions(
    weights,
    covariance
):
    """
    Calculate portfolio risk contribution.
    """

    weights = np.asarray(
        weights,
        dtype=float
    )

    covariance = np.asarray(
        covariance,
        dtype=float
    )

    variance = (
        weights
        @ covariance
        @ weights
    )

    volatility = np.sqrt(
        max(
            variance,
            1e-16
        )
    )

    marginal_risk = (
        covariance
        @ weights
    ) / volatility

    component_risk = (
        weights
        * marginal_risk
    )

    risk_contribution = (
        component_risk
        / volatility
    )

    return (
        volatility,
        marginal_risk,
        component_risk,
        risk_contribution
    )


# ============================================================
# 4. RISK PARITY
# ============================================================

def risk_parity(
    expected_returns=None,
    covariance=None,
    industry_data=None,
    max_industry_weight=0.35,
    current_weights=None,
    max_turnover=None
):
    """
    Risk parity portfolio.

    Objective:

        minimize Σ(RC_i - target_i)^2
    """

    covariance = pd.DataFrame(
        covariance
    )

    assets = covariance.index

    sigma = covariance.values

    n_assets = len(
        assets
    )

    target = (
        np.ones(n_assets)
        / n_assets
    )

    initial_weights = target.copy()

    def objective(weights):

        _, _, _, risk_contribution = (
            _risk_contributions(
                weights,
                sigma
            )
        )

        return np.sum(
            (
                risk_contribution
                - target
            ) ** 2
        )

    weights = _gradient_descent(
        objective=objective,
        initial_weights=initial_weights,
        learning_rate=0.05,
        max_iterations=2000
    )

    weights = pd.Series(
        weights,
        index=assets
    )

    weights = _apply_constraints(
        weights,
        assets,
        industry_data,
        max_industry_weight,
        current_weights,
        max_turnover
    )

    return weights


# ============================================================
# CORRELATION MATRIX
# ============================================================

def _covariance_to_correlation(
    covariance
):
    """
    Convert covariance matrix to
    correlation matrix.
    """

    covariance = np.asarray(
        covariance,
        dtype=float
    )

    standard_deviation = np.sqrt(
        np.maximum(
            np.diag(covariance),
            1e-16
        )
    )

    correlation = (
        covariance
        / np.outer(
            standard_deviation,
            standard_deviation
        )
    )

    correlation = np.clip(
        correlation,
        -1.0,
        1.0
    )

    np.fill_diagonal(
        correlation,
        1.0
    )

    return correlation


# ============================================================
# HRP DISTANCE
# ============================================================

def _correlation_distance(
    correlation
):
    """
    Convert correlation to HRP distance.

        d(i,j)
        =
        sqrt(
            0.5 * (1 - rho)
        )
    """

    distance = np.sqrt(
        np.maximum(
            0.0,
            0.5
            * (
                1.0
                - correlation
            )
        )
    )

    np.fill_diagonal(
        distance,
        0.0
    )

    return distance


# ============================================================
# HIERARCHICAL CLUSTERING
# ============================================================

def _hierarchical_clustering(
    distance
):
    """
    Simple agglomerative clustering.

    Average linkage is calculated manually
    using NumPy.
    """

    n_assets = distance.shape[0]

    clusters = [
        [i]
        for i in range(n_assets)
    ]

    while len(clusters) > 1:

        best_i = 0
        best_j = 1

        best_distance = np.inf

        for i in range(
            len(clusters)
        ):

            for j in range(
                i + 1,
                len(clusters)
            ):

                pair_distances = []

                for asset_i in clusters[i]:

                    for asset_j in clusters[j]:

                        pair_distances.append(
                            distance[
                                asset_i,
                                asset_j
                            ]
                        )

                cluster_distance = (
                    np.mean(
                        pair_distances
                    )
                )

                if (
                    cluster_distance
                    < best_distance
                ):

                    best_distance = (
                        cluster_distance
                    )

                    best_i = i
                    best_j = j

        merged_cluster = (
            clusters[best_i]
            + clusters[best_j]
        )

        new_clusters = []

        for index, cluster in enumerate(
            clusters
        ):

            if (
                index != best_i
                and index != best_j
            ):

                new_clusters.append(
                    cluster
                )

        new_clusters.append(
            merged_cluster
        )

        clusters = new_clusters

    return clusters[0]


# ============================================================
# HRP QUASI DIAGONAL ORDER
# ============================================================

def _quasi_diagonal_order(
    distance
):
    """
    Obtain hierarchical asset ordering.
    """

    return _hierarchical_clustering(
        distance
    )


# ============================================================
# HRP CLUSTER VARIANCE
# ============================================================

def _cluster_variance(
    covariance,
    cluster
):
    """
    Calculate inverse-variance cluster risk.
    """

    cluster = list(
        cluster
    )

    cluster_covariance = (
        covariance[
            np.ix_(
                cluster,
                cluster
            )
        ]
    )

    diagonal = np.diag(
        cluster_covariance
    )

    diagonal = np.maximum(
        diagonal,
        1e-16
    )

    inverse_variance = (
        1.0 / diagonal
    )

    inverse_variance /= (
        inverse_variance.sum()
    )

    variance = (
        inverse_variance
        @ cluster_covariance
        @ inverse_variance
    )

    return float(
        variance
    )


# ============================================================
# 5. HIERARCHICAL RISK PARITY
# ============================================================

def hierarchical_risk_parity(
    expected_returns=None,
    covariance=None,
    industry_data=None,
    max_industry_weight=0.35,
    current_weights=None,
    max_turnover=None
):
    """
    Hierarchical Risk Parity.

    Steps:

        1. Covariance
        2. Correlation
        3. Distance
        4. Hierarchical clustering
        5. Quasi-diagonal ordering
        6. Recursive bisection
        7. Risk-based allocation
    """

    covariance = pd.DataFrame(
        covariance
    )

    assets = covariance.index

    sigma = covariance.values

    n_assets = len(
        assets
    )

    if n_assets == 1:

        return pd.Series(
            [1.0],
            index=assets,
            name="weight"
        )

    correlation = (
        _covariance_to_correlation(
            sigma
        )
    )

    distance = (
        _correlation_distance(
            correlation
        )
    )

    order = (
        _quasi_diagonal_order(
            distance
        )
    )

    weights = np.ones(
        n_assets
    )

    clusters = [
        order
    ]

    while len(clusters) > 0:

        next_clusters = []

        for cluster in clusters:

            if len(cluster) <= 1:
                continue

            split = (
                len(cluster)
                // 2
            )

            left = cluster[
                :split
            ]

            right = cluster[
                split:
            ]

            left_variance = (
                _cluster_variance(
                    sigma,
                    left
                )
            )

            right_variance = (
                _cluster_variance(
                    sigma,
                    right
                )
            )

            total_variance = (
                left_variance
                + right_variance
            )

            if total_variance <= 0:

                left_allocation = 0.5

            else:

                left_allocation = (
                    1.0
                    - (
                        left_variance
                        / total_variance
                    )
                )

            right_allocation = (
                1.0
                - left_allocation
            )

            for index in left:

                weights[index] *= (
                    left_allocation
                )

            for index in right:

                weights[index] *= (
                    right_allocation
                )

            next_clusters.append(
                left
            )

            next_clusters.append(
                right
            )

        clusters = next_clusters

    weights = (
        weights
        / weights.sum()
    )

    weights = pd.Series(
        weights,
        index=assets
    )

    weights = _apply_constraints(
        weights,
        assets,
        industry_data,
        max_industry_weight,
        current_weights,
        max_turnover
    )

    return weights


# ============================================================
# 6. BLACK-LITTERMAN
# ============================================================

def black_litterman(
    expected_returns=None,
    covariance=None,
    market_weights=None,
    views=None,
    view_matrix=None,
    tau=0.05,
    view_confidence=None,
    risk_aversion=2.5,
    industry_data=None,
    max_industry_weight=0.35,
    current_weights=None,
    max_turnover=None
):
    """
    Black-Litterman model.

    Equilibrium return:

        Pi = delta * Sigma * W

    Posterior return:

        BL = M^-1 [
            M_tau^-1 Pi
            +
            P' Omega^-1 Q
        ]

    The posterior expected returns are then
    passed to the Maximum Sharpe optimizer.
    """

    covariance = pd.DataFrame(
        covariance
    )

    assets = covariance.index

    sigma = covariance.values

    # If no explicit BL inputs are supplied, use a neutral default:
    # equal market weights and zero views. This keeps BL compatible
    # with the generic backtest optimizer interface without inventing
    # directional investment views.
    if market_weights is None:
        market_weights = pd.Series(
            np.ones(len(assets)) / len(assets),
            index=assets,
            dtype=float
        )
    else:
        market_weights = pd.Series(
            market_weights,
            index=assets,
            dtype=float
        )

    market_weights = (
        market_weights
        / market_weights.sum()
    )

    if views is None or view_matrix is None:
        views = np.zeros(len(assets))
        P = np.eye(len(assets))
    else:
        views = np.asarray(
            views,
            dtype=float
        )

        P = np.asarray(
            view_matrix,
            dtype=float
        )

    if P.ndim == 1:

        P = P.reshape(
            1,
            -1
        )

    if views.ndim == 0:

        views = views.reshape(
            1
        )

    if P.shape[1] != len(
        assets
    ):

        raise ValueError(
            "View matrix columns must "
            "equal number of assets."
        )

    if P.shape[0] != len(
        views
    ):

        raise ValueError(
            "Number of views must equal "
            "number of rows in view matrix."
        )

    # --------------------------------------------------------
    # EQUILIBRIUM RETURNS
    # --------------------------------------------------------

    pi = (
        risk_aversion
        * sigma
        @ market_weights.values
    )

    # --------------------------------------------------------
    # VIEW UNCERTAINTY
    # --------------------------------------------------------

    tau_sigma = (
        tau * sigma
    )

    omega_base = (
        P
        @ tau_sigma
        @ P.T
    )

    if view_confidence is None:

        omega = omega_base

    else:

        confidence = np.asarray(
            view_confidence,
            dtype=float
        )

        if confidence.ndim == 0:

            confidence = np.repeat(
                confidence,
                len(views)
            )

        confidence = np.clip(
            confidence,
            1e-6,
            1.0
        )

        omega = np.diag(
            np.diag(
                omega_base
            )
            / confidence
        )

    # --------------------------------------------------------
    # POSTERIOR
    # --------------------------------------------------------

    tau_sigma_inverse = (
        np.linalg.pinv(
            tau_sigma
        )
    )

    omega_inverse = (
        np.linalg.pinv(
            omega
        )
    )

    posterior_precision = (
        tau_sigma_inverse
        +
        P.T
        @ omega_inverse
        @ P
    )

    posterior_covariance = (
        np.linalg.pinv(
            posterior_precision
        )
    )

    posterior_returns = (
        posterior_covariance
        @ (
            tau_sigma_inverse
            @ pi
            +
            P.T
            @ omega_inverse
            @ views
        )
    )

    posterior_returns = pd.Series(
        posterior_returns,
        index=assets
    )

    # --------------------------------------------------------
    # OPTIMIZE POSTERIOR RETURNS
    # --------------------------------------------------------

    weights = maximum_sharpe(
        expected_returns=posterior_returns,
        covariance=covariance,
        risk_free_rate=0.0,
        industry_data=industry_data,
        max_industry_weight=max_industry_weight,
        current_weights=current_weights,
        max_turnover=max_turnover
    )

    return weights


# ============================================================
# RISK CONTRIBUTION TABLE
# ============================================================

def risk_contribution_table(
    weights,
    covariance
):
    """
    Create a risk contribution DataFrame.

    Columns:

        weight
        marginal_risk
        component_risk
        risk_contribution
    """

    weights = pd.Series(
        weights,
        dtype=float
    )

    covariance = pd.DataFrame(
        covariance
    )

    covariance = covariance.loc[
        weights.index,
        weights.index
    ]

    (
        volatility,
        marginal_risk,
        component_risk,
        risk_contribution
    ) = _risk_contributions(
        weights.values,
        covariance.values
    )

    result = pd.DataFrame(
        {
            "weight": weights.values,
            "marginal_risk": marginal_risk,
            "component_risk": component_risk,
            "risk_contribution":
                risk_contribution
        },
        index=weights.index
    )

    return result


# ============================================================
# OPTIMIZATION SUMMARY
# ============================================================

def optimization_summary(
    weights,
    expected_returns,
    covariance,
    risk_free_rate=0.0
):
    """
    Generate portfolio optimization metrics.
    """

    expected_returns, covariance = (
        _validate_inputs(
            expected_returns,
            covariance
        )
    )

    weights = _validate_weights(
        weights,
        expected_returns.index
    )

    expected_return = (
        portfolio_return(
            weights.values,
            expected_returns.values
        )
    )

    volatility = (
        portfolio_volatility(
            weights.values,
            covariance.values
        )
    )

    sharpe = (
        portfolio_sharpe(
            weights.values,
            expected_returns.values,
            covariance.values,
            risk_free_rate
        )
    )

    return {
        "expected_return":
            expected_return,

        "volatility":
            volatility,

        "sharpe_ratio":
            sharpe,

        "weight_sum":
            float(weights.sum()),

        "max_weight":
            float(weights.max()),

        "min_weight":
            float(weights.min())
    }


# ============================================================
# RUN ALL OPTIMIZERS
# ============================================================

def optimize_all(
    expected_returns,
    covariance,
    market_weights=None,
    views=None,
    view_matrix=None,
    view_confidence=None,
    tau=0.05,
    risk_aversion=2.5,
    risk_free_rate=0.0,
    industry_data=None,
    max_industry_weight=0.35,
    current_weights=None,
    max_turnover=None
):
    """
    Run all portfolio optimization methods.

    Returns
    -------
    dict
        Dictionary containing portfolio weights.
    """

    expected_returns, covariance = (
        _validate_inputs(
            expected_returns,
            covariance
        )
    )

    results = {}

    # --------------------------------------------------------
    # 1. EQUAL WEIGHT
    # --------------------------------------------------------

    results["equal_weight"] = (
        equal_weight(
            expected_returns=expected_returns,
            covariance=covariance,
            industry_data=industry_data,
            max_industry_weight=max_industry_weight,
            current_weights=current_weights,
            max_turnover=max_turnover
        )
    )

    # --------------------------------------------------------
    # 2. MINIMUM VARIANCE
    # --------------------------------------------------------

    results["minimum_variance"] = (
        minimum_variance(
            expected_returns=expected_returns,
            covariance=covariance,
            industry_data=industry_data,
            max_industry_weight=max_industry_weight,
            current_weights=current_weights,
            max_turnover=max_turnover
        )
    )

    # --------------------------------------------------------
    # 3. MAXIMUM SHARPE
    # --------------------------------------------------------

    results["maximum_sharpe"] = (
        maximum_sharpe(
            expected_returns=expected_returns,
            covariance=covariance,
            risk_free_rate=risk_free_rate,
            industry_data=industry_data,
            max_industry_weight=max_industry_weight,
            current_weights=current_weights,
            max_turnover=max_turnover
        )
    )

    # --------------------------------------------------------
    # 4. RISK PARITY
    # --------------------------------------------------------

    results["risk_parity"] = (
        risk_parity(
            expected_returns=expected_returns,
            covariance=covariance,
            industry_data=industry_data,
            max_industry_weight=max_industry_weight,
            current_weights=current_weights,
            max_turnover=max_turnover
        )
    )

    # --------------------------------------------------------
    # 5. HRP
    # --------------------------------------------------------

    results["hrp"] = (
        hierarchical_risk_parity(
            expected_returns=expected_returns,
            covariance=covariance,
            industry_data=industry_data,
            max_industry_weight=max_industry_weight,
            current_weights=current_weights,
            max_turnover=max_turnover
        )
    )

    # --------------------------------------------------------
    # 6. BLACK-LITTERMAN
    # --------------------------------------------------------

    if (
        market_weights is not None
        and views is not None
        and view_matrix is not None
    ):

        results["black_litterman"] = (
            black_litterman(
                expected_returns=expected_returns,
                covariance=covariance,
                market_weights=market_weights,
                views=views,
                view_matrix=view_matrix,
                tau=tau,
                view_confidence=view_confidence,
                risk_aversion=risk_aversion,
                industry_data=industry_data,
                max_industry_weight=max_industry_weight,
                current_weights=current_weights,
                max_turnover=max_turnover
            )
        )

    return results